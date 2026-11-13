"""Window/lease regressions execute production methods with CPU endpoints."""

import ast
import json
import logging
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, call

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4] / "python/sglang/srt"
CACHE = ROOT / "mem_cache/storage/flexkv"


def production_class(filename, name, base=object, **extra):
    path = CACHE / filename
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    node.bases = [ast.Name(id="Base", ctx=ast.Load())]
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_request_key", "_namespace_kwargs")
    ]
    scope = dict(
        Base=base,
        torch=torch,
        json=json,
        logger=logging.getLogger(__name__),
        _request_key=lambda h: json.dumps([h.rid, h.attempt_id], separators=(",", ":")),
        _namespace_kwargs=lambda *a: {},
        MatchPrefixParams=lambda **kw: NS(**kw),
        _RestoreLease=NS,
        FlexKVMode=NS(IP="ip", MP="mp"),
        **extra,
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *functions,
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope[name]


Lifecycle = production_class("flexkv_cache_lifecycle.py", "FlexKVCacheLifecycleMixin")


def handle():
    return NS(rid="same", attempt_id=3)


@pytest.mark.parametrize(
    "chunked,enabled", [(False, False), (False, True), (True, False), (True, True)]
)
def test_window_requires_an_active_chunked_runtime(chunked, enabled):
    cache = Lifecycle()
    cache.flexkv_connector = NS(_chunked_prefetch=chunked, _prefetch_enabled=enabled)
    if chunked and enabled:
        cache._configure_prefetch_window(NS(flexkv_prefetch_window_size=4))
        assert cache.defer_host_lookup_until_admission
    else:
        with pytest.raises(ValueError, match="chunked"):
            cache._configure_prefetch_window(NS(flexkv_prefetch_window_size=4))
    cache._configure_prefetch_window(NS(flexkv_prefetch_window_size=0))
    assert not cache.defer_host_lookup_until_admission


@pytest.mark.parametrize("hit", [4, 2, 0])
def test_held_get_requires_the_complete_admitted_span(hit):
    cache = Lifecycle()
    cache.defer_host_lookup_until_admission = True
    cache.flexkv_connector = Mock()
    cache.flexkv_connector.lookup_kv.return_value = (7, hit)
    assert cache._acquire_window_restore_get(
        list(range(16)), 4, 4, "attempt", "same", {}
    ) is (hit == 4)
    args, kwargs = cache.flexkv_connector.lookup_kv.call_args
    assert args[0] == list(range(8))
    assert args[1].tolist() == [False] * 4 + [True] * 4
    assert kwargs == dict(rid="attempt", sglang_req_id="same", cpu_only=True)
    assert cache.flexkv_connector.release_pending.call_count == int(hit != 4)


class Key(list):
    extra_key = None
    cache_salt = None
    is_bigram = False

    @property
    def token_ids(self):
        return self

    def child_key(self, page_size):
        return tuple(self[:page_size])

    def page_aligned(self, page_size):
        return self[: len(self) // page_size * page_size]

    def __getitem__(self, key):
        value = super().__getitem__(key)
        return Key(value) if isinstance(key, slice) else value

    def raw_token_ids(self):
        return list(self)


@pytest.mark.parametrize("kind", ["ordinary", "hybrid"])
@pytest.mark.parametrize("layerwise", [False, True])
@pytest.mark.parametrize(
    "allocation,hit", [(False, 4), (True, 4), (True, 2), (True, 0), (True, None)]
)
def test_restore_preserves_result_lease_until_allocation_and_revalidates(
    kind, layerwise, allocation, hit
):
    node = NS(priority=0, children={})

    class Base(Lifecycle):
        def match_prefix(self, params):
            return NS(last_device_node=node)

    cache_type = production_class(
        "flexkv_radix_cache.py"
        if kind == "ordinary"
        else "flexkv_hybrid_radix_cache.py",
        "FlexKVRadixCache" if kind == "ordinary" else "FlexKVHybridRadixCache",
        base=Base,
        TreeNode=lambda **kw: NS(**kw, children={}),
    )
    cache = cache_type.__new__(cache_type)
    cache.defer_host_lookup_until_admission = True
    cache._init_restore_state()
    cache._defer_duplicate_restores = False
    cache.path_device_indices = lambda n: torch.empty(0, dtype=torch.int64)
    cache.supports_swa = lambda: False
    cache._node_lock = None
    cache.page_size = 1
    cache.evictable_size_ = 0
    cache._update_leaf_status = Mock()
    cache._record_store_event = Mock()
    cache.kv_events = Mock()
    slots = torch.arange(4, dtype=torch.int64)
    allocator = Mock()
    allocator.available_size.return_value = 16
    allocator.alloc.return_value = slots if allocation else None
    cache.token_to_kv_pool_allocator = allocator
    cache._alloc_restore_slots = lambda *a: slots if allocation else None
    connector, events = Mock(enable_layerwise=layerwise), []
    cache.flexkv_connector = connector
    connector.sync_load_ready.side_effect = lambda ready: events.append("vote") or ready

    def lookup(*args, **kwargs):
        events.append("held_get")
        if hit is None:
            raise RuntimeError("lookup unavailable")
        return (9, hit)

    def load(*args, **kwargs):
        events.append("load")
        return 4

    connector.lookup_kv.side_effect = lookup
    connector.retrieve_kv.side_effect = load
    connector.start_load_kv_layerwise.side_effect = lambda *a, **kw: (load(), None)
    req = NS(
        rid="same",
        cache_request_handle=handle(),
        full_untruncated_fill_ids=list(range(8)),
        extra_key=None,
        cache_salt=None,
        last_node=node,
        kv=NS(cache_protected_len=0),
    )
    cache._load_markers = {'["same",3]': NS(device_length=0)}

    def run():
        if kind == "hybrid":
            return cache.init_load_back(NS(req=req, host_hit_length=4))
        return cache._allocate_and_load(
            key=Key(range(4)),
            value_numel=0,
            uncached_len=4,
            last_node=node,
            tracking_rid='["same",3]',
            sglang_req_id="same",
            load_fn=load,
            request_owned_req=req if layerwise else None,
        )

    if allocation and hit is None:
        with pytest.raises(RuntimeError, match="lookup unavailable"):
            run()
    else:
        result = run()
        if allocation and hit == 4:
            assert events == ["vote", "held_get", "load"]
            assert result[0] == 4 if kind == "hybrid" else result[0].numel() == 4
            if layerwise or kind == "hybrid":
                assert req.pending_restore_slots is slots
            else:
                assert not cache._restore_leases
            allocator.free.assert_not_called()
            return
    assert events == (["vote"] if not allocation else ["vote", "held_get"])
    assert not cache._restore_leases
    if allocation:
        allocator.free.assert_called_once()
        assert allocator.free.call_args.args[0] is slots
    else:
        connector.lookup_kv.assert_not_called()


def test_admitted_cleanup_uses_attempt_identity_and_waits_for_flush():
    ordinary = production_class(
        "flexkv_radix_cache.py", "FlexKVRadixCache", base=Lifecycle
    )
    cache = ordinary.__new__(ordinary)
    cache._mode = "ip"
    cache.defer_host_lookup_until_admission = True
    cache.flexkv_connector = Mock()
    cache._load_markers = {}
    req = NS(cache_request_handle=handle())
    cache.launch_pending_loads([req])
    assert cache.flexkv_connector.mock_calls == [
        call.flush_layerwise_loads(['["same",3]']),
        call.cancel_prefetch('["same",3]'),
    ]
    cache.flexkv_connector.reset_mock()
    cache.flexkv_connector.flush_layerwise_loads.side_effect = RuntimeError(
        "unknown launch"
    )
    with pytest.raises(RuntimeError, match="unknown launch"):
        cache.launch_pending_loads([req])
    cache.flexkv_connector.cancel_prefetch.assert_not_called()


@pytest.mark.parametrize("kind", ["ordinary", "hybrid"])
@pytest.mark.parametrize("window", [False, True])
def test_admission_host_probe_preserves_result_lease(kind, window):
    class Result(NS):
        def _replace(self, **kwargs):
            return Result(**(vars(self) | kwargs))

    adapter = production_class(
        "flexkv_radix_cache.py"
        if kind == "ordinary"
        else "flexkv_hybrid_radix_cache.py",
        "FlexKVRadixCache" if kind == "ordinary" else "FlexKVHybridRadixCache",
        base=Lifecycle,
        RadixKey=lambda ids, *a, **kw: Key(ids),
        _LoadBackMarker=lambda **kw: NS(**kw),
        _LoadMarker=lambda **kw: NS(**kw),
        MatchResult=lambda **kw: Result(**kw),
    )
    cache = adapter.__new__(adapter)
    cache.defer_host_lookup_until_admission = window
    cache.flexkv_connector = Mock()
    cache.flexkv_connector.lookup_kv.return_value = (9, 4)
    cache._init_restore_state()
    cache._load_markers = {}
    cache.disable, cache.page_size = False, 4
    node = object()
    base = Result(device_prefix_len=0, last_device_node=node)
    cache._inner_cache = NS(match_prefix=lambda params: base)
    req = NS(rid="same", cache_request_handle=handle())
    if kind == "ordinary":
        cache._match_host_prefix(Key(range(4)), base, torch.empty(0), node, req)
    else:
        cache.match_prefix(NS(req=req, key=Key(range(4))))
    assert cache.flexkv_connector.lookup_kv.call_args.kwargs["rid"] == (
        None if window else '["same",3]'
    )


@pytest.mark.parametrize(
    "size,enabled,backend,disabled,valid",
    [
        (-1, True, None, False, False),
        (4, False, None, False, False),
        (4, True, "lmcache", False, False),
        (4, True, None, True, False),
        (4, True, None, False, True),
        (4, False, "flexkv", False, True),
        (0, False, None, False, True),
    ],
)
def test_window_argument_validation(size, enabled, backend, disabled, valid):
    path = ROOT / "arg_groups/validation_hook.py"
    function = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "check_flexkv_prefetch_window"
    )
    scope = {"resolving_view": lambda value: value, "Any": object}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope
    )
    config = NS(
        flexkv_prefetch_window_size=size,
        enable_flexkv=enabled,
        radix_cache_backend=backend,
        disable_radix_cache=disabled,
    )
    if valid:
        scope["check_flexkv_prefetch_window"](config)
    else:
        with pytest.raises(ValueError):
            scope["check_flexkv_prefetch_window"](config)


@pytest.mark.parametrize("window", [False, True])
@pytest.mark.parametrize("scheduling", [False, True])
def test_priority_probes_do_not_acquire_tail_host_references(window, scheduling):
    path = ROOT / "mem_cache/common.py"
    function = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "match_kv_cache"
    )
    scope = dict(
        MatchPrefixParams=lambda **kw: NS(**kw),
        _req_radix_key=lambda *a: None,
        envs=NS(SGLANG_RADIX_FORCE_MISS=NS(get=lambda: False)),
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            function,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    result = NS(
        device_prefix_len=2,
        last_device_node=None,
        last_host_node=None,
        best_match_node=None,
        host_hit_length=0,
        mamba_host_hit_length=0,
        window_host_hit_lengths={},
        swa_branching_seqlen=None,
        mamba_branching_seqlen=None,
        cache_protected_len=None,
    )
    cache = NS(
        defer_host_lookup_until_admission=window, match_prefix=Mock(return_value=result)
    )
    req = NS(
        origin_input_ids=[1, 2, 3],
        output_ids=[],
        _compute_max_prefix_len=lambda n: n - 1,
    )
    scope["match_kv_cache"](req, cache, for_scheduling=scheduling)
    assert cache.match_prefix.call_args.args[0].req is (
        None if window and scheduling else req
    )
