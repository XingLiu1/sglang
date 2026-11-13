"""Queue-pressure regressions for bounded speculative KV residency."""

import argparse
import ast
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, call

import msgspec
import pytest

from sglang.srt.arg_groups.arg_utils import A, add_cli_args_from_dataclass
from sglang.srt.managers.flexkv_prefetch_window import FlexKVPrefetchWindow
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def requests(n):
    return [NS(rid=str(i), cache_request_handle=(str(i), 0)) for i in range(n)]


def cache():
    return NS(
        has_uncommitted_restore=lambda req: False,
        prefetch_request=Mock(),
        terminate_prefetch=Mock(),
    )


def test_large_backlog_keeps_the_same_window_until_consumed():
    reqs, tree = requests(1000), cache()
    window = FlexKVPrefetchWindow(4)
    for _ in range(10):
        window.update(reqs, tree)
    assert tree.prefetch_request.call_args_list == [call(r) for r in reqs[:4]]
    assert all(window.contains(r) == (i < 4) for i, r in enumerate(reqs))
    tree.terminate_prefetch.assert_not_called()

    # Completion alone does not move the window. Actual admission does.
    window.update(reqs[2:], tree)
    assert tree.prefetch_request.call_args_list == [call(r) for r in reqs[:6]]
    tree.terminate_prefetch.assert_not_called()
    assert not window.contains(reqs[0])


def test_priority_change_returns_tail_leases_before_starting_new_head():
    reqs, tree = requests(4), cache()
    window = FlexKVPrefetchWindow(2)
    window.update(reqs, tree)
    events = Mock()
    events.attach_mock(tree.prefetch_request, "start")
    events.attach_mock(tree.terminate_prefetch, "cancel")
    window.update([reqs[3], reqs[0], reqs[1], reqs[2]], tree)
    assert events.mock_calls == [call.cancel(("1", 0)), call.start(reqs[3])]
    assert window.contains(reqs[0]) and window.contains(reqs[3])


def test_active_restore_keeps_ownership_and_does_not_block_window():
    reqs, tree = requests(4), cache()
    window = FlexKVPrefetchWindow(2)
    window.update(reqs, tree)
    tree.has_uncommitted_restore = lambda req: req is reqs[0]
    window.update(reqs, tree)
    tree.terminate_prefetch.assert_not_called()
    assert tree.prefetch_request.call_args_list == [call(r) for r in reqs[:3]]
    assert not window.contains(reqs[0])
    assert window.contains(reqs[1]) and window.contains(reqs[2])


def test_ineligible_lora_requests_do_not_occupy_all_slots():
    reqs, tree = requests(5), cache()
    window = FlexKVPrefetchWindow(2)
    window.update(reqs, tree, eligible=lambda req: int(req.rid) >= 2)
    assert tree.prefetch_request.call_args_list == [call(r) for r in reqs[2:4]]


def test_abort_and_rid_reuse_do_not_inherit_old_admission():
    old, other = requests(2)
    tree, window = cache(), FlexKVPrefetchWindow(1)
    window.update([old, other], tree)
    replacement = NS(rid=old.rid, cache_request_handle=(old.rid, 1))
    assert not window.contains(replacement)
    # The abort handler already cleaned up old's transfer. The new Req starts.
    window.update([replacement, other], tree)
    assert tree.prefetch_request.call_args_list == [call(old), call(replacement)]
    assert window.contains(replacement) and not window.contains(old)
    window.update([], tree)
    assert not window.contains(replacement)
    tree.terminate_prefetch.assert_not_called()


@pytest.mark.parametrize("size", [0, -1])
def test_invalid_window_size(size):
    with pytest.raises(ValueError, match="positive"):
        FlexKVPrefetchWindow(size)


def test_retraction_of_same_request_starts_a_distinct_attempt():
    req = requests(1)[0]
    tree, window = cache(), FlexKVPrefetchWindow(1)
    window.update([req], tree)
    old = req.cache_request_handle
    req.cache_request_handle = (req.rid, 1)
    assert not window.contains(req)
    window.update([req], tree)
    tree.terminate_prefetch.assert_called_once_with(old)
    assert tree.prefetch_request.call_count == 2


def test_window_size_is_registered_with_the_production_cli_generator():
    root = Path(__file__).resolve().parents[4] / "python/sglang/srt/arg_groups"
    field = next(
        n
        for n in ast.walk(ast.parse((root / "fields/memory.py").read_text()))
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name)
        and n.target.id == "flexkv_prefetch_window_size"
    )
    cls = ast.ClassDef(
        name="WindowArgs",
        bases=[ast.Name(id="Struct", ctx=ast.Load())],
        keywords=[],
        body=[field],
        decorator_list=[],
    )
    scope = {"Struct": msgspec.Struct, "A": A}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
            "memory.py",
            "exec",
        ),
        scope,
    )
    parser = argparse.ArgumentParser()
    add_cli_args_from_dataclass(parser, scope["WindowArgs"])
    assert parser.parse_args([]).flexkv_prefetch_window_size == 0
    assert (
        parser.parse_args(
            ["--flexkv-prefetch-window-size", "8"]
        ).flexkv_prefetch_window_size
        == 8
    )
    assert '"flexkv_prefetch_window_size"' in (root / "field_order.py").read_text()
