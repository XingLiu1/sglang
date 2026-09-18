"""FlexKV on a PD decode server: store draining in the decode loop, no host lookups."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _load_flexkv_radix_cache_module():
    connector_name = "flexkv.integration.sglang.connector"
    connector_stub = ModuleType(connector_name)
    connector_stub.FlexKVConnector = object
    connector_stub.FlexKVHostReleaseShim = object
    module_path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/mem_cache/storage/flexkv/flexkv_radix_cache.py"
    )
    module_name = "_flexkv_radix_cache_decode_ut"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        with patch.dict(sys.modules, {connector_name: connector_stub}):
            spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


class TestFlexKVDecodeWiring(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(
                model_path="dummy",
                disaggregation_mode="decode",
                disaggregation_transfer_backend="nixl",
                enable_flexkv=True,
            ),
            role="tokenizer",
        )

    def _bare_decode_scheduler(self, *, enable_flexkv: bool) -> Scheduler:
        sched = Scheduler.__new__(Scheduler)
        sched.scheduler_stage_metrics = None
        sched.enable_decode_hicache = False
        sched.enable_flexkv = enable_flexkv
        sched.enable_hisparse = False
        sched.tree_cache = MagicMock()
        sched.waiting_queue = []
        sched.disagg_decode_transfer_queue = MagicMock()
        sched.disagg_decode_transfer_queue.pop_transferred.return_value = []
        sched.disagg_decode_prealloc_queue = MagicMock()
        sched.disagg_decode_prealloc_queue.resume_retracted_reqs.return_value = []
        sched.disagg_decode_prealloc_queue.retracted_queue = []
        sched.disagg_decode_prealloc_queue.pop_preallocated.return_value = ([], [])
        return sched

    def test_decode_loop_drains_flexkv_stores(self):
        sched = self._bare_decode_scheduler(enable_flexkv=True)
        sched.process_decode_queue()
        sched.tree_cache.check_hicache_events.assert_called_once_with()

    def test_decode_loop_without_flexkv_or_hicache_skips_the_hook(self):
        sched = self._bare_decode_scheduler(enable_flexkv=False)
        sched.process_decode_queue()
        sched.tree_cache.check_hicache_events.assert_not_called()

    def test_flexkv_radix_cache_skips_host_lookup_on_decode(self):
        module = _load_flexkv_radix_cache_module()
        cache = module.FlexKVRadixCache.__new__(module.FlexKVRadixCache)
        cache.disable = False
        cache.page_size = 1
        cache._pd_decode = True
        cache.flexkv_connector = MagicMock()
        base_result = object()
        req = SimpleNamespace(rid="r1")
        params = MatchPrefixParams(key=[1, 2, 3, 4], req=req)
        with patch.object(
            module.RadixCache, "match_prefix", return_value=base_result
        ) as base_match:
            result = cache.match_prefix(params)
        assert result is base_result
        base_match.assert_called_once_with(params)
        cache.flexkv_connector.lookup_kv.assert_not_called()
