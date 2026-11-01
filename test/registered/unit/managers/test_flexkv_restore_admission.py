import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.prefill_budget import PrefillBudget
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class ControlledCache:
    disable = False

    def __init__(self, prefer=True, actual=None):
        self.prefer_complete_restored_requests = prefer
        self.actual = actual
        self.loads = []

    def supports_mamba(self):
        return False

    def evictable_size(self):
        return 0

    def is_tree_cache(self):
        return False

    def inc_lock_ref(self, node):
        return SimpleNamespace(to_dec_params=lambda: object())

    def dec_lock_ref(self, node, *args):
        pass

    def init_load_back(self, params):
        self.loads.append(params.req.rid)
        size = params.host_hit_length if self.actual is None else self.actual
        return torch.arange(size, dtype=torch.int64), object()

    def finish_storage_prefetch_admission(self, *args, **kwargs):
        pass


class ControlledAllocator:
    page_size = 64

    def __init__(self, capacity):
        self.capacity = capacity

    def available_size(self):
        return self.capacity

    def create_prefill_budget(self, cache, **kwargs):
        return PrefillBudget(self, cache, **kwargs)


class RestoredReq:
    def __init__(self, rid=0, tail=5120, host=97280, device=0):
        self.rid = str(rid)
        self.full_untruncated_fill_ids = range(device + host + tail)
        self.prefix_indices = torch.arange(device, dtype=torch.int64)
        self.host_hit_length = host
        self.host_loaded_length = 0
        self.swa_host_hit_length = 0
        self.storage_hit_length = 0
        self.storage_hit_start = None
        self.host_hit_is_storage = False
        self.last_node = self.best_match_node = object()
        self.output_ids = []
        self.retracted_stain = False
        self.sampling_params = SimpleNamespace(ignore_eos=False, max_new_tokens=1)
        self.kv = SimpleNamespace(cache_protected_len=0)
        self.cache_request_handle = object()
        self.extend_input_len = host + tail

    def needs_host_load_back(self):
        return self.host_hit_length > 0

    def materialized_host_hit_len(self):
        return self.host_loaded_length

    def fulfilled_storage_hit_len(self, prefix_len):
        return 0

    def set_extend_range(self, begin, end):
        self.extend_input_len = end - begin
        self.extend_range = SimpleNamespace(length=end - begin)


class TestRestoredTailAdmission(CustomTestCase):
    """A crowded batch must defer a short restore before I/O and a delayer vote."""

    def setUp(self):
        super().setUp()
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)

    def adder(self, cache, chunk=32768, capacity=10**9, exact=False):
        with patch(
            "sglang.srt.managers.schedule_policy._use_exact_chunk_fill",
            return_value=exact,
        ):
            return PrefillAdder(
                64, cache, ControlledAllocator(capacity), None, 1.0, 32768, chunk
            )

    def test_zero_chunk_budget_rejects_before_storage_restore(self):
        tree_cache = MagicMock()
        tree_cache.disable = False
        tree_cache.supports_mamba.return_value = False
        tree_cache.evictable_size.return_value = 0
        tree_cache.is_tree_cache.return_value = False

        allocator = MagicMock()
        allocator.available_size.return_value = 4096

        adder = PrefillAdder.__new__(PrefillAdder)
        adder.page_size = 4
        adder.exact_chunk_fill = False
        allocator.page_size = 4
        adder.memory_budget = PrefillBudget(allocator, tree_cache)
        adder.tree_cache = tree_cache
        adder.token_to_kv_pool_allocator = allocator
        adder.running_batch = None
        adder.prefill_delayer_single_pass = None
        adder.dsa_prefill_cp_in_seq_split = False
        adder.prefill_max_requests = None
        adder.can_run_list = []
        adder.dllm_config = None
        adder.rem_chunk_tokens = 0
        adder.rem_input_tokens = 4096
        adder.rem_total_token_offset = 0
        adder.cur_rem_token_offset = 0
        adder.is_hybrid_swa = False
        adder.is_all_swa = False
        adder.is_hybrid_ssm_cache = False
        adder._mamba_slot_cost = 0
        adder.rem_mamba_slots = None

        req = SimpleNamespace(
            sampling_params=SimpleNamespace(max_new_tokens=16, ignore_eos=False),
            output_ids=[],
            full_untruncated_fill_ids=list(range(12)),
            prefix_indices=torch.arange(4, dtype=torch.int64),
            host_hit_length=4,
            swa_host_hit_length=0,
            last_node=object(),
            retracted_stain=False,
            needs_host_load_back=lambda: True,
        )

        result = adder.add_one_req(
            req,
            has_chunked_req=False,
            truncation_align_size=None,
        )

        assert result is AddReqResult.OTHER
        tree_cache.init_load_back.assert_not_called()
        assert adder.can_run_list == []

    def test_35_short_restored_tails_finish_without_split_or_rejected_io(self):
        for exact in [False, True]:
            with self.subTest(exact=exact):
                cache = ControlledCache()
                waiting = [RestoredReq(i) for i in range(35)]
                batches = []
                while waiting:
                    adder = self.adder(cache, exact=exact)
                    before = len(cache.loads)
                    for req in waiting:
                        if (
                            adder.add_one_req(req, False, None)
                            is not AddReqResult.CONTINUE
                        ):
                            break
                    self.assertTrue(adder.can_run_list)
                    self.assertIsNone(adder.new_chunked_req)
                    self.assertEqual(len(cache.loads) - before, len(adder.can_run_list))
                    self.assertTrue(
                        all(req.extend_input_len == 5120 for req in adder.can_run_list)
                    )
                    batches.append(len(adder.can_run_list))
                    waiting = [req for req in waiting if req not in adder.can_run_list]
                self.assertEqual(batches, [6, 6, 6, 6, 6, 5])
                self.assertEqual(len(set(cache.loads)), 35)

    def test_deferred_tail_neither_votes_prefillable_nor_starts_restore(self):
        cache = ControlledCache()
        adder = self.adder(cache)
        for i in range(6):
            adder.add_one_req(RestoredReq(i), False, None)
        votes = []
        adder.running_batch = SimpleNamespace(batch_size=lambda: 0)
        adder.prefill_delayer_single_pass = SimpleNamespace(
            negotiate_should_allow_prefill=lambda **kwargs: votes.append(kwargs) or True
        )
        result = adder.add_one_req(RestoredReq(6), False, None)
        self.assertIs(result, AddReqResult.OTHER)
        self.assertEqual(len(cache.loads), 6)
        self.assertFalse(votes)

    def test_opt_out_and_gpu_hits_keep_current_chunk_packing(self):
        for prefer, host, device in [(False, 97280, 0), (True, 0, 97280)]:
            with self.subTest(prefer=prefer, host=host):
                cache = ControlledCache(prefer)
                adder = self.adder(cache)
                for i in range(35):
                    req = RestoredReq(i, host=host, device=device)
                    if adder.add_one_req(req, False, None) is not AddReqResult.CONTINUE:
                        break
                self.assertEqual(len(adder.can_run_list), 7)
                self.assertEqual(adder.new_chunked_req.extend_input_len, 2048)

    def test_empty_batches_and_long_tails_always_progress(self):
        for tail, chunk in [(40000, 32768), (5120, 2048)]:
            with self.subTest(tail=tail, chunk=chunk):
                cache = ControlledCache()
                adder = self.adder(cache, chunk=chunk)
                req = RestoredReq(tail=tail)
                adder.add_one_req(req, False, None)
                self.assertIs(adder.new_chunked_req, req)
                self.assertEqual(req.extend_input_len, chunk)
                self.assertEqual(len(cache.loads), 1)

    def test_capacity_rejection_starts_no_restore(self):
        cache = ControlledCache()
        result = self.adder(cache, capacity=100).add_one_req(RestoredReq(), False, None)
        self.assertIs(result, AddReqResult.NO_TOKEN)
        self.assertFalse(cache.loads)

    def test_zero_restore_retains_current_recompute_fallback(self):
        cache = ControlledCache(actual=0)
        adder = self.adder(cache)
        req = RestoredReq()
        adder.add_one_req(req, False, None)
        self.assertIs(adder.new_chunked_req, req)
        self.assertEqual(req.extend_input_len, 32768)
        self.assertEqual(len(cache.loads), 1)


if __name__ == "__main__":
    unittest.main()
