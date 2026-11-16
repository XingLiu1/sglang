"""Byte admission, block alignment and queued/restore credit regressions."""

import sys
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock, call, patch

import pytest

from sglang.srt.managers.flexkv_prefetch_window import (
    FlexKVPrefetchWindow,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def requests(n, tokens=8):
    return [
        NS(rid=str(i), cache_request_handle=(str(i), 0), tokens=tokens)
        for i in range(n)
    ]


def cache():
    return NS(
        has_uncommitted_restore=lambda req: False,
        prefetch_token_count=lambda req: req.tokens,
        prefetch_request=Mock(return_value=True),
        terminate_prefetch=Mock(),
    )


def window(capacity=40, swa=0):
    return FlexKVPrefetchWindow(4, 10, capacity_bytes=capacity, swa_slot_bytes=swa)


def test_long_backlog_retains_unconsumed_credit_and_refills_after_admission():
    reqs, tree, budget = requests(1000), cache(), window()
    for _ in range(10):
        budget.update(reqs, tree)
    assert tree.prefetch_request.call_args_list == [
        call(r, max_tokens=8) for r in reqs[:2]
    ]
    assert budget.reserved_bytes == 40
    # Finishing I/O does not return the completed, unconsumed result's credit.
    budget.update(reqs[1:], tree)
    assert tree.prefetch_request.call_args_list == [
        call(r, max_tokens=8) for r in reqs[:3]
    ]
    assert budget.reserved_bytes == 40
    assert not budget.contains(reqs[0])
    tree.terminate_prefetch.assert_not_called()


def test_mixed_lengths_and_sidecars_fit_bytes_instead_of_request_count():
    reqs, tree, budget = requests(5), cache(), window(56, swa=3)
    reqs[0].tokens, reqs[1].tokens = 4, 12
    budget.update(reqs, tree)
    # 13 + 33 bytes leaves ten bytes, too little for a block plus its sidecar.
    assert tree.prefetch_request.call_args_list == [
        call(reqs[0], max_tokens=4),
        call(reqs[1], max_tokens=12),
    ]
    assert budget.reserved_bytes == 46
    short = requests(150, tokens=4)
    budget = window(1500)
    budget.update(short, tree)
    assert all(budget.contains(req) for req in short)
    assert budget.reserved_bytes == 1500  # no hidden fixed K, including 128


def test_oversized_head_reserves_only_an_aligned_prefix_without_starving():
    reqs, tree, budget = requests(2, tokens=100), cache(), window(36, swa=5)
    budget.update(reqs, tree)
    assert tree.prefetch_request.call_args_list == [call(reqs[0], max_tokens=12)]
    assert budget.reserved_bytes == 35
    assert not budget.contains(reqs[1])
    # A sub-block pool starts no I/O and cannot block normal compute admission.
    small = window(14, swa=5)
    small.update(reqs, tree)
    assert small.reserved_bytes == 0


def test_reprioritization_cancels_old_lease_before_starting_new_head():
    reqs, tree, budget = requests(4), cache(), window()
    budget.update(reqs, tree)
    events = Mock()
    events.attach_mock(tree.prefetch_request, "start")
    events.attach_mock(tree.terminate_prefetch, "cancel")
    budget.update([reqs[3], reqs[0], reqs[1], reqs[2]], tree)
    assert events.mock_calls == [
        call.cancel(("1", 0)),
        call.start(reqs[3], max_tokens=8),
    ]
    assert budget.contains(reqs[0]) and budget.contains(reqs[3])
    assert budget.reserved_bytes == 40


def test_h2d_restore_keeps_credit_after_leaving_waiting_queue():
    reqs, tree, budget = requests(4), cache(), window()
    budget.update(reqs, tree)
    tree.has_uncommitted_restore = lambda req: req is reqs[0]
    budget.update(reqs[1:], tree)
    assert budget.contains(reqs[0]) and budget.contains(reqs[1])
    assert not budget.contains(reqs[2])
    assert budget.reserved_bytes == 40
    tree.terminate_prefetch.assert_not_called()
    tree.has_uncommitted_restore = lambda req: False
    budget.update(reqs[1:], tree)
    assert not budget.contains(reqs[0]) and budget.contains(reqs[2])
    assert budget.reserved_bytes == 40


def test_ineligible_lora_does_not_spend_prefetch_credit():
    reqs, tree, budget = requests(5), cache(), window()
    budget.update(reqs, tree, eligible=lambda req: int(req.rid) >= 2)
    assert budget.contains(reqs[2]) and budget.contains(reqs[3])
    assert not budget.contains(reqs[0])
    assert budget.reserved_bytes == 40


def test_abort_and_rid_reuse_do_not_inherit_old_reservation():
    old, other = requests(2)
    tree, budget = cache(), window(20)
    budget.update([old, other], tree)
    replacement = NS(rid=old.rid, cache_request_handle=(old.rid, 1), tokens=8)
    assert not budget.contains(replacement)
    budget.update([replacement, other], tree)
    assert tree.prefetch_request.call_args_list == [
        call(old, max_tokens=8),
        call(replacement, max_tokens=8),
    ]
    assert budget.contains(replacement) and not budget.contains(old)
    budget.update([], tree)
    assert budget.reserved_bytes == 0


def test_rejected_backend_admission_is_retried_without_phantom_credit():
    reqs, tree, budget = requests(1), cache(), window()
    tree.prefetch_request.side_effect = [False, True]
    budget.update(reqs, tree)
    assert budget.reserved_bytes == 0
    budget.update(reqs, tree)
    assert budget.reserved_bytes == 20 and budget.contains(reqs[0])


@pytest.mark.parametrize("blocks,nbytes,swa", [(0, 10, 0), (4, 0, 0), (4, 10, -1)])
def test_invalid_geometry(blocks, nbytes, swa):
    with pytest.raises(ValueError, match="geometry"):
        FlexKVPrefetchWindow(blocks, nbytes, capacity_bytes=100, swa_slot_bytes=swa)


@pytest.mark.parametrize("swa_aware", [False, True])
def test_admission_reuses_the_resolved_backend_budget(swa_aware):
    module = ModuleType("flexkv.prefetch.budget")
    resolved = NS(
        tokens_per_block=4,
        block_bytes=7,
        swa_slot_bytes=11 if swa_aware else 0,
        byte_limit=(3 if swa_aware else 128) * 1024**3,
    )
    module.resolve_prefetch_budget = Mock(return_value=resolved)
    connector = NS(
        model_config=object(),
        cache_config=NS(prefetch_max_pinned_bytes=128 * 1024**3),
        _swa_kv_pool=object() if swa_aware else None,
    )
    with patch.dict(sys.modules, {"flexkv.prefetch.budget": module}):
        budget = FlexKVPrefetchWindow.from_connector(connector)
    # The backend has already combined the configured limit and half-pool cap.
    assert budget.capacity_bytes == (3 if swa_aware else 128) * 1024**3
    module.resolve_prefetch_budget.assert_called_once_with(
        connector.model_config, connector.cache_config, swa_aware=swa_aware
    )
    budget.update(requests(1, tokens=4), cache())
    assert budget.reserved_bytes == 7 + (11 if swa_aware else 0)


def test_retraction_of_mutated_request_cancels_only_its_old_attempt():
    req = requests(1)[0]
    tree, budget = cache(), window()
    budget.update([req], tree)
    old = req.cache_request_handle
    req.cache_request_handle = (req.rid, 1)
    assert not budget.contains(req)
    budget.update([req], tree)
    assert tree.terminate_prefetch.call_args_list == [call(old)]
    assert tree.prefetch_request.call_count == 2
    assert budget.reserved_bytes == 20


def test_reused_request_identity_with_equal_fields_starts_new_prefetch():
    old = requests(1)[0]
    replacement = NS(**vars(old))
    tree, budget = cache(), window(20)
    budget.update([old], tree)
    budget.update([replacement], tree)
    assert budget.contains(replacement) and not budget.contains(old)
    assert tree.prefetch_request.call_count == 2
    assert budget.reserved_bytes == 20


def test_sub_block_request_tail_does_not_end_the_whole_lookahead():
    reqs, tree, budget = requests(3, tokens=11), cache(), window(40)
    budget.update(reqs, tree)
    # Two complete blocks per request. Its three unprefetchable tokens do not
    # consume bytes or prevent the next request from using the other half.
    assert tree.prefetch_request.call_args_list == [
        call(reqs[0], max_tokens=8),
        call(reqs[1], max_tokens=8),
    ]
    assert budget.reserved_bytes == 40


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
