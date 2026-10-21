"""Advertise blocks stored into the FlexKV host tier on the KV event wire.

A KV-aware router (dynamo, llm-d) only sees this engine's KV events, so a
prefix offloaded to FlexKV looks like a miss once the device drops it. After a
FlexKV store completes, the stored node is published as ``medium=CPU``, the
same way HiCache reports its host tier, so the hashes match the device events.

FlexKV reports no evictions here. The budget keeps the router's CPU view
bounded by retracting the least recently advertised blocks once their count
exceeds the host tier's capacity. That approximates FlexKV's own eviction, so
the CPU view is a routing hint, not an exact residency map.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, Optional

from sglang.srt.disaggregation.kv_events import StorageMedium

if TYPE_CHECKING:
    from sglang.srt.mem_cache.events import KVCacheEventRecorder


class HostTierEventBudget:
    """Publishes host-tier stores and bounds how many stay advertised."""

    def __init__(
        self,
        recorder: Optional[KVCacheEventRecorder],
        capacity_blocks: Callable[[], int],
    ) -> None:
        self._recorder = recorder
        # Read per call: FlexKV resizes its CPU pool after the cache is built.
        self._capacity_blocks = capacity_blocks
        self._advertised: OrderedDict[int, None] = OrderedDict()

    @property
    def advertised_blocks(self) -> int:
        return len(self._advertised)

    def record_store(self, node: Any) -> None:
        if self._recorder is None or not self._recorder.enabled:
            return
        for block_hash in self._recorder.record_store(node, medium=StorageMedium.CPU):
            self._advertised[block_hash] = None
            self._advertised.move_to_end(block_hash)
        excess = len(self._advertised) - max(int(self._capacity_blocks()), 0)
        if excess > 0:
            evicted = [self._advertised.popitem(last=False)[0] for _ in range(excess)]
            self._recorder.record_remove_hashes(evicted, medium=StorageMedium.CPU)

    def clear(self) -> None:
        # The owning cache's reset publishes AllBlocksCleared itself.
        self._advertised.clear()
