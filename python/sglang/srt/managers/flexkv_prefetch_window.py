"""Bound scheduler-ordered speculative KV by bytes in the existing Host pool."""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle


@dataclass(frozen=True, eq=False)
class _Reservation:
    req: Req
    tokens: int
    nbytes: int

    def matches(self, other: _Reservation | None) -> bool:
        return (
            other is not None
            and self.req is other.req
            and self.tokens == other.tokens
            and self.nbytes == other.nbytes
        )


class FlexKVPrefetchWindow:
    def __init__(
        self,
        tokens_per_block: int,
        block_bytes: int,
        *,
        capacity_bytes: int,
        swa_slot_bytes: int = 0,
    ):
        if tokens_per_block <= 0 or block_bytes <= 0 or swa_slot_bytes < 0:
            raise ValueError("Invalid FlexKV prefetch block geometry")
        self.tokens_per_block = tokens_per_block
        self.block_bytes = block_bytes
        self.swa_slot_bytes = swa_slot_bytes
        self.capacity_bytes = max(0, capacity_bytes)
        self._requests: dict[CacheRequestHandle, _Reservation] = {}

    @classmethod
    def from_connector(cls, connector):
        # The backend owns both policy and block/SWA geometry. This also avoids
        # retaining a stale 2 GiB frontend limit from an older config default.
        from flexkv.prefetch.budget import resolve_prefetch_budget

        budget = resolve_prefetch_budget(
            connector.model_config,
            connector.cache_config,
            swa_aware=connector._swa_kv_pool is not None,
        )
        return cls(
            budget.tokens_per_block,
            budget.block_bytes,
            capacity_bytes=budget.byte_limit,
            swa_slot_bytes=budget.swa_slot_bytes,
        )

    @property
    def reserved_bytes(self) -> int:
        return sum(entry.nbytes for entry in self._requests.values())

    def contains(self, req: Req) -> bool:
        entry = self._requests.get(req.cache_request_handle)
        return entry is not None and entry.req is req

    def update(
        self,
        waiting_queue: Collection[Req],
        tree_cache,
        eligible: Callable[[Req], bool] | None = None,
    ) -> None:
        """Reserve aligned prefixes in priority order, without capping batching.

        The complete hash-chain prefix is conservatively charged, even for
        shared/local hits. A completed, unconsumed result keeps its reservation.
        After admission, a live restore keeps the credit until its checkpoint;
        backend pins separately protect all source buffers until DMA drains.
        """
        previous = self._requests
        selected = {
            key: entry
            for key, entry in previous.items()
            if tree_cache.has_uncommitted_restore(entry.req)
        }
        available = self.capacity_bytes - sum(e.nbytes for e in selected.values())
        for req in waiting_queue:
            if tree_cache.has_uncommitted_restore(req):
                continue
            if eligible is not None and not eligible(req):
                continue
            tokens = (
                tree_cache.prefetch_token_count(req)
                // self.tokens_per_block
                * self.tokens_per_block
            )
            blocks = min(
                tokens // self.tokens_per_block,
                max(0, available - self.swa_slot_bytes) // self.block_bytes,
            )
            if tokens < self.tokens_per_block:
                continue
            if blocks == 0:
                break
            span = blocks * self.tokens_per_block
            nbytes = blocks * self.block_bytes + self.swa_slot_bytes
            selected[req.cache_request_handle] = _Reservation(req, span, nbytes)
            available -= nbytes
            if span < tokens:
                break

        waiting = {id(req) for req in waiting_queue}
        for key, entry in previous.items():
            replacement = selected.get(key)
            if entry.matches(replacement):
                continue
            if id(entry.req) in waiting and not tree_cache.has_uncommitted_restore(
                entry.req
            ):
                tree_cache.terminate_prefetch(key)

        self._requests = {}
        for key, entry in selected.items():
            if not entry.matches(previous.get(key)):
                # A backend session-capacity rejection is retryable next pass.
                # Do not manufacture a retained reservation for failed I/O.
                if (
                    tree_cache.prefetch_request(entry.req, max_tokens=entry.tokens)
                    is False
                ):
                    continue
            self._requests[key] = entry
