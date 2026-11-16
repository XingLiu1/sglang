"""Bound scheduler-ordered speculative KV by bytes in the existing Host pool."""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle


PREFETCH_BYTE_BUDGET = 128 * 1024**3


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
        self.capacity_bytes = max(0, min(PREFETCH_BYTE_BUDGET, capacity_bytes))
        self._requests: dict[CacheRequestHandle, _Reservation] = {}

    @classmethod
    def from_connector(cls, connector):
        # Use the backend's pool geometry, including block-level compression.
        # Dividing a compressed block into rounded integer bytes/token loses KV.
        from flexkv.common.config import (
            GLOBAL_CONFIG_FROM_ENV,
            RankInfo,
            block_size_in_bytes_for_cache,
        )

        model, cache = connector.model_config, connector.cache_config
        replicas = 1
        if (
            model.num_kv_heads == 1
            and GLOBAL_CONFIG_FROM_ENV.kv_shared_across_ranks_mode == "all_write"
        ):
            replicas = model.effective_tp_size_per_node
        block_bytes = replicas * block_size_in_bytes_for_cache(
            model, cache, RankInfo(model)
        )
        swa_bytes = 0
        swa = cache.swa
        if swa is not None and swa.enabled and connector._swa_kv_pool is not None:
            if swa.multi_group and swa.snapshot_bytes is None:
                raise ValueError("SWA prefetch requires registered snapshot geometry")
            swa_bytes = replicas * (
                swa.snapshot_bytes
                or swa.num_swa_layers
                * cache.tokens_per_block
                * swa.bytes_per_token_per_layer
            )
        # A smaller backend limit still wins. Leave a Full-KV block available
        # for foreground work; the backend remains the authoritative ledger
        # across clients, canceled DMA, tree-node pins and H2D ownership.
        capacity = min(
            cache.prefetch_max_pinned_bytes,
            max(0, cache.num_cpu_blocks - 1) * block_bytes,
        )
        return cls(
            cache.tokens_per_block,
            block_bytes,
            capacity_bytes=capacity,
            swa_slot_bytes=swa_bytes,
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
