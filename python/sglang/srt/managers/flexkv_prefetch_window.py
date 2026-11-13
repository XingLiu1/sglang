"""Bound speculative Host KV residency by the scheduler's current queue order."""

from __future__ import annotations

from collections.abc import Callable, Collection
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
    from sglang.srt.mem_cache.storage.flexkv.flexkv_cache_lifecycle import (
        FlexKVCacheLifecycleMixin,
    )


class FlexKVPrefetchWindow:
    def __init__(self, size: int):
        if size <= 0:
            raise ValueError("FlexKV prefetch window size must be positive")
        self.size = size
        self._requests: dict[CacheRequestHandle, Req] = {}

    def contains(self, req: Req) -> bool:
        # A reused business rid must not inherit an older request's admission.
        return self._requests.get(req.cache_request_handle) is req

    def update(
        self,
        waiting_queue: Collection[Req],
        tree_cache: FlexKVCacheLifecycleMixin,
        eligible: Callable[[Req], bool] | None = None,
    ) -> None:
        """Start only the first eligible requests; retain completed result leases.

        Leaving the queue means normal admission/abort owns cleanup. A request
        that remains queued but loses its position relinquishes its speculative
        lease. Never cancel a restore that already owns device destinations.
        FlexKV retains canceled in-flight buffers until their transfers drain.
        """
        selected = {}
        for req in waiting_queue:
            if tree_cache.has_uncommitted_restore(req):
                continue
            if eligible is not None and not eligible(req):
                continue
            selected[req.cache_request_handle] = req
            if len(selected) >= self.size:
                break

        waiting = {id(req) for req in waiting_queue}
        for rid, req in self._requests.items():
            if selected.get(rid) is req:
                continue
            if id(req) in waiting and not tree_cache.has_uncommitted_restore(req):
                tree_cache.terminate_prefetch(rid)

        previous = self._requests
        self._requests = selected
        for rid, req in selected.items():
            if previous.get(rid) is not req:
                tree_cache.prefetch_request(req)
