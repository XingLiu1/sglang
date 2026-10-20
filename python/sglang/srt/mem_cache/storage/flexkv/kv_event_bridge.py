"""Forward FlexKV's host-tier placement events onto sglang's KV event wire.

A KV-aware router (llm-d, dynamo) sees only what this engine publishes, so
blocks offloaded into FlexKV look gone and their prefixes score as misses.
The bridge relays FlexKV's own stores and evictions on the existing channel;
the router still sees one publisher per engine, just with a larger cache.

* A store carries ``token_ids`` and a removal does not: the router maps
  FlexKV's hashes onto its own keys once, at store time.
* The medium is ``CPU`` (``CPU_PINNED`` on the wire), so a router that weights
  tiers can tell these blocks cost a host-to-device copy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from sglang.srt.disaggregation.kv_events import StorageMedium

if TYPE_CHECKING:
    from sglang.srt.mem_cache.events import KVCacheEventRecorder

logger = logging.getLogger(__name__)


class FlexKVEventBridge:
    """Pumps :mod:`flexkv.common.kv_events` events into a recorder.

    Inert unless KV events are on and FlexKV can report evictions too:
    stores without removes would make the router's view only ever grow.
    """

    def __init__(
        self,
        connector: Any,
        recorder: Optional[KVCacheEventRecorder],
        page_size: int,
    ) -> None:
        self._connector = connector
        self._recorder = recorder
        self._page_size = int(page_size)
        self._active = False
        self._warned_block_size = False

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        """Arm the bridge if both sides support it. Safe to call twice."""
        if self._active:
            return
        if self._recorder is None or not self._recorder.enabled:
            return
        # Version gate: FlexKV releases before the event API lack this method.
        start = getattr(self._connector, "start_kv_events", None)
        if start is None:
            return
        try:
            self._active = bool(start())
        except Exception as exc:  # noqa: BLE001
            logger.warning("[FlexKV] KV event bridge failed to start: %s", exc)
            return
        if self._active:
            logger.info(
                "[FlexKV] KV event bridge active: host-tier blocks are "
                "published as medium=%s",
                StorageMedium.CPU.value,
            )

    def drain(self) -> None:
        """Move one batch of FlexKV events into the recorder. Non-blocking."""
        if not self._active:
            return
        try:
            events = self._connector.take_kv_events()
        except Exception as exc:  # noqa: BLE001 - never fail a scheduler tick
            logger.warning("[FlexKV] KV event drain failed: %s", exc)
            return
        for event in events:
            try:
                self._forward(event)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[FlexKV] dropping malformed KV event %s: %s",
                    type(event).__name__,
                    exc,
                )

    def _forward(self, event: Any) -> None:
        name = type(event).__name__
        if name == "BlocksStored":
            self._check_block_size(event.block_size)
            self._recorder.record_external_store(
                block_hashes=event.block_hashes,
                token_ids=event.token_ids,
                parent_block_hash=event.parent_block_hash,
                medium=_medium(event.medium),
            )
        elif name == "BlocksRemoved":
            self._recorder.record_external_remove(
                block_hashes=event.block_hashes,
                medium=_medium(event.medium),
            )
        elif name == "AllBlocksCleared":
            # Also retracts the device tier; the wire protocol has no per-tier clear.
            logger.warning("[FlexKV] host tier requested a KV event resync")
            self._recorder.record_all_cleared()
        else:
            raise ValueError(f"unknown event type {name}")

    def _check_block_size(self, block_size: int) -> None:
        # The connector sets tokens_per_block = page_size; a mismatch is a config bug.
        if int(block_size) == self._page_size or self._warned_block_size:
            return
        self._warned_block_size = True
        logger.error(
            "[FlexKV] KV event block_size=%d does not match sglang "
            "page_size=%d; a router will mis-score every host-tier hit",
            block_size,
            self._page_size,
        )


def _medium(value: Optional[str]) -> StorageMedium:
    # FlexKV says "CPU"; StorageMedium.CPU serializes as "CPU_PINNED" on the wire.
    if value is None:
        return StorageMedium.CPU
    name = str(value).strip().upper()
    if name in ("CPU", "CPU_PINNED", "HOST"):
        return StorageMedium.CPU
    if name == "DISK" or name == "SSD":
        return StorageMedium.DISK
    if name in ("EXTERNAL", "REMOTE"):
        return StorageMedium.EXTERNAL
    if name == "GPU":
        return StorageMedium.GPU
    return StorageMedium.CPU
