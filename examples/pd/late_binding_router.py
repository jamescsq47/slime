"""Late-binding PD router for the agentic serving experiments.

The stock SGLang PD router selects a prefill/decode pair before either request
is dispatched.  In P-ready mode the prefill worker can compute without a
decode destination, so this router deliberately delays decode dispatch until
the prefill worker publishes ``<bootstrap_room>.ready``.

This module only changes routing/admission timing.  KV transfer, HiCache and
Mooncake continue to use the SGLang implementations selected by the workers.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import aiohttp
import orjson
from fastapi.responses import ORJSONResponse, StreamingResponse
from sglang.srt.disaggregation.agentic_early_claim import AgenticEarlyClaimStore
from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    AgenticRequestMetadata,
    unpack_agentic_extra_key,
)
from sglang_router.mini_lb import (
    AIOHTTP_STREAM_READ_CHUNK_SIZE,
    MiniLoadBalancer,
)


logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DecodeLoad:
    url: str
    used_tokens: int
    capacity_tokens: int
    running: int
    waiting: int
    prealloc: int
    transfer: int
    max_running: int
    running_kv_tokens: int = 0
    prealloc_tokens: int = 0
    transfer_tokens: int = 0
    physical_used_tokens: int = 0


@dataclass(frozen=True)
class DecodeReservation:
    reservation_id: str
    url: str
    prompt_tokens: int
    admission_tokens: int
    request_count: int
    rooms: tuple[int, ...]
    created_at: float
    draining: bool = False


@dataclass
class _PrefillWorkReservation:
    domain: int
    tokens: int
    requests: int = 1
    released: bool = False
    route_pending: bool = False


@dataclass(eq=False)
class _PrefillAdmissionWaiter:
    parent_turn: bool
    enqueued_at: float
    sequence: int


@dataclass(frozen=True)
class _GenerationResponse:
    payload: dict[str, Any]
    status: int
    completed_at: float


class _PrefillRedirect(RuntimeError):
    def __init__(self, domain: int, route: str):
        super().__init__(f"redirect Prefill to P{domain} via {route}")
        self.domain = domain
        self.route = route


class _PrefillAdmissionGate:
    """Bound hidden HTTP/bootstrap work before requests reach P.

    Parent turns always precede initial requests.  This is intentionally a
    strict agentic-serving priority boundary: a New request must never consume
    the short D-HBM Direct deadline of a returned parent turn.  FIFO is kept
    within each class.
    """

    def __init__(self, limit: int, new_aging_seconds: float):
        if limit <= 0:
            raise ValueError("P admission limit must be positive")
        self.limit = limit
        self.new_aging_seconds = max(0.0, new_aging_seconds)
        self.active = 0
        self._sequence = 0
        self._waiters: list[_PrefillAdmissionWaiter] = []
        self._condition = asyncio.Condition()

    def _next_waiter(self, now: float) -> _PrefillAdmissionWaiter:
        def key(waiter: _PrefillAdmissionWaiter):
            # Never promote an aged New request above a parent.  GPU-side
            # admission has its own P-ready soft caps, while this gate only
            # orders the small HTTP/tokenizer work window in front of P.
            queue_class = 0 if waiter.parent_turn else 1
            return queue_class, waiter.enqueued_at, waiter.sequence

        return min(self._waiters, key=key)

    async def acquire(self, *, parent_turn: bool) -> float:
        waiter = _PrefillAdmissionWaiter(
            parent_turn=parent_turn,
            enqueued_at=time.monotonic(),
            sequence=self._sequence,
        )
        self._sequence += 1
        async with self._condition:
            self._waiters.append(waiter)
            try:
                while True:
                    if (
                        self.active < self.limit
                        and self._next_waiter(time.monotonic()) is waiter
                    ):
                        self._waiters.remove(waiter)
                        self.active += 1
                        self._condition.notify_all()
                        return time.monotonic() - waiter.enqueued_at
                    # Aging changes priority even when no admission completes.
                    try:
                        await asyncio.wait_for(
                            self._condition.wait(),
                            timeout=max(0.01, self.new_aging_seconds),
                        )
                    except asyncio.TimeoutError:
                        continue
            except BaseException:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                    self._condition.notify_all()
                raise

    async def release(self) -> None:
        async with self._condition:
            if self.active <= 0:
                raise RuntimeError("P admission release without active request")
            self.active -= 1
            self._condition.notify_all()


class LateBindingMiniLoadBalancer(MiniLoadBalancer):
    """Dispatch P first and bind D only after the exact P workload is ready."""

    def __init__(self, router_args):
        super().__init__(router_args)
        ready_dir = os.environ.get("SGLANG_PD_P_READY_DIR", "")
        if not ready_dir:
            raise ValueError(
                "Late-binding requires SGLANG_PD_P_READY_DIR on router, P and D"
            )
        self.p_ready_dir = Path(ready_dir)
        self.p_ready_dir.mkdir(parents=True, exist_ok=True)
        self.ready_timeout = _env_float("SGLANG_PD_LATE_BIND_READY_TIMEOUT_S", 600.0)
        self.ready_poll_interval = _env_float(
            "SGLANG_PD_LATE_BIND_POLL_INTERVAL_S", 0.02
        )
        self.max_prefill_inflight = _env_int(
            "SGLANG_PD_LATE_BIND_MAX_PREFILL_INFLIGHT", 64
        )
        self.prefill_accept_timeout = _env_float(
            "SGLANG_PD_LATE_BIND_ACCEPT_TIMEOUT_S", self.ready_timeout
        )
        self.prefill_queue_timeout = _env_float(
            "SGLANG_PD_LATE_BIND_QUEUE_TIMEOUT_S", 3600.0
        )
        self.prefill_new_aging_seconds = _env_float(
            "SGLANG_PD_LATE_BIND_NEW_AGING_S", 10.0
        )
        self._prefill_admission = _PrefillAdmissionGate(
            self.max_prefill_inflight, self.prefill_new_aging_seconds
        )
        self.load_timeout = _env_float("SGLANG_PD_LATE_BIND_LOAD_TIMEOUT_S", 2.0)
        self.reservation_timeout = _env_float(
            "SGLANG_PD_LATE_BIND_RESERVATION_TIMEOUT_S", 120.0
        )
        self.decode_headroom_tokens = _env_int(
            "SGLANG_PD_LATE_BIND_DECODE_HEADROOM_TOKENS", 512
        )
        self.max_decode_admission_tokens = _env_int(
            "SGLANG_PD_LATE_BIND_MAX_ADMISSION_TOKENS", 4096
        )
        self.request_load_weight = _env_float(
            "SGLANG_PD_LATE_BIND_REQUEST_LOAD_WEIGHT", 0.05
        )
        self.transfer_request_weight = _env_float(
            "SGLANG_PD_LATE_BIND_TRANSFER_REQUEST_WEIGHT", 2.0
        )
        self.context_token_floor = _env_int(
            "SGLANG_PD_LATE_BIND_CONTEXT_TOKEN_FLOOR", 2048
        )
        self.context_token_ceiling = _env_int(
            "SGLANG_PD_LATE_BIND_CONTEXT_TOKEN_CEILING", 8192
        )
        self.wait_for_feasible_decode = _env_bool(
            "SGLANG_PD_LATE_BIND_WAIT_FOR_FEASIBLE", True
        )
        self.target_decode_kv_fraction = _env_float(
            "SGLANG_PD_LATE_BIND_TARGET_KV_FRACTION", 0.90
        )
        if not (0.0 < self.target_decode_kv_fraction <= 1.0):
            raise ValueError(
                "SGLANG_PD_LATE_BIND_TARGET_KV_FRACTION must be in (0, 1]"
            )
        self.no_capacity_poll_interval = _env_float(
            "SGLANG_PD_LATE_BIND_NO_CAPACITY_POLL_S", 0.01
        )
        self.soft_reservation_delay = _env_float(
            "SGLANG_PD_LATE_BIND_SOFT_RESERVATION_DELAY_S", 30.0
        )
        self.soft_reservation_min_tokens = _env_int(
            "SGLANG_PD_LATE_BIND_SOFT_RESERVATION_MIN_TOKENS", 20_000
        )
        self.soft_reservation_force_after = _env_float(
            "SGLANG_PD_LATE_BIND_SOFT_RESERVATION_FORCE_AFTER_S", 120.0
        )
        self.load_cache_ttl = _env_float(
            "SGLANG_PD_LATE_BIND_LOAD_CACHE_TTL_S", 0.20
        )
        self._selection_lock = asyncio.Lock()
        # Serialize P-ready submission order without serializing D admission.
        # Submitted sequences remain visible in the filesystem until D has
        # allocated KV, so track them separately from not-yet-submitted work.
        self._p_ready_fifo_lock = asyncio.Lock()
        self._p_ready_fifo_locks: dict[int, asyncio.Lock] = {}
        self._p_ready_submitted_sequences: set[Any] = set()
        # Multi-P runs can have hundreds of HTTP handlers waiting for P-ready.
        # Polling and JSON-decoding the same /dev/shm directory independently
        # in every handler creates an avoidable O(waiters * ready_files) control
        # path.  One watcher owns the directory scan and wakes every matching
        # waiter in a batch.  The single-P path intentionally keeps its proven
        # per-request behavior.
        self._p_ready_monitor_task: Optional[asyncio.Task] = None
        self._p_ready_waiters: dict[int, set[asyncio.Future]] = {}
        self._p_ready_snapshot: dict[int, dict[str, Any]] = {}
        self._reservations: dict[str, DecodeReservation] = {}
        self._last_loads: dict[str, DecodeLoad] = {}
        self._load_cache: list[DecodeLoad] = []
        self._load_cache_at = 0.0
        self._load_refresh_task: Optional[asyncio.Task] = None
        # A reservation remains charged after D consumes its P-ready marker
        # until a newer load snapshot includes that allocation.  This lets the
        # hot dispatch path use stale-while-revalidate snapshots safely.
        self._admitted_reservation_at: dict[str, float] = {}
        self._legacy_load_urls: set[str] = (
            set(self.decode_urls)
            if _env_bool("SGLANG_PD_LATE_BIND_FORCE_LEGACY_LOADS")
            else set()
        )
        self._prefill_index = 0
        # Router-owned shadow queues are the only load signal used to choose a
        # P worker.  Selection never performs an HTTP request or waits for a P
        # scheduler: the counters are charged at routing time and released as
        # soon as P publishes its ready marker.
        self._prefill_work_lock = asyncio.Lock()
        self._prefill_pending_tokens = [0] * len(self.prefill_urls)
        self._prefill_pending_requests = [0] * len(self.prefill_urls)
        self._prefill_work_tiebreak = 0
        # One logical request-generation may outlive an HTTP client's timeout.
        # Keep the actual P->D dispatch detached from that client and let every
        # retry await the same task.  Without this fence, a retry can select a
        # second D and duplicate both Decode compute and KV lifecycle writes.
        self._generation_lock = asyncio.Lock()
        self._generation_tasks: dict[
            str, asyncio.Task[tuple[dict[str, Any], int]]
        ] = {}
        self._generation_results: dict[str, _GenerationResponse] = {}
        self.generation_result_ttl = _env_float(
            "SGLANG_PD_GENERATION_RESULT_TTL_S", 3600.0
        )
        self.max_generation_results = _env_int(
            "SGLANG_PD_MAX_GENERATION_RESULTS", 8192
        )
        self.numa_domains = _env_bool("SGLANG_PD_LATE_BIND_NUMA_DOMAINS")
        self.dynamic_prefill_domains = _env_bool(
            "SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS"
        )
        self.global_decode = _env_bool("SGLANG_PD_LATE_BIND_GLOBAL_DECODE")
        if self.numa_domains and (
            len(self.prefill_urls) < 2
            or len(self.decode_urls) % len(self.prefill_urls) != 0
        ):
            raise ValueError(
                "NUMA-domain routing requires at least two P workers and an "
                "equal contiguous D partition per P"
            )
        if self.dynamic_prefill_domains and not self.numa_domains:
            raise ValueError(
                "dynamic Prefill domains require SGLANG_PD_LATE_BIND_NUMA_DOMAINS"
            )
        self.early_claim_store = None
        if _env_bool("SGLANG_AGENTIC_KV_EARLY_CLAIM"):
            early_claim_dir = os.environ.get("SGLANG_AGENTIC_KV_EARLY_CLAIM_DIR", "")
            if not early_claim_dir:
                early_claim_dir = str(self.p_ready_dir / "early-claims")
            self.early_claim_store = AgenticEarlyClaimStore(early_claim_dir)
        logger.info(
            "Late-binding PD enabled: ready_dir=%s decodes=%d headroom=%d "
            "early_claim=%s max_prefill_inflight=%d",
            self.p_ready_dir,
            len(self.decode_urls),
            self.decode_headroom_tokens,
            self.early_claim_store is not None,
            self.max_prefill_inflight,
        )

    def select_pair(self):
        """Select only P here; the endpoint passes the D placeholder to us."""
        if not self.prefill_urls:
            raise RuntimeError("No prefill servers available")
        index = self._prefill_index % len(self.prefill_urls)
        self._prefill_index += 1
        return (
            self.prefill_urls[index],
            self.prefill_bootstrap_ports[index],
            None,
        )

    @staticmethod
    def _rooms(modified_request: dict[str, Any]) -> tuple[int, ...]:
        rooms = modified_request["bootstrap_room"]
        if not isinstance(rooms, list):
            rooms = [rooms]
        return tuple(int(room) for room in rooms)

    def _ready_path(self, room: int) -> Path:
        return self.p_ready_dir / f"{room}.ready"

    def _accepted_path(self, room: int) -> Path:
        return self.p_ready_dir / f"{room}.accepted"

    def _request_domain(self, metadata, rooms: tuple[int, ...]) -> int:
        if not getattr(self, "numa_domains", False):
            return 0
        key = metadata.request_id if metadata is not None else str(rooms[0])
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % len(self.prefill_urls)

    @staticmethod
    def _request_input_tokens(request: dict[str, Any]) -> int:
        input_ids = request.get("input_ids")
        if not isinstance(input_ids, list):
            return 1
        if not input_ids:
            return 1
        if isinstance(input_ids[0], list):
            return max(1, sum(len(item) for item in input_ids))
        return max(1, len(input_ids))

    def _estimated_prefill_tokens(
        self,
        request: dict[str, Any],
        snapshot_tokens: Optional[int] = None,
        ) -> int:
        current_tokens = self._request_input_tokens(request)
        if snapshot_tokens is None:
            return current_tokens
        # The reusable D snapshot contains the previous prompt and generated
        # response.  The remaining suffix is the tool result plus chat framing.
        return max(1, current_tokens - max(0, int(snapshot_tokens)))

    async def _reserve_prefill_work(
        self,
        tokens: int,
        *,
        domain: Optional[int] = None,
    ) -> _PrefillWorkReservation:
        tokens = max(1, int(tokens))
        async with self._prefill_work_lock:
            if domain is None:
                count = len(self.prefill_urls)
                start = self._prefill_work_tiebreak % count
                domain = min(
                    range(count),
                    key=lambda candidate: (
                        self._prefill_pending_tokens[candidate],
                        self._prefill_pending_requests[candidate],
                        (candidate - start) % count,
                    ),
                )
                self._prefill_work_tiebreak = (domain + 1) % count
            if not 0 <= domain < len(self.prefill_urls):
                raise RuntimeError(f"invalid Prefill domain {domain}")
            self._prefill_pending_tokens[domain] += tokens
            self._prefill_pending_requests[domain] += 1
            reservation = _PrefillWorkReservation(domain=domain, tokens=tokens)
            logger.info(
                "PD_P_WORK_RESERVE P=%d tokens=%d pending_tokens=%d "
                "pending_requests=%d",
                domain,
                tokens,
                self._prefill_pending_tokens[domain],
                self._prefill_pending_requests[domain],
            )
            return reservation

    async def _move_prefill_work(
        self, reservation: _PrefillWorkReservation, domain: int
    ) -> None:
        if not 0 <= domain < len(self.prefill_urls):
            raise RuntimeError(f"invalid Prefill domain {domain}")
        if reservation.released or reservation.domain == domain:
            return
        async with self._prefill_work_lock:
            if reservation.released or reservation.domain == domain:
                return
            previous = reservation.domain
            self._prefill_pending_tokens[previous] -= reservation.tokens
            self._prefill_pending_requests[previous] -= reservation.requests
            self._prefill_pending_tokens[domain] += reservation.tokens
            self._prefill_pending_requests[domain] += reservation.requests
            reservation.domain = domain
            logger.info(
                "PD_P_WORK_MOVE from_P=%d to_P=%d tokens=%d pending_tokens=%s",
                previous,
                domain,
                reservation.tokens,
                self._prefill_pending_tokens,
            )

    async def _resize_prefill_work(
        self, reservation: _PrefillWorkReservation, tokens: int
    ) -> None:
        """Correct shadow accounting when reuse falls back to recompute."""

        tokens = max(1, int(tokens))
        if reservation.released or reservation.tokens == tokens:
            return
        async with self._prefill_work_lock:
            if reservation.released or reservation.tokens == tokens:
                return
            domain = reservation.domain
            previous = reservation.tokens
            self._prefill_pending_tokens[domain] += tokens - previous
            reservation.tokens = tokens
            logger.info(
                "PD_P_WORK_RESIZE P=%d old_tokens=%d tokens=%d "
                "pending_tokens=%d",
                domain,
                previous,
                tokens,
                self._prefill_pending_tokens[domain],
            )

    async def _release_prefill_work(
        self, reservation: Optional[_PrefillWorkReservation]
    ) -> None:
        if reservation is None or reservation.released:
            return
        async with self._prefill_work_lock:
            if reservation.released:
                return
            domain = reservation.domain
            self._prefill_pending_tokens[domain] -= reservation.tokens
            self._prefill_pending_requests[domain] -= reservation.requests
            reservation.released = True
            if (
                self._prefill_pending_tokens[domain] < 0
                or self._prefill_pending_requests[domain] < 0
            ):
                raise RuntimeError("Prefill shadow queue accounting underflow")
            logger.info(
                "PD_P_WORK_RELEASE P=%d tokens=%d pending_tokens=%d "
                "pending_requests=%d",
                domain,
                reservation.tokens,
                self._prefill_pending_tokens[domain],
                self._prefill_pending_requests[domain],
            )

    def _bind_prefill_domain(
        self, request: dict[str, Any], domain: int
    ) -> str:
        prefill_server = self.prefill_urls[domain]
        parsed = urllib.parse.urlparse(prefill_server)
        hostname = parsed.hostname
        if hostname is None:
            raise ValueError(f"Invalid P URL {prefill_server!r}")
        port = self.prefill_bootstrap_ports[domain]
        batch = isinstance(request.get("bootstrap_room"), list)
        request["bootstrap_host"] = (
            [hostname] * len(request["bootstrap_room"]) if batch else hostname
        )
        request["bootstrap_port"] = (
            [port] * len(request["bootstrap_room"]) if batch else port
        )
        return prefill_server

    def _domain_decode_urls(self, domain: int) -> set[str]:
        if (
            not getattr(self, "numa_domains", False)
            or getattr(self, "global_decode", False)
        ):
            return set(self.decode_urls)
        width = len(self.decode_urls) // len(self.prefill_urls)
        return set(self.decode_urls[domain * width : (domain + 1) * width])

    async def _resolve_dynamic_prefill_work(
        self,
        request: dict[str, Any],
        metadata: Optional[AgenticRequestMetadata],
        arrival_at: Optional[float],
    ) -> _PrefillWorkReservation:
        """Choose P as soon as DIRECT_READY is visible.

        Returning here is intentional: request admission and Direct receive
        must progress concurrently.  A later NUMA-local Host transition is
        handled by the dispatch-time redirect watcher.
        """

        parent = None if metadata is None else metadata.parent
        store = self.early_claim_store
        if parent is None or store is None:
            return await self._reserve_prefill_work(
                self._estimated_prefill_tokens(request)
            )

        deadline = time.monotonic() + self.ready_timeout
        reservation: Optional[_PrefillWorkReservation] = None
        try:
            while True:
                route = store.read_route(
                    parent,
                    max_age_seconds=max(
                        self.ready_timeout,
                        getattr(self, "generation_result_ttl", 3600.0),
                    ),
                )
                if route is not None:
                    mode = route.get("route")
                    snapshot_tokens = route.get("snapshot_tokens")
                    if mode == "direct_ready":
                        if reservation is None:
                            reservation = await self._reserve_prefill_work(
                                self._estimated_prefill_tokens(
                                    request, snapshot_tokens
                                )
                            )
                            self._publish_parent_arrival(
                                request,
                                target_prefill_domain=reservation.domain,
                                arrived_at=arrival_at,
                            )
                            logger.info(
                                "PD_PREFILL_ROUTE snapshot=%s route=%s "
                                "selected_P=%d estimated_tokens=%d",
                                parent.snapshot_id,
                                mode,
                                reservation.domain,
                                reservation.tokens,
                            )
                        reservation.route_pending = True
                        return reservation
                    elif mode in {
                        "direct_complete",
                        "host_writing",
                        "host_ready",
                    }:
                        domain = int(route["prefill_domain"])
                        if not 0 <= domain < len(self.prefill_urls):
                            raise RuntimeError(f"invalid Prefill domain {domain}")
                        if reservation is None:
                            reservation = await self._reserve_prefill_work(
                                self._estimated_prefill_tokens(
                                    request, snapshot_tokens
                                ),
                                domain=domain,
                            )
                        else:
                            await self._move_prefill_work(reservation, domain)
                        self._publish_parent_arrival(
                            request,
                            target_prefill_domain=domain,
                            arrived_at=arrival_at,
                        )
                        logger.info(
                            "PD_PREFILL_ROUTE snapshot=%s route=%s P=%d "
                            "estimated_tokens=%d",
                            parent.snapshot_id,
                            mode,
                            domain,
                            reservation.tokens,
                        )
                        return reservation
                    elif mode == "recompute":
                        store.remove_arrival(parent)
                        await self._release_prefill_work(reservation)
                        return await self._reserve_prefill_work(
                            self._request_input_tokens(request)
                        )
                if store.read_final(
                    parent, not_before=0.0, max_age_seconds=self.ready_timeout
                ) is not None:
                    store.remove_arrival(parent)
                    await self._release_prefill_work(reservation)
                    return await self._reserve_prefill_work(
                        self._request_input_tokens(request)
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for P route of {parent.snapshot_id}"
                    )
                await asyncio.sleep(self.ready_poll_interval)
        except BaseException:
            await self._release_prefill_work(reservation)
            raise

    async def _watch_dynamic_prefill_route(
        self,
        request: dict[str, Any],
        metadata: AgenticRequestMetadata,
        reservation: _PrefillWorkReservation,
    ) -> dict[str, Any]:
        """Observe the Direct outcome without delaying the first P submit."""

        parent = metadata.parent
        store = self.early_claim_store
        if parent is None or store is None:
            return {"action": "settled", "route": "none"}
        deadline = time.monotonic() + self.ready_timeout
        while True:
            route = store.read_route(
                parent,
                max_age_seconds=max(
                    self.ready_timeout,
                    getattr(self, "generation_result_ttl", 3600.0),
                ),
            )
            if route is not None:
                mode = route.get("route")
                if mode in {"direct_complete", "host_writing", "host_ready"}:
                    domain = int(route["prefill_domain"])
                    if not 0 <= domain < len(self.prefill_urls):
                        raise RuntimeError(f"invalid Prefill domain {domain}")
                    if domain != reservation.domain:
                        logger.info(
                            "PD_PREFILL_REDIRECT snapshot=%s route=%s "
                            "from_P=%d to_P=%d",
                            parent.snapshot_id,
                            mode,
                            reservation.domain,
                            domain,
                        )
                        return {
                            "action": "redirect",
                            "route": mode,
                            "domain": domain,
                        }
                    return {"action": "settled", "route": mode, "domain": domain}
                if mode == "recompute":
                    store.remove_arrival(parent)
                    await self._resize_prefill_work(
                        reservation, self._request_input_tokens(request)
                    )
                    return {"action": "recompute", "route": mode}
            if store.read_final(
                parent, not_before=0.0, max_age_seconds=self.ready_timeout
            ) is not None:
                store.remove_arrival(parent)
                await self._resize_prefill_work(
                    reservation, self._request_input_tokens(request)
                )
                return {"action": "recompute", "route": "final"}
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for P route of {parent.snapshot_id}"
                )
            await asyncio.sleep(self.ready_poll_interval)

    @staticmethod
    def _raise_prefill_redirect(route_task: Optional[asyncio.Task]) -> None:
        if route_task is None or not route_task.done():
            return
        outcome = route_task.result()
        if outcome.get("action") == "redirect":
            raise _PrefillRedirect(int(outcome["domain"]), str(outcome["route"]))

    @staticmethod
    def _set_prefill_attempt_rid(request: dict[str, Any], *, replace: bool) -> None:
        rid = request.get("rid")
        batch = isinstance(request.get("bootstrap_room"), list)
        if not replace and rid is not None:
            return
        if batch:
            request["rid"] = [uuid.uuid4().hex for _ in request["bootstrap_room"]]
        else:
            request["rid"] = uuid.uuid4().hex

    @staticmethod
    def _replace_prefill_attempt_rooms(request: dict[str, Any]) -> None:
        room = request.get("bootstrap_room")
        if isinstance(room, list):
            request["bootstrap_room"] = [
                uuid.uuid4().int & ((1 << 63) - 1) for _ in room
            ]
        else:
            request["bootstrap_room"] = uuid.uuid4().int & ((1 << 63) - 1)

    async def _abort_prefill_attempt(
        self,
        session: aiohttp.ClientSession,
        prefill_server: str,
        request: dict[str, Any],
        prefill_task: Optional[asyncio.Task],
    ) -> bool:
        rids = request.get("rid")
        if not isinstance(rids, list):
            rids = [] if rids is None else [rids]
        aborted = bool(rids)
        for rid in rids:
            try:
                response = await session.post(
                    f"{prefill_server}/abort_request", json={"rid": rid}
                )
                if response.status >= 400:
                    aborted = False
                    logger.warning(
                        "PD_PREFILL_REDIRECT abort failed P=%s rid=%s status=%d",
                        prefill_server,
                        rid,
                        response.status,
                    )
                release = getattr(response, "release", None)
                if release is not None:
                    release()
            except Exception:
                aborted = False
                logger.exception(
                    "PD_PREFILL_REDIRECT abort request failed P=%s rid=%s",
                    prefill_server,
                    rid,
                )
        if prefill_task is not None and not prefill_task.done():
            prefill_task.cancel()
            await asyncio.gather(prefill_task, return_exceptions=True)
        return aborted

    def _scheduled_path(self, room: int) -> Path:
        return self.p_ready_dir / f"{room}.scheduled"

    async def _wait_until_prefill_accepted(
        self,
        rooms: tuple[int, ...],
        prefill_task: asyncio.Task,
        route_task: Optional[asyncio.Task] = None,
    ) -> None:
        deadline = time.monotonic() + self.prefill_accept_timeout
        while True:
            self._raise_prefill_redirect(route_task)
            paths = [self._accepted_path(room) for room in rooms]
            if all(path.exists() for path in paths):
                for path in paths:
                    path.unlink(missing_ok=True)
                return
            if prefill_task.done():
                response = prefill_task.result()
                if response.status >= 400:
                    body = await response.text()
                    raise RuntimeError(
                        "Prefill failed before scheduler acceptance: "
                        f"status={response.status} body={body}"
                    )
            if time.monotonic() >= deadline:
                missing = [room for room, path in zip(rooms, paths) if not path.exists()]
                raise TimeoutError(
                    "Timed out waiting "
                    f"{self.prefill_accept_timeout}s for P-accepted rooms {missing}"
                )
            await asyncio.sleep(self.ready_poll_interval)

    async def _wait_until_prefill_scheduled(
        self,
        rooms: tuple[int, ...],
        prefill_task: asyncio.Task,
        route_task: Optional[asyncio.Task] = None,
    ) -> None:
        deadline = time.monotonic() + self.prefill_queue_timeout
        while True:
            self._raise_prefill_redirect(route_task)
            paths = [self._scheduled_path(room) for room in rooms]
            if all(path.exists() for path in paths):
                for path in paths:
                    path.unlink(missing_ok=True)
                return
            if prefill_task.done():
                response = prefill_task.result()
                if response.status >= 400:
                    body = await response.text()
                    raise RuntimeError(
                        "Prefill failed while queued: "
                        f"status={response.status} body={body}"
                    )
            if time.monotonic() >= deadline:
                missing = [room for room, path in zip(rooms, paths) if not path.exists()]
                raise TimeoutError(
                    "Timed out waiting "
                    f"{self.prefill_queue_timeout}s in P queue for rooms {missing}"
                )
            await asyncio.sleep(self.ready_poll_interval)

    def _publish_parent_arrival(
        self,
        request: dict[str, Any],
        *,
        target_prefill_domain: Optional[int] = None,
        arrived_at: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        """Publish next-turn arrival before P's GPU scheduler sees the request."""

        store = self.early_claim_store
        if store is None:
            return None
        sampling = request.get("sampling_params")
        custom_params = (
            sampling.get("custom_params") if isinstance(sampling, dict) else None
        )
        try:
            metadata = AgenticRequestMetadata.from_custom_params(custom_params)
            if metadata is None:
                envelope = unpack_agentic_extra_key(request.get("extra_key"))
                if envelope is not None:
                    _, custom_params = envelope
                    metadata = AgenticRequestMetadata.from_custom_params(custom_params)
        except (TypeError, ValueError):
            logger.warning("Ignoring malformed agentic early-claim metadata")
            return None
        if metadata is None or metadata.parent is None:
            return None
        try:
            payload = store.publish_arrival(
                metadata.parent,
                target_prefill_domain=target_prefill_domain,
                arrived_at=arrived_at,
            )
        except OSError:
            # Early claim is a performance hint.  A marker write failure must
            # not turn an otherwise valid serving request into an HTTP error.
            logger.exception(
                "Failed to publish agentic early-claim marker for snapshot=%s",
                metadata.parent.snapshot_id,
            )
            return None
        logger.info(
            "PD_EARLY_CLAIM_ARRIVAL snapshot=%s generation=%d arrived_at=%.6f P=%s",
            metadata.parent.snapshot_id,
            metadata.generation,
            payload["arrived_at"],
            target_prefill_domain,
        )
        return payload

    @staticmethod
    def _agentic_generation_key(request: dict[str, Any]) -> Optional[str]:
        sampling = request.get("sampling_params")
        custom_params = (
            sampling.get("custom_params") if isinstance(sampling, dict) else None
        )
        try:
            metadata = AgenticRequestMetadata.from_custom_params(custom_params)
            if metadata is None:
                envelope = unpack_agentic_extra_key(request.get("extra_key"))
                if envelope is not None:
                    _, params = envelope
                    metadata = AgenticRequestMetadata.from_custom_params(params)
        except (TypeError, ValueError):
            return None
        return None if metadata is None else metadata.current.snapshot_id

    def _ensure_generation_dedup_state(self) -> None:
        """Support focused tests that construct the router via ``__new__``."""

        if not hasattr(self, "_generation_lock"):
            self._generation_lock = asyncio.Lock()
            self._generation_tasks = {}
            self._generation_results = {}
            self.generation_result_ttl = 3600.0
            self.max_generation_results = 8192

    def _prune_generation_results(self, now: float) -> None:
        expired = [
            key
            for key, result in self._generation_results.items()
            if now - result.completed_at > self.generation_result_ttl
        ]
        for key in expired:
            self._generation_results.pop(key, None)
        overflow = len(self._generation_results) - self.max_generation_results
        if overflow > 0:
            oldest = sorted(
                self._generation_results.items(),
                key=lambda item: item[1].completed_at,
            )[:overflow]
            for key, _ in oldest:
                self._generation_results.pop(key, None)

    async def _record_generation_task(
        self,
        key: str,
        task: asyncio.Task[tuple[dict[str, Any], int]],
    ) -> None:
        try:
            payload, status = task.result()
        except BaseException:
            async with self._generation_lock:
                if self._generation_tasks.get(key) is task:
                    self._generation_tasks.pop(key, None)
            return
        async with self._generation_lock:
            if self._generation_tasks.get(key) is task:
                self._generation_tasks.pop(key, None)
                self._generation_results[key] = _GenerationResponse(
                    payload=payload,
                    status=status,
                    completed_at=time.monotonic(),
                )
                self._prune_generation_results(time.monotonic())

    async def _generate_once(
        self,
        modified_request: dict[str, Any],
        prefill_server: str,
        endpoint: str,
    ) -> tuple[dict[str, Any], int]:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        ) as session:
            prefill_response, decode_response = await self._late_dispatch(
                session, modified_request, prefill_server, endpoint, {}
            )
            if "return_logprob" in modified_request:
                prefill_json = await prefill_response.json()
                ret_json = await decode_response.json()
                if (
                    "meta_info" in ret_json
                    and "input_token_logprobs" in ret_json["meta_info"]
                ):
                    ret_json["meta_info"]["input_token_logprobs"] = (
                        prefill_json["meta_info"]["input_token_logprobs"]
                        + ret_json["meta_info"]["input_token_logprobs"]
                    )
            else:
                ret_json = await decode_response.json()
            return ret_json, decode_response.status

    async def _generate_singleflight(
        self,
        key: str,
        modified_request: dict[str, Any],
        prefill_server: str,
        endpoint: str,
    ) -> tuple[dict[str, Any], int]:
        self._ensure_generation_dedup_state()
        now = time.monotonic()
        async with self._generation_lock:
            self._prune_generation_results(now)
            completed = self._generation_results.get(key)
            if completed is not None:
                logger.info("PD_GENERATION_DEDUP cache_hit generation=%s", key)
                return completed.payload, completed.status
            task = self._generation_tasks.get(key)
            if task is None:
                # The request is mutated while binding its P/NUMA destination;
                # isolate the elected producer from a retry's request object.
                request_copy = orjson.loads(orjson.dumps(modified_request))
                task = asyncio.create_task(
                    self._generate_once(request_copy, prefill_server, endpoint)
                )
                self._generation_tasks[key] = task
                task.add_done_callback(
                    lambda done, generation=key: asyncio.create_task(
                        self._record_generation_task(generation, done)
                    )
                )
                logger.info("PD_GENERATION_DEDUP producer generation=%s", key)
            else:
                logger.warning("PD_GENERATION_DEDUP joined_retry generation=%s", key)
        # Client cancellation must not cancel the elected serving operation.
        return await asyncio.shield(task)

    async def _wait_until_prefill_ready(
        self,
        rooms: tuple[int, ...],
        prefill_task: asyncio.Task,
        route_task: Optional[asyncio.Task] = None,
    ) -> int:
        if getattr(self, "dynamic_prefill_domains", False):
            return await self._wait_until_prefill_ready_shared(
                rooms, prefill_task, route_task
            )
        deadline = time.monotonic() + self.ready_timeout
        while True:
            paths = [self._ready_path(room) for room in rooms]
            if all(path.exists() for path in paths):
                prompt_tokens = 0
                for path in paths:
                    try:
                        payload = orjson.loads(path.read_bytes())
                    except (OSError, orjson.JSONDecodeError):
                        # Backward compatibility with the old marker containing
                        # only a request id.  Late binding still works, but its
                        # local token reservation is conservative/unknown.
                        continue
                    prompt_tokens += int(payload.get("num_kv_tokens", 0))
                return prompt_tokens
            self._raise_prefill_redirect(route_task)

            # A non-streaming P response should not finish before transfer.  A
            # streaming response can return headers early, so only treat an
            # already returned HTTP error as terminal.
            if prefill_task.done():
                try:
                    response = prefill_task.result()
                except Exception:
                    raise
                if response.status >= 400:
                    body = await response.text()
                    raise RuntimeError(
                        f"Prefill failed before P-ready: status={response.status} body={body}"
                    )

            if time.monotonic() >= deadline:
                missing = [room for room, path in zip(rooms, paths) if not path.exists()]
                raise TimeoutError(
                    f"Timed out waiting {self.ready_timeout}s for P-ready rooms {missing}"
                )
            await asyncio.sleep(self.ready_poll_interval)

    def _scan_p_ready_markers(self) -> dict[int, dict[str, Any]]:
        snapshot: dict[int, dict[str, Any]] = {}
        for path in self.p_ready_dir.glob("*.ready"):
            try:
                room = int(path.stem)
                payload = orjson.loads(path.read_bytes())
                if not isinstance(payload, dict):
                    continue
                payload["_path"] = path
                snapshot[room] = payload
            except (OSError, orjson.JSONDecodeError, TypeError, ValueError):
                # Atomic rename makes partial files impossible, but D may
                # unlink a marker between glob and read.  The next scan is the
                # authoritative state in either case.
                continue
        return snapshot

    async def _p_ready_monitor_loop(self) -> None:
        try:
            while True:
                snapshot = await asyncio.to_thread(self._scan_p_ready_markers)
                self._p_ready_snapshot = snapshot
                for room, futures in list(self._p_ready_waiters.items()):
                    payload = snapshot.get(room)
                    if payload is None:
                        continue
                    for future in tuple(futures):
                        if not future.done():
                            future.set_result(payload)
                await asyncio.sleep(self.ready_poll_interval)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Shared P-ready watcher failed")
            raise

    def _ensure_p_ready_monitor(self) -> None:
        task = self._p_ready_monitor_task
        if task is None or task.done():
            self._p_ready_monitor_task = asyncio.create_task(
                self._p_ready_monitor_loop(), name="pd-p-ready-monitor"
            )

    async def _wait_until_prefill_ready_shared(
        self,
        rooms: tuple[int, ...],
        prefill_task: asyncio.Task,
        route_task: Optional[asyncio.Task] = None,
    ) -> int:
        self._ensure_p_ready_monitor()
        loop = asyncio.get_running_loop()
        futures: dict[int, asyncio.Future] = {}
        for room in rooms:
            future = loop.create_future()
            futures[room] = future
            self._p_ready_waiters.setdefault(room, set()).add(future)
            payload = self._p_ready_snapshot.get(room)
            if payload is not None:
                future.set_result(payload)

        deadline = time.monotonic() + self.ready_timeout
        try:
            while True:
                if all(future.done() for future in futures.values()):
                    return sum(
                        int(future.result().get("num_kv_tokens", 0))
                        for future in futures.values()
                    )
                self._raise_prefill_redirect(route_task)
                if prefill_task.done():
                    response = prefill_task.result()
                    if response.status >= 400:
                        body = await response.text()
                        raise RuntimeError(
                            "Prefill failed before P-ready: "
                            f"status={response.status} body={body}"
                        )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    missing = [
                        room for room, future in futures.items() if not future.done()
                    ]
                    raise TimeoutError(
                        f"Timed out waiting {self.ready_timeout}s for P-ready rooms {missing}"
                    )
                pending = [future for future in futures.values() if not future.done()]
                await asyncio.wait(
                    pending,
                    timeout=min(remaining, 0.25),
                    return_when=asyncio.FIRST_COMPLETED,
                )
        finally:
            for room, future in futures.items():
                waiters = self._p_ready_waiters.get(room)
                if waiters is None:
                    continue
                waiters.discard(future)
                if not waiters:
                    self._p_ready_waiters.pop(room, None)

    def _p_ready_sequence(self, rooms: tuple[int, ...]) -> int:
        """Read P's authoritative completion sequence from ready markers."""

        sequences: list[int] = []
        for room in rooms:
            path = self._ready_path(room)
            try:
                payload = getattr(self, "_p_ready_snapshot", {}).get(room)
                if payload is None:
                    payload = orjson.loads(path.read_bytes())
                sequence = payload.get("ready_sequence")
                if sequence is not None:
                    sequences.append(int(sequence))
                    continue
            except (OSError, orjson.JSONDecodeError, TypeError, ValueError):
                pass
            # Compatibility with markers produced before ready_sequence was
            # added.  Dedicated run directories make mtime a stable fallback.
            sequences.append(path.stat().st_mtime_ns)
        return min(sequences)

    def _oldest_p_ready_sequence(self, domain: int = 0) -> Optional[int]:
        sequences: list[int] = []
        if getattr(self, "dynamic_prefill_domains", False):
            markers = tuple(getattr(self, "_p_ready_snapshot", {}).values())
        else:
            markers = tuple(self.p_ready_dir.glob("*.ready"))
        for marker in markers:
            try:
                if isinstance(marker, dict):
                    payload = marker
                    path = payload.get("_path")
                else:
                    path = marker
                    payload = orjson.loads(path.read_bytes())
                if getattr(self, "dynamic_prefill_domains", False) and int(
                    payload.get("prefill_domain", -1)
                ) != domain:
                    continue
                sequence = payload.get("ready_sequence")
                sequence = (
                    int(sequence)
                    if sequence is not None
                    else path.stat().st_mtime_ns
                )
                key = (
                    (domain, sequence)
                    if getattr(self, "dynamic_prefill_domains", False)
                    else sequence
                )
                if key not in self._p_ready_submitted_sequences:
                    sequences.append(sequence)
            except (OSError, orjson.JSONDecodeError, TypeError, ValueError):
                continue
        return min(sequences) if sequences else None

    async def _acquire_p_ready_fifo(
        self, sequence: int, domain: int = 0
    ) -> asyncio.Lock:
        """Acquire the dispatch lock only for the oldest published P result."""

        if getattr(self, "dynamic_prefill_domains", False):
            locks = getattr(self, "_p_ready_fifo_locks", None)
            if locks is None:
                self._p_ready_fifo_locks = {}
                locks = self._p_ready_fifo_locks
            lock = locks.setdefault(domain, asyncio.Lock())
            key: Any = (domain, sequence)
        else:
            lock = self._p_ready_fifo_lock
            key = sequence
        while True:
            await lock.acquire()
            oldest = self._oldest_p_ready_sequence(domain)
            if oldest is None or sequence <= oldest:
                self._p_ready_submitted_sequences.add(key)
                return lock
            lock.release()
            await asyncio.sleep(self.ready_poll_interval)

    @staticmethod
    def _requested_decode_tokens(request: dict[str, Any]) -> Optional[int]:
        for key in ("max_new_tokens", "max_completion_tokens", "max_tokens"):
            value = request.get(key)
            if value is not None:
                try:
                    return max(0, int(value))
                except (TypeError, ValueError):
                    return None
        sampling = request.get("sampling_params")
        if isinstance(sampling, dict) and sampling.get("max_new_tokens") is not None:
            try:
                return max(0, int(sampling["max_new_tokens"]))
            except (TypeError, ValueError):
                return None
        return None

    async def _fetch_decode_load(
        self, session: aiohttp.ClientSession, url: str
    ) -> DecodeLoad:
        if url in self._legacy_load_urls:
            return await self._fetch_decode_load_legacy(session, url)
        endpoint = f"{url}/v1/loads?include=disagg,memory,queues"
        try:
            async with session.get(
                endpoint, timeout=aiohttp.ClientTimeout(total=self.load_timeout)
            ) as response:
                response.raise_for_status()
                payload = await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            # SGLang 0.5.10 has a known-in-this-environment FastAPI middleware
            # incompatibility where an included APIRouter can make /v1/loads
            # return 500.  /get_load comes from the same scheduler snapshot and
            # includes physical + queued/prealloc/transfer token demand.
            self._legacy_load_urls.add(url)
            return await self._fetch_decode_load_legacy(session, url)

        loads = payload.get("loads") or []
        if not loads:
            raise RuntimeError(f"D load endpoint returned no ranks: {url}")
        physical_used = sum(int(load.get("num_used_tokens", 0)) for load in loads)
        capacity = sum(int(load.get("max_total_num_tokens", 0)) for load in loads)
        running = sum(int(load.get("num_running_reqs", 0)) for load in loads)
        waiting = sum(int(load.get("num_waiting_reqs", 0)) for load in loads)
        max_running = sum(int(load.get("max_running_requests", 0)) for load in loads)
        prealloc = 0
        transfer = 0
        prealloc_tokens = 0
        transfer_tokens = 0
        for load in loads:
            disagg = load.get("disaggregation") or {}
            prealloc += int(disagg.get("decode_prealloc_queue_reqs", 0))
            transfer += int(disagg.get("decode_transfer_queue_reqs", 0))
            prealloc_tokens += int(disagg.get("decode_prealloc_queue_tokens", 0))
            transfer_tokens += int(disagg.get("decode_transfer_queue_tokens", 0))
        if capacity <= 0:
            raise RuntimeError(f"D load endpoint returned invalid KV capacity: {url}")
        running_kv_tokens = sum(
            int(load.get("num_running_kv_tokens", 0)) for load in loads
        )
        # SGLang's num_used_tokens is already the non-evictable occupancy:
        # max_total - (allocator.available + radix.evictable).  In particular,
        # it also includes protected direct/Host-staging snapshots that are not
        # represented by running_kv_tokens.  Those pages cannot be promised to
        # a new P->D handoff, so use the scheduler's value as the hard bound.
        hard_used = physical_used
        return DecodeLoad(
            url=url,
            used_tokens=hard_used,
            capacity_tokens=capacity,
            running=running,
            waiting=waiting,
            prealloc=prealloc,
            transfer=transfer,
            max_running=max_running,
            running_kv_tokens=running_kv_tokens,
            prealloc_tokens=prealloc_tokens,
            transfer_tokens=transfer_tokens,
            physical_used_tokens=physical_used,
        )

    async def _fetch_decode_load_legacy(
        self, session: aiohttp.ClientSession, url: str
    ) -> DecodeLoad:
        timeout = aiohttp.ClientTimeout(total=self.load_timeout)
        async with session.get(f"{url}/get_load", timeout=timeout) as response:
            response.raise_for_status()
            rows = await response.json()
        if isinstance(rows, dict):
            rows = [rows]
        if not rows:
            raise RuntimeError(f"D legacy load endpoint returned no ranks: {url}")

        reported_capacity = sum(
            int(row.get("max_total_num_tokens", 0)) for row in rows
        )
        reported_max_running = sum(
            int(row.get("max_running_requests", 0)) for row in rows
        )
        cached = self._last_loads.get(url)
        if reported_capacity > 0:
            capacity = reported_capacity
            max_running = reported_max_running
        elif cached is not None:
            capacity = cached.capacity_tokens
            max_running = cached.max_running
        else:
            async with session.get(f"{url}/server_info", timeout=timeout) as response:
                response.raise_for_status()
                server_info = await response.json()
            internal = server_info.get("internal_states") or []
            capacity = sum(
                int((state.get("memory_usage") or {}).get("token_capacity", 0))
                for state in internal
            )
            max_running = sum(
                int(state.get("effective_max_running_requests_per_dp", 0))
                for state in internal
            )
        if capacity <= 0:
            raise RuntimeError(f"D server_info returned invalid KV capacity: {url}")

        total_reqs = sum(int(row.get("num_reqs", 0)) for row in rows)
        waiting = sum(int(row.get("num_waiting_reqs", 0)) for row in rows)
        prealloc = sum(int(row.get("decode_prealloc_queue_reqs", 0)) for row in rows)
        transfer = sum(int(row.get("decode_transfer_queue_reqs", 0)) for row in rows)
        physical_used = sum(
            int(row.get("num_physical_used_tokens", row.get("num_tokens", 0)))
            for row in rows
        )
        running_kv_tokens = sum(
            int(row.get("num_running_kv_tokens", 0)) for row in rows
        )
        prealloc_tokens = sum(
            int(row.get("decode_prealloc_queue_tokens", 0)) for row in rows
        )
        transfer_tokens = sum(
            int(row.get("decode_transfer_queue_tokens", 0)) for row in rows
        )
        return DecodeLoad(
            url=url,
            # Despite the compatibility field's historical name, SGLang fills
            # num_physical_used_tokens from _get_token_info()[0], which already
            # excludes evictable Radix pages and includes protected snapshots.
            used_tokens=physical_used,
            capacity_tokens=capacity,
            running=max(0, total_reqs - waiting),
            waiting=waiting,
            prealloc=prealloc if prealloc or transfer else waiting,
            transfer=transfer,
            max_running=max_running,
            running_kv_tokens=running_kv_tokens,
            prealloc_tokens=prealloc_tokens,
            transfer_tokens=transfer_tokens,
            physical_used_tokens=physical_used,
        )

    async def _refresh_decode_loads(
        self, session: aiohttp.ClientSession
    ) -> list[DecodeLoad]:
        results = await asyncio.gather(
            *(self._fetch_decode_load(session, url) for url in self.decode_urls),
            return_exceptions=True,
        )
        loads: list[DecodeLoad] = []
        for url, result in zip(self.decode_urls, results):
            if isinstance(result, Exception):
                cached = self._last_loads.get(url)
                if cached is None:
                    logger.warning("Skipping D with unavailable load endpoint %s: %s", url, result)
                    continue
                logger.warning("Using last load snapshot for D %s: %s", url, result)
                loads.append(cached)
            else:
                self._last_loads[url] = result
                loads.append(result)
        if not loads:
            raise RuntimeError("No decode server has a usable /v1/loads response")
        self._load_cache = loads
        self._load_cache_at = time.monotonic()
        return loads

    async def _refresh_decode_loads_background(self) -> None:
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.load_timeout)
            ) as session:
                await self._refresh_decode_loads(session)
        except Exception:
            # The current cached snapshot plus local reservations remains a
            # conservative admission view.  A later dispatch will retry.
            logger.warning("Background D load refresh failed", exc_info=True)

    async def _all_decode_loads(
        self, session: aiohttp.ClientSession, *, force: bool = False
    ) -> list[DecodeLoad]:
        now = time.monotonic()
        fresh = (
            self._load_cache
            and now - self._load_cache_at < self.load_cache_ttl
        )
        if not force and fresh:
            return self._load_cache

        if not force and self._load_cache:
            # Never put HTTP load polling in the per-request reservation
            # critical path.  Dispatch immediately from the last snapshot and
            # refresh it independently; local token/request reservations make
            # stale admission conservative rather than allowing oversubscribe.
            task = self._load_refresh_task
            if task is None or task.done():
                self._load_refresh_task = asyncio.create_task(
                    self._refresh_decode_loads_background()
                )
            return self._load_cache

        return await self._refresh_decode_loads(session)

    def _reserved_for(
        self, url: str, *, exclude_id: Optional[str] = None
    ) -> tuple[int, int, int]:
        prompt = admission = requests = 0
        for reservation_id, reservation in self._reservations.items():
            if reservation_id == exclude_id:
                continue
            admitted_at = self._admitted_reservation_at.get(reservation_id)
            if admitted_at is not None and self._load_cache_at >= admitted_at:
                continue
            if reservation.url == url:
                prompt += reservation.prompt_tokens
                admission += reservation.admission_tokens
                requests += reservation.request_count
        return prompt, admission, requests

    def _prune_accounted_reservations(self) -> None:
        accounted = [
            reservation_id
            for reservation_id, admitted_at in self._admitted_reservation_at.items()
            if self._load_cache_at >= admitted_at
        ]
        for reservation_id in accounted:
            self._admitted_reservation_at.pop(reservation_id, None)
            self._reservations.pop(reservation_id, None)

    def _average_running_context(self, loads: list[DecodeLoad]) -> float:
        running = sum(load.running for load in loads)
        running_tokens = sum(load.running_kv_tokens for load in loads)
        observed = running_tokens / running if running > 0 and running_tokens > 0 else 4096.0
        floor = max(1, getattr(self, "context_token_floor", 2048))
        ceiling = max(floor, getattr(self, "context_token_ceiling", 8192))
        return min(float(ceiling), max(float(floor), observed))

    async def _select_and_reserve_decode(
        self,
        session: aiohttp.ClientSession,
        request: dict[str, Any],
        rooms: tuple[int, ...],
        prompt_tokens: int,
        domain: int = 0,
    ) -> DecodeReservation:
        requested_decode = self._requested_decode_tokens(request)
        decode_headroom = self.decode_headroom_tokens
        if requested_decode is not None:
            decode_headroom = min(requested_decode, self.max_decode_admission_tokens)
        admission_tokens = prompt_tokens + decode_headroom * len(rooms)

        # This lock makes load observation + local reservation atomic for all
        # concurrent requests handled by this router process.  When every D is
        # full, retain the request as cheap P-ready state and retry; do not turn
        # it into an expensive D preallocation that blocks Decode KV capacity.
        deadline = time.monotonic() + self.ready_timeout
        wait_started = time.monotonic()
        next_wait_log = time.monotonic() + 5.0
        draining_reservation: Optional[DecodeReservation] = None
        try:
            while True:
                async with self._selection_lock:
                    loads = await self._all_decode_loads(session)
                    self._prune_accounted_reservations()
                    allowed_urls = self._domain_decode_urls(domain)
                    loads = [load for load in loads if load.url in allowed_urls]
                    if not loads:
                        raise RuntimeError(f"No usable D worker in domain {domain}")
                    if draining_reservation is not None:
                        selected = next(
                            (
                                load
                                for load in loads
                                if load.url == draining_reservation.url
                            ),
                            None,
                        )
                        if selected is not None:
                            _, reserved_admission, _ = self._reserved_for(
                                selected.url,
                                exclude_id=draining_reservation.reservation_id,
                            )
                            free_after_others = (
                                selected.capacity_tokens
                                - selected.used_tokens
                                - reserved_admission
                            )
                            if free_after_others >= admission_tokens:
                                logger.info(
                                    "PD_LATE_BIND_DRAIN_READY rooms=%s D=%s "
                                    "admission_tokens=%d D_used=%d/%d wait_s=%.3f",
                                    rooms,
                                    selected.url,
                                    admission_tokens,
                                    selected.used_tokens,
                                    selected.capacity_tokens,
                                    time.monotonic() - wait_started,
                                )
                                return draining_reservation

                    # Capacity is a hard admission constraint.  Among workers
                    # that can fit the request, compare request-equivalent
                    # Decode work: population + running/handoff KV normalized
                    # by the cluster's current average context.  A transfer is
                    # counted once as future Decode work and once more as DMA /
                    # scheduler interference (default total weight: 2).
                    average_context = self._average_running_context(loads)
                    transfer_weight = max(
                        1.0, getattr(self, "transfer_request_weight", 2.0)
                    )
                    scored: list[
                        tuple[bool, float, float, int, DecodeLoad]
                    ] = []
                    for load in loads:
                        reserved_prompt, reserved_admission, reserved_reqs = (
                            self._reserved_for(load.url)
                        )
                        free_after_pending = (
                            load.capacity_tokens - load.used_tokens - reserved_admission
                        )
                        projected_kv = (
                            load.used_tokens
                            + reserved_admission
                            + admission_tokens
                        ) / load.capacity_tokens
                        # Preserve a small D-side growth/egress margin.
                        # P remains work-conserving and may accumulate complete
                        # P-ready snapshots; only P->D admission pauses here.
                        # The default 90% target leaves room for Decode growth
                        # and completed-parent egress without unnecessarily
                        # suppressing D running concurrency.
                        target_kv = getattr(
                            self, "target_decode_kv_fraction", 1.0
                        )
                        feasible = (
                            free_after_pending >= admission_tokens
                            and projected_kv <= target_kv
                        )
                        # /get_load reports prealloc as waiting, whereas the
                        # richer endpoint exposes handoff queues separately.
                        # max() avoids double-counting either representation.
                        queued_or_handoff_reqs = max(
                            load.waiting, load.prealloc + load.transfer
                        )
                        projected_decode_reqs = (
                            load.running
                            + queued_or_handoff_reqs
                            + reserved_reqs
                            + len(rooms)
                        )
                        projected_compute_kv = (
                            load.running_kv_tokens
                            + load.prealloc_tokens
                            + load.transfer_tokens
                            + reserved_prompt
                            + prompt_tokens
                        )
                        kv_request_equivalents = (
                            projected_compute_kv / average_context
                        )
                        work_score = (
                            projected_decode_reqs
                            + kv_request_equivalents
                            + (transfer_weight - 1.0) * load.transfer
                        )
                        scored.append(
                            (
                                not feasible,
                                work_score,
                                projected_kv,
                                projected_decode_reqs,
                                load,
                            )
                        )

                    feasible = [item for item in scored if not item[0]]
                    if draining_reservation is None and (
                        feasible or not self.wait_for_feasible_decode
                    ):
                        candidates = feasible or scored
                        (
                            _,
                            work_score,
                            projected_kv,
                            projected_decode_reqs,
                            selected,
                        ) = min(
                            candidates,
                            key=lambda item: (item[1], item[2], item[4].url),
                        )
                        reservation = DecodeReservation(
                            reservation_id=uuid.uuid4().hex,
                            url=selected.url,
                            prompt_tokens=prompt_tokens,
                            admission_tokens=admission_tokens,
                            request_count=len(rooms),
                            rooms=rooms,
                            created_at=time.monotonic(),
                        )
                        self._reservations[reservation.reservation_id] = reservation
                        logger.info(
                            "PD_LATE_BIND rooms=%s D=%s prompt_tokens=%d admission_tokens=%d "
                            "D_used=%d/%d running=%d waiting=%d prealloc=%d transfer=%d "
                            "running_kv_tokens=%d prealloc_tokens=%d transfer_tokens=%d "
                            "projected_decode_reqs=%d average_context=%.1f "
                            "work_score=%.4f projected_kv=%.4f "
                            "policy=feasible_least_work",
                            rooms,
                            selected.url,
                            prompt_tokens,
                            admission_tokens,
                            selected.used_tokens,
                            selected.capacity_tokens,
                            selected.running,
                            selected.waiting,
                            selected.prealloc,
                            selected.transfer,
                            selected.running_kv_tokens,
                            selected.prealloc_tokens,
                            selected.transfer_tokens,
                            projected_decode_reqs,
                            average_context,
                            work_score,
                            projected_kv,
                        )
                        return reservation

                    # Reserve future, not current, D capacity for one old
                    # request per worker.  This prevents a large P-ready request
                    # from starving forever while later short requests consume
                    # every small gap.  No KV is allocated on D at this point.
                    waited = time.monotonic() - wait_started
                    should_soft_reserve = (
                        admission_tokens >= self.soft_reservation_min_tokens
                        and waited >= self.soft_reservation_delay
                    ) or waited >= self.soft_reservation_force_after
                    if draining_reservation is None and should_soft_reserve:
                        draining_urls = {
                            reservation.url
                            for reservation in self._reservations.values()
                            if reservation.draining
                        }
                        drain_candidates = [
                            item for item in scored if item[4].url not in draining_urls
                        ]
                        if drain_candidates:
                            (
                                _,
                                work_score,
                                projected_kv,
                                projected_decode_reqs,
                                selected,
                            ) = min(
                                drain_candidates,
                                # Preserve the old KV-pressure-oriented drain
                                # target so the anti-starvation mechanism is
                                # unchanged by the normal admission policy.
                                key=lambda item: (item[2], item[1], item[4].url),
                            )
                            draining_reservation = DecodeReservation(
                                reservation_id=uuid.uuid4().hex,
                                url=selected.url,
                                prompt_tokens=prompt_tokens,
                                admission_tokens=admission_tokens,
                                request_count=len(rooms),
                                rooms=rooms,
                                created_at=time.monotonic(),
                                draining=True,
                            )
                            self._reservations[
                                draining_reservation.reservation_id
                            ] = draining_reservation
                            logger.info(
                                "PD_LATE_BIND_DRAIN_RESERVE rooms=%s D=%s "
                                "admission_tokens=%d D_used=%d/%d "
                                "projected_decode_reqs=%d projected_kv=%.4f "
                                "work_score=%.4f",
                                rooms,
                                selected.url,
                                admission_tokens,
                                selected.used_tokens,
                                selected.capacity_tokens,
                                projected_decode_reqs,
                                projected_kv,
                                work_score,
                            )

                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting {self.ready_timeout}s for feasible Decode "
                        f"capacity for P-ready rooms {rooms} ({admission_tokens} tokens)"
                    )
                if now >= next_wait_log:
                    logger.info(
                        "PD_LATE_BIND_WAIT rooms=%s admission_tokens=%d: all D workers full",
                        rooms,
                        admission_tokens,
                    )
                    next_wait_log = now + 5.0
                await asyncio.sleep(self.no_capacity_poll_interval)
        except BaseException:
            if draining_reservation is not None:
                async with self._selection_lock:
                    self._reservations.pop(
                        draining_reservation.reservation_id, None
                    )
                    self._load_cache_at = 0.0
            raise

    async def _release_reservation_when_admitted(
        self, reservation: DecodeReservation, ready_key: Optional[Any] = None
    ) -> None:
        deadline = time.monotonic() + self.reservation_timeout
        admitted = False
        try:
            while time.monotonic() < deadline:
                # D removes P-ready only after it has allocated destination KV.
                if all(not self._ready_path(room).exists() for room in reservation.rooms):
                    admitted = True
                    return
                await asyncio.sleep(self.ready_poll_interval)
            logger.warning(
                "Late-binding reservation timed out: rooms=%s D=%s",
                reservation.rooms,
                reservation.url,
            )
        finally:
            if ready_key is not None:
                self._p_ready_submitted_sequences.discard(ready_key)
            async with self._selection_lock:
                if admitted:
                    self._admitted_reservation_at[
                        reservation.reservation_id
                    ] = time.monotonic()
                else:
                    self._reservations.pop(reservation.reservation_id, None)

    async def _late_dispatch(
        self,
        session: aiohttp.ClientSession,
        modified_request: dict[str, Any],
        prefill_server: str,
        endpoint: str,
        headers: dict[str, str],
    ):
        rooms = self._rooms(modified_request)
        sampling = modified_request.get("sampling_params")
        custom_params = sampling.get("custom_params") if isinstance(sampling, dict) else None
        try:
            metadata = AgenticRequestMetadata.from_custom_params(custom_params)
        except (TypeError, ValueError):
            metadata = None
        parent_turn = metadata is not None and metadata.parent is not None
        prefill_work: Optional[_PrefillWorkReservation] = None
        arrival_at: Optional[float] = None
        if getattr(self, "dynamic_prefill_domains", False):
            # Notify D immediately that the tool result has returned.  The
            # marker starts untargeted; Router then charges its local shadow
            # queues and targets the lighter P for Direct.  A failed Direct is
            # moved to the D worker's NUMA-local P when host_ready appears.
            arrival = self._publish_parent_arrival(modified_request)
            arrival_at = (
                None if arrival is None else float(arrival["arrived_at"])
            )
            prefill_work = await self._resolve_dynamic_prefill_work(
                modified_request,
                metadata,
                arrival_at,
            )
            domain = prefill_work.domain
        else:
            self._publish_parent_arrival(modified_request)
            domain = self._request_domain(metadata, rooms)
        self._set_prefill_attempt_rid(modified_request, replace=False)
        route_task: Optional[asyncio.Task] = None
        if (
            prefill_work is not None
            and prefill_work.route_pending
            and metadata is not None
        ):
            route_task = asyncio.create_task(
                self._watch_dynamic_prefill_route(
                    modified_request,
                    metadata,
                    prefill_work,
                )
            )
        prefill_task: Optional[asyncio.Task] = None
        decode_task: Optional[asyncio.Task] = None
        admission_task: Optional[asyncio.Task] = None
        reservation: Optional[DecodeReservation] = None
        ready_sequence: Optional[int] = None
        ready_key: Optional[Any] = None
        fifo_lock: Optional[asyncio.Lock] = None
        try:
            while True:
                if getattr(self, "numa_domains", False):
                    prefill_server = self._bind_prefill_domain(
                        modified_request, domain
                    )
                rooms = self._rooms(modified_request)
                admission_wait = await self._prefill_admission.acquire(
                    parent_turn=parent_turn
                )
                if admission_wait >= 1.0:
                    logger.info(
                        "PD_P_ADMISSION rooms=%s parent_turn=%s wait_s=%.3f "
                        "active=%d limit=%d",
                        rooms,
                        parent_turn,
                        admission_wait,
                        self._prefill_admission.active,
                        self.max_prefill_inflight,
                    )
                prefill_submit_at = time.monotonic()
                prefill_task = asyncio.create_task(
                    session.post(
                        f"{prefill_server}/{endpoint}",
                        json=modified_request,
                        headers=headers,
                    )
                )
                try:
                    await self._wait_until_prefill_accepted(rooms, prefill_task)
                    accept_wait = time.monotonic() - prefill_submit_at
                    if parent_turn and accept_wait >= 0.1:
                        logger.info(
                            "PD_P_PARENT_ACCEPT rooms=%s wait_s=%.3f "
                            "tokenizer_inflight_limit=%d",
                            rooms,
                            accept_wait,
                            self.max_prefill_inflight,
                        )
                finally:
                    await self._prefill_admission.release()
                try:
                    await self._wait_until_prefill_scheduled(
                        rooms, prefill_task, route_task
                    )
                    prompt_tokens = await self._wait_until_prefill_ready(
                        rooms, prefill_task, route_task
                    )
                    break
                except _PrefillRedirect as redirect:
                    old_rooms = rooms
                    old_server = prefill_server
                    aborted = await self._abort_prefill_attempt(
                        session, old_server, modified_request, prefill_task
                    )
                    if not aborted:
                        raise RuntimeError(
                            "Cannot safely redirect Prefill because abort was not "
                            f"acknowledged by {old_server}"
                        )
                    prefill_task = None
                    for room in old_rooms:
                        self._accepted_path(room).unlink(missing_ok=True)
                        self._scheduled_path(room).unlink(missing_ok=True)
                        self._ready_path(room).unlink(missing_ok=True)
                        self._p_ready_snapshot.pop(room, None)
                    await self._move_prefill_work(prefill_work, redirect.domain)
                    self._publish_parent_arrival(
                        modified_request,
                        target_prefill_domain=redirect.domain,
                        arrived_at=arrival_at,
                    )
                    domain = redirect.domain
                    route_task = None
                    self._replace_prefill_attempt_rooms(modified_request)
                    self._set_prefill_attempt_rid(modified_request, replace=True)
                    logger.info(
                        "PD_PREFILL_REDIRECT_SUBMIT route=%s from_P=%s to_P=%d "
                        "old_rooms=%s new_rooms=%s",
                        redirect.route,
                        old_server,
                        domain,
                        old_rooms,
                        self._rooms(modified_request),
                    )
                    continue
            if route_task is not None and not route_task.done():
                route_task.cancel()
                await asyncio.gather(route_task, return_exceptions=True)
            await self._release_prefill_work(prefill_work)
            # Keep Router reservations and P's transfer queue in the same
            # FIFO order.  Without this gate, later requests can consume D
            # prealloc KV while P correctly refuses to let them overtake the
            # head, producing a cross-order deadlock.
            ready_sequence = self._p_ready_sequence(rooms)
            ready_key = (
                (domain, ready_sequence)
                if getattr(self, "dynamic_prefill_domains", False)
                else ready_sequence
            )
            fifo_lock = await self._acquire_p_ready_fifo(ready_sequence, domain)
            try:
                reservation = await self._select_and_reserve_decode(
                    session, modified_request, rooms, prompt_tokens, domain
                )
                decode_task = asyncio.create_task(
                    session.post(
                        f"{reservation.url}/{endpoint}",
                        json=modified_request,
                        headers=headers,
                    )
                )
                # Preserve P-ready submission order, but never hold the global
                # FIFO lock while one D waits to allocate destination KV.  The
                # independent reservation task keeps capacity charged until D
                # removes the ready marker, so later submissions cannot
                # double-spend that space or globally head-of-line block.
                admission_task = asyncio.create_task(
                    self._release_reservation_when_admitted(
                        reservation, ready_key
                    )
                )
                await asyncio.sleep(0)
            finally:
                fifo_lock.release()
            prefill_response, decode_response = await asyncio.gather(
                prefill_task, decode_task
            )
            await admission_task
            return prefill_response, decode_response
        except BaseException:
            await self._release_prefill_work(prefill_work)
            if route_task is not None and not route_task.done():
                route_task.cancel()
                await asyncio.gather(route_task, return_exceptions=True)
            if ready_key is not None:
                self._p_ready_submitted_sequences.discard(ready_key)
            if prefill_task is not None and not prefill_task.done():
                prefill_task.cancel()
            if decode_task is not None and not decode_task.done():
                decode_task.cancel()
            if admission_task is not None and not admission_task.done():
                admission_task.cancel()
                await asyncio.gather(admission_task, return_exceptions=True)
            if reservation is not None:
                async with self._selection_lock:
                    self._reservations.pop(reservation.reservation_id, None)
            for room in rooms:
                try:
                    self._accepted_path(room).unlink(missing_ok=True)
                    self._scheduled_path(room).unlink(missing_ok=True)
                    self._ready_path(room).unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    async def generate(
        self, modified_request, prefill_server, decode_server, endpoint
    ) -> ORJSONResponse:
        assert decode_server is None, "Late-binding D must not be selected early"
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"
        generation_key = self._agentic_generation_key(modified_request)
        if generation_key is None:
            ret_json, status = await self._generate_once(
                modified_request, prefill_server, endpoint
            )
        else:
            ret_json, status = await self._generate_singleflight(
                generation_key, modified_request, prefill_server, endpoint
            )
        return ORJSONResponse(content=ret_json, status_code=status)

    async def generate_stream(
        self, modified_request, prefill_server, decode_server, endpoint="generate"
    ):
        assert decode_server is None, "Late-binding D must not be selected early"

        async def stream_results():
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                prefill_response, decode_response = await self._late_dispatch(
                    session, modified_request, prefill_server, endpoint, {}
                )
                if modified_request.get("return_logprob", False):
                    prefill_chunks = [chunk async for chunk in prefill_response.content]
                    first = orjson.loads(prefill_chunks[0].decode("utf-8")[5:].strip())
                    async for chunk in decode_response.content:
                        decoded = chunk.decode("utf-8")
                        if decoded.startswith("data:") and "[DONE]" not in decoded:
                            result = orjson.loads(decoded[5:].strip())
                            result["meta_info"]["input_token_logprobs"] = (
                                first["meta_info"]["input_token_logprobs"]
                                + result["meta_info"]["input_token_logprobs"]
                            )
                            yield b"data: " + orjson.dumps(result) + b"\n\n"
                        else:
                            yield chunk
                else:
                    async for chunk in decode_response.content.iter_chunked(
                        AIOHTTP_STREAM_READ_CHUNK_SIZE
                    ):
                        yield chunk

        return StreamingResponse(stream_results(), media_type="text/event-stream")
