"""CPU lifecycle regression for TP1 recovery from another P's Host arena."""
from types import SimpleNamespace
from pathlib import Path
import tempfile

import pytest

from sglang.srt.disaggregation.agentic_host_staging import (
    AgenticPHostStagingManager, HostStageState, SharedHostStagingLedger,
)


@pytest.fixture
def ledger():
    directory = tempfile.TemporaryDirectory(prefix="mamba-foreign-test-", dir="/dev/shm")
    result = SharedHostStagingLedger(str(Path(directory.name) / "staging.json"))

    def seed(entries):
        entries["req:0"] = dict(
            snapshot_id="req:0", state="host_ready", p_owner="arena-p1",
            tp_size=1, recovery_prefill_domain=0,
            recovery_reservation_id="reservation", loader_acks=[], binder_acks=[],
        )
        return True, True

    result._mutate(seed, event_snapshot_id="req:0")
    yield result
    directory.cleanup()


def claim(ledger):
    assert ledger.claim_d2p_recovery_rank(
        "req:0", "recovery-p0", tp_rank=0, tp_size=1,
        claim_id="claim", prefill_domain=0, reservation_id="reservation",
    )
    assert ledger.attach_d2p_recovery_lease_rank(
        "req:0", "recovery-p0", tp_rank=0, tp_size=1,
        claim_id="claim", lease_id=7,
    )


def prepare(ledger, owner="recovery-p0", tp_size=1):
    manager = SimpleNamespace(ledger=ledger, owner=owner, tp_rank=0, tp_size=tp_size)
    return AgenticPHostStagingManager._prepare_h2d_load_ledger(
        manager, {"request_generation": SimpleNamespace(snapshot_id="req:0")}
    )


def test_foreign_tp1_prepare_and_complete_handoff(ledger):
    claim(ledger)
    # Generic transitions retain their original arena-owner authorization.
    assert not ledger.transition("req:0", HostStageState.H2D_LOADING, owner="recovery-p0")
    assert prepare(ledger)
    assert prepare(ledger)  # Idempotent, no new lease or claim.
    entry = ledger.get("req:0")
    assert entry["h2d_prepared_ranks"] == [0]
    assert entry["recovery_claims"]["0"]["lease_id"] == 7
    assert not ledger.begin_host_eviction("req:0", "arena-p1", tp_size=1, reason="test")
    assert ledger.mark_d2p_recovery_phase_rank(
        "req:0", "recovery-p0", tp_rank=0, tp_size=1,
        claim_id="claim", lease_id=7, phase="io_inflight",
    )
    assert ledger.complete_d2p_host_load_rank("req:0", "recovery-p0", tp_rank=0, tp_size=1)
    assert ledger.get("req:0")["state"] == "hbm_ready"
    assert ledger.complete_host_bind_rank("req:0", "recovery-p0", tp_rank=0, tp_size=1)
    handed = []
    assert ledger.commit_d2p_handoff_rank(
        "req:0", "recovery-p0", tp_rank=0, tp_size=1,
        claim_id="claim", lease_id=7, handoff=lambda: handed.append(7),
    )
    assert handed == [7]
    entry = ledger.get("req:0")
    assert entry["state"] == "consumed"
    assert entry["recovery_claims"]["0"]["phase"] == "handed"
    assert entry["p_owner"] == "arena-p1"


@pytest.mark.parametrize("owner,tp_size", [("arena-p1", 1), ("wrong-p", 1), ("recovery-p0", 2)])
def test_prepare_rejects_wrong_recovery_owner_or_tp_size(ledger, owner, tp_size):
    claim(ledger)
    before = ledger.get("req:0")
    assert not prepare(ledger, owner, tp_size)
    assert ledger.get("req:0") == before


def test_foreign_prepare_cancellation_preserves_host_and_can_retry(ledger):
    claim(ledger)
    assert prepare(ledger)
    assert ledger.cancel_d2p_recovery_rank(
        "req:0", "recovery-p0", tp_rank=0, tp_size=1,
        claim_id="claim", lease_id=7,
    )
    assert ledger.get("req:0")["state"] == "host_ready"
    claim(ledger)
    assert prepare(ledger)


def test_incompatible_parent_uses_current_assignment_not_stale_offer(ledger):
    manager = SimpleNamespace(ledger=ledger, owner="recovery-p0", arena_domain=0)
    assert AgenticPHostStagingManager._fail_incompatible_host_recovery(manager, "req:0")
    entry = ledger.get("req:0")
    assert entry["state"] == "failed"
    assert entry["reason"] == "permanent_parent_digest_mismatch"
    assert entry["failed_recovery_reservation_id"] == "reservation"


def test_incompatible_parent_cannot_fail_another_assignment(ledger):
    manager = SimpleNamespace(ledger=ledger, owner="wrong-p", arena_domain=3)
    before = ledger.get("req:0")
    assert not AgenticPHostStagingManager._fail_incompatible_host_recovery(manager, "req:0")
    assert ledger.get("req:0") == before
