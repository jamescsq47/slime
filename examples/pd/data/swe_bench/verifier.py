"""SWE-bench verifier shared by local Docker and remote sandbox backends.

The model never receives the hidden test patch.  After the agent loop has
irreversibly stopped, the harness builds the canonical SWE-bench evaluation
script on the host, uploads it to the existing task sandbox, and parses the
test log with ``swebench.harness.grading``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


_GIT_OBJECT_RE = re.compile(r"[0-9a-fA-F]{40,64}")


class Sandbox(Protocol):
    async def execute(
        self,
        command: str,
        *,
        timeout: float | None = None,
        phase: str = "internal",
    ) -> tuple[int, str]: ...

    async def upload_bytes(self, contents: bytes, path: str) -> None: ...


class VerifierInfrastructureError(RuntimeError):
    """The benchmark environment failed independently of the model patch."""


@dataclass(frozen=True)
class RepositoryBaseline:
    official_base_commit: str
    official_tree: str
    image_commit: str
    image_tree: str
    kind: str
    image_commits_ahead: int | None
    fingerprint: str
    source_image_commit: str | None = None
    initial_worktree_status: str = ""


@dataclass(frozen=True)
class VerifierResult:
    status: str
    resolved: bool | None
    reward: int | None
    report: dict[str, Any] | None
    test_exit_code: int | None
    timed_out: bool
    duration_seconds: float
    output_tail: str
    error_type: str | None = None
    error: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


def _require_official_row(metadata: dict[str, Any]) -> None:
    required = (
        "instance_id",
        "repo",
        "version",
        "base_commit",
        "test_patch",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise VerifierInfrastructureError(
            f"SWE-bench verifier metadata is missing {missing}"
        )


async def prepare_repository_baseline(
    sandbox: Sandbox, expected_commit: str
) -> RepositoryBaseline:
    """Validate and preserve the clean repository shipped in the official image.

    Official images occasionally contain a clean compatibility commit on top of
    the public ``base_commit``.  As in the hardened Miles evaluator, this accepts
    an exact base, a clean descendant, or a complete-tree-equivalent alias, but
    never rewrites the image before the agent starts.
    """

    expected_commit = expected_commit.strip().lower()
    if not _GIT_OBJECT_RE.fullmatch(expected_commit):
        raise VerifierInfrastructureError(
            f"invalid official base_commit {expected_commit!r}"
        )
    status_exit, status = await sandbox.execute(
        "git status --short", timeout=60, phase="baseline"
    )
    head_exit, source_image_commit = await sandbox.execute(
        "git rev-parse --verify HEAD", timeout=60, phase="baseline"
    )
    source_image_commit = source_image_commit.strip().lower()
    if (
        status_exit != 0
        or head_exit != 0
        or not _GIT_OBJECT_RE.fullmatch(source_image_commit)
    ):
        raise VerifierInfrastructureError(
            "failed to inspect initial repository state: "
            f"status_exit={status_exit}, head_exit={head_exit}, "
            f"head={source_image_commit!r}"
        )

    source_tree_exit, source_image_tree = await sandbox.execute(
        "git rev-parse --verify HEAD^{tree}", timeout=60, phase="baseline"
    )
    official_tree_exit, official_tree = await sandbox.execute(
        f"git rev-parse --verify {expected_commit}^{{tree}}",
        timeout=60,
        phase="baseline",
    )
    source_image_tree = source_image_tree.strip().lower()
    official_tree = official_tree.strip().lower()
    if (
        source_tree_exit != 0
        or official_tree_exit != 0
        or not _GIT_OBJECT_RE.fullmatch(source_image_tree)
        or not _GIT_OBJECT_RE.fullmatch(official_tree)
    ):
        raise VerifierInfrastructureError(
            "failed to compare image and official repository trees: "
            f"image_tree_exit={source_tree_exit}, official_tree_exit={official_tree_exit}"
        )

    if source_image_commit == expected_commit:
        ancestor_exit, ancestor_output = 0, ""
    else:
        ancestor_exit, ancestor_output = await sandbox.execute(
            f"git merge-base --is-ancestor {expected_commit} HEAD",
            timeout=60,
            phase="baseline",
        )
    if ancestor_exit not in (0, 1):
        raise VerifierInfrastructureError(
            "failed to check official base ancestry: " + ancestor_output[-2000:]
        )
    is_descendant = ancestor_exit == 0
    tree_equivalent = source_image_tree == official_tree
    if not is_descendant and not tree_equivalent:
        raise VerifierInfrastructureError(
            "image HEAD is neither a descendant nor tree-equivalent to the official "
            f"base: image={source_image_commit}, official={expected_commit}"
        )

    initial_status = status.strip()
    image_commit = source_image_commit
    image_tree = source_image_tree
    if initial_status:
        # Some official/mirrored SWE-bench images intentionally ship with
        # compatibility edits in the worktree (commonly tox.ini/setup.py).
        # Snapshot that exact state into an unreachable synthetic commit so
        # subsequent patch capture records only agent changes.  A temporary
        # index keeps the original staging state and working files untouched.
        snapshot_exit, snapshot_output = await sandbox.execute(
            "set -e; index_file=$(mktemp); rm -f \"$index_file\"; "
            "trap 'rm -f \"$index_file\"' EXIT; "
            "GIT_INDEX_FILE=\"$index_file\" git read-tree HEAD; "
            "GIT_INDEX_FILE=\"$index_file\" git add -A; "
            "tree=$(GIT_INDEX_FILE=\"$index_file\" git write-tree); "
            "commit=$(printf '%s\\n' 'pd-swe initial image worktree' | "
            "git -c user.name=pd-swe -c user.email=pd-swe@localhost "
            "commit-tree \"$tree\" -p HEAD); "
            "printf '%s %s\\n' \"$commit\" \"$tree\"",
            timeout=300,
            phase="baseline",
        )
        parts = snapshot_output.strip().lower().split()
        if (
            snapshot_exit != 0
            or len(parts) != 2
            or not all(_GIT_OBJECT_RE.fullmatch(value) for value in parts)
        ):
            raise VerifierInfrastructureError(
                "failed to snapshot initial dirty image worktree: "
                + snapshot_output[-2000:]
            )
        image_commit, image_tree = parts

    commits_ahead: int | None = None
    if is_descendant:
        count_exit, count_output = await sandbox.execute(
            f"git rev-list --count {expected_commit}..HEAD",
            timeout=60,
            phase="baseline",
        )
        if count_exit != 0 or not count_output.strip().isdigit():
            raise VerifierInfrastructureError(
                f"failed to audit image commits ahead of base: {count_output[-2000:]}"
            )
        commits_ahead = int(count_output.strip())

    if initial_status and source_image_commit == expected_commit:
        kind = "dirty_snapshot_exact"
    elif initial_status and is_descendant:
        kind = "dirty_snapshot_compatibility_descendant"
    elif initial_status:
        kind = "dirty_snapshot_equivalent_tree_alias"
    elif image_commit == expected_commit:
        kind = "exact"
    elif is_descendant and tree_equivalent:
        kind = "equivalent_tree_descendant"
    elif is_descendant:
        kind = "compatibility_descendant"
    else:
        kind = "equivalent_tree_alias"
    identity = {
        "official_base_commit": expected_commit,
        "official_tree": official_tree,
        "image_commit": image_commit,
        "image_tree": image_tree,
        "kind": kind,
    }
    fingerprint_identity = {
        **identity,
        "source_image_commit": source_image_commit,
        "initial_worktree_status_sha256": hashlib.sha256(
            initial_status.encode()
        ).hexdigest(),
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_identity, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    return RepositoryBaseline(
        **identity,
        image_commits_ahead=commits_ahead,
        fingerprint=fingerprint,
        source_image_commit=source_image_commit,
        initial_worktree_status=initial_status,
    )


async def capture_repository_patch(
    sandbox: Sandbox, baseline_commit: str
) -> str:
    """Capture committed, staged, unstaged, binary, and untracked changes."""

    if not _GIT_OBJECT_RE.fullmatch(baseline_commit):
        raise VerifierInfrastructureError(
            f"invalid preserved image commit {baseline_commit!r}"
        )
    index_prefix = (
        "set -e; index_file=$(mktemp); rm -f \"$index_file\"; "
        "trap 'rm -f \"$index_file\"' EXIT; "
        f"GIT_INDEX_FILE=\"$index_file\" git read-tree {baseline_commit}; "
    )
    tracked_exit, tracked = await sandbox.execute(
        index_prefix
        + f"GIT_INDEX_FILE=\"$index_file\" git diff --binary --no-ext-diff "
        f"{baseline_commit} --",
        timeout=120,
        phase="patch_capture",
    )
    if tracked_exit != 0:
        raise VerifierInfrastructureError(
            f"tracked patch capture failed ({tracked_exit}): {tracked[-2000:]}"
        )
    untracked_exit, untracked = await sandbox.execute(
        index_prefix
        + "set -o pipefail; GIT_INDEX_FILE=\"$index_file\" "
        "git ls-files --others --exclude-standard -z | "
        "xargs -0 -r -n 1 sh -c 'git diff --binary --no-ext-diff "
        "--no-index -- /dev/null \"$1\"; code=$?; "
        "[ \"$code\" -eq 0 ] || [ \"$code\" -eq 1 ]' sh",
        timeout=120,
        phase="patch_capture",
    )
    if untracked_exit != 0:
        raise VerifierInfrastructureError(
            f"untracked patch capture failed ({untracked_exit}): {untracked[-2000:]}"
        )
    parts = [part.rstrip() for part in (tracked, untracked) if part.strip()]
    return ("\n".join(parts) + "\n") if parts else ""


def _swebench_api() -> tuple[Any, Any]:
    """Load the Miles/Harbor-compatible SWE-bench 4.x verifier API lazily."""

    try:
        from swebench.harness.grading import get_eval_report
        from swebench.harness.test_spec.test_spec import make_test_spec
    except ImportError as exc:
        raise VerifierInfrastructureError(
            "inline SWE-bench verification requires swebench==4.0.3; install "
            "examples/pd/requirements-swe-bench.txt in the selected PD environment"
        ) from exc
    return make_test_spec, get_eval_report


def build_eval_script(metadata: dict[str, Any]) -> tuple[Any, str]:
    _require_official_row(metadata)
    make_test_spec, _ = _swebench_api()
    try:
        test_spec = make_test_spec(dict(metadata))
    except Exception as exc:
        raise VerifierInfrastructureError(
            f"failed to build official test spec for {metadata.get('instance_id')}: {exc}"
        ) from exc
    return test_spec, test_spec.eval_script


def grade_test_output(
    metadata: dict[str, Any], test_spec: Any, model_patch: str, output: str
) -> dict[str, Any]:
    _, get_eval_report = _swebench_api()
    prediction = {
        "instance_id": str(metadata["instance_id"]),
        "model_name_or_path": str(metadata.get("model_name_or_path", "pd-serving")),
        # Preserve official n=1 semantics: an empty patch is still a submitted
        # attempt, while None means no prediction was produced.
        "model_patch": model_patch,
    }
    path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix="pd-swe-verifier-", suffix=".log", delete=False
        ) as handle:
            handle.write(output)
            path = handle.name
        return get_eval_report(test_spec, prediction, path, include_tests_status=True)
    except Exception as exc:
        raise VerifierInfrastructureError(
            f"failed to parse SWE-bench verifier output: {exc}"
        ) from exc
    finally:
        if path:
            Path(path).unlink(missing_ok=True)


async def run_inline_verifier(
    sandbox: Sandbox,
    metadata: dict[str, Any],
    model_patch: str,
    *,
    timeout_seconds: float,
    output_tail_chars: int = 12000,
) -> VerifierResult:
    """Run hidden tests in the existing sandbox and return canonical resolution."""

    started = time.monotonic()
    try:
        test_spec, script = build_eval_script(metadata)
        await sandbox.upload_bytes(script.encode("utf-8"), "/eval.sh")
        exit_code, output = await sandbox.execute(
            "chmod +x /eval.sh && /bin/bash /eval.sh",
            timeout=timeout_seconds,
            phase="verifier",
        )
        duration = time.monotonic() - started
        if exit_code in (124, 137):
            return VerifierResult(
                status="timeout",
                resolved=False,
                reward=0,
                report=None,
                test_exit_code=exit_code,
                timed_out=True,
                duration_seconds=duration,
                output_tail=output[-output_tail_chars:],
                error_type="VerifierTimeout",
                error=f"verifier exceeded {timeout_seconds:g} seconds",
            )
        report = grade_test_output(metadata, test_spec, model_patch, output)
        instance_id = str(metadata["instance_id"])
        instance_report = report.get(instance_id)
        if not isinstance(instance_report, dict) or type(
            instance_report.get("resolved")
        ) is not bool:
            raise VerifierInfrastructureError(
                f"verifier report has no boolean resolved entry for {instance_id!r}"
            )
        resolved = instance_report["resolved"]
        return VerifierResult(
            status="completed",
            resolved=resolved,
            reward=int(resolved),
            report=report,
            test_exit_code=exit_code,
            timed_out=False,
            duration_seconds=duration,
            output_tail=output[-output_tail_chars:],
        )
    except VerifierInfrastructureError as exc:
        return VerifierResult(
            status="infrastructure_error",
            resolved=None,
            reward=None,
            report=None,
            test_exit_code=None,
            timed_out=False,
            duration_seconds=time.monotonic() - started,
            output_tail="",
            error_type=type(exc).__name__,
            error=str(exc),
        )
    except TimeoutError as exc:
        return VerifierResult(
            status="timeout",
            resolved=False,
            reward=0,
            report=None,
            test_exit_code=124,
            timed_out=True,
            duration_seconds=time.monotonic() - started,
            output_tail="",
            error_type="VerifierTimeout",
            error=str(exc) or f"verifier exceeded {timeout_seconds:g} seconds",
        )
    except Exception as exc:
        return VerifierResult(
            status="infrastructure_error",
            resolved=None,
            reward=None,
            report=None,
            test_exit_code=None,
            timed_out=False,
            duration_seconds=time.monotonic() - started,
            output_tail="",
            error_type=type(exc).__name__,
            error=str(exc),
        )


def patch_metadata(model_patch: str) -> dict[str, Any]:
    return {
        "model_patch_chars": len(model_patch),
        "model_patch_sha256": hashlib.sha256(model_patch.encode()).hexdigest(),
    }


def daytona_api_key(explicit: str = "") -> str:
    value = explicit or os.environ.get("DAYTONA_API_KEY", "")
    if not value:
        raise VerifierInfrastructureError(
            "DAYTONA_API_KEY is required when sandbox_backend=daytona"
        )
    return value
