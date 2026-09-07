#!/usr/bin/env bash
set -euo pipefail

# Keep the serving environment immutable: import the exact upstream source
# revision through PYTHONPATH instead of installing/upgrading its dependencies.
MINISWE_REPOSITORY="${MINISWE_REPOSITORY:-https://github.com/SWE-agent/mini-swe-agent.git}"
MINISWE_COMMIT="${MINISWE_COMMIT:-25941c89cfbc91eb40b3f8756348c91d9977d57e}"
MINISWE_ROOT="${MINISWE_ROOT:-/tmp/pd-third-party/mini-swe-agent-${MINISWE_COMMIT}}"

if [[ ! -d "${MINISWE_ROOT}/.git" ]]; then
  mkdir -p "$(dirname -- "${MINISWE_ROOT}")"
  git clone --filter=blob:none --no-checkout "${MINISWE_REPOSITORY}" "${MINISWE_ROOT}"
fi

git -C "${MINISWE_ROOT}" fetch --quiet origin "${MINISWE_COMMIT}"
git -C "${MINISWE_ROOT}" checkout --quiet --detach "${MINISWE_COMMIT}"
actual_commit="$(git -C "${MINISWE_ROOT}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${MINISWE_COMMIT}" ]]; then
  echo "mini-SWE-agent revision mismatch: ${actual_commit}" >&2
  exit 1
fi

printf '%s\n' "${MINISWE_ROOT}"
