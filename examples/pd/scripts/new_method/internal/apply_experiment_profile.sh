#!/usr/bin/env bash

apply_experiment_profile() {
  local profile_path="$1"
  local loader_path="$2"
  local python_bin="$3"
  local key value origin
  local -a profile_keys=()

  profile_path="$(realpath -- "${profile_path}")"
  while IFS=$'\t' read -r key value; do
    profile_keys+=("${key}")
    if [[ -z "${!key+x}" ]]; then
      printf -v "${key}" '%s' "${value}"
      export "${key}"
      printf -v "PROFILE_ORIGIN_${key}" '%s' profile
    else
      printf -v "PROFILE_ORIGIN_${key}" '%s' override
    fi
  done < <("${python_bin}" "${loader_path}" "${profile_path}")

  export EXPERIMENT_CONFIG="${profile_path}"
  echo "Experiment profile: ${profile_path}"
  for key in "${profile_keys[@]}"; do
    origin="PROFILE_ORIGIN_${key}"
    printf '  %-42s = %-24s [%s]\n' "${key}" "${!key}" "${!origin}"
  done
}
