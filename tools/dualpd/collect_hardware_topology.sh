#!/usr/bin/env bash
# Passive Linux inventory for remote Agentic-PD development. No sudo or installs.
set -euo pipefail
umask 077

usage() {
  cat <<'EOF'
Usage: bash collect_hardware_topology.sh [--output NEW_DIRECTORY]
       [--peer PEER_IP ...] [--timeout SECONDS] [--python PYTHON_EXECUTABLE]

Run once on EACH prospective P/D node, preferably inside the actual serving
container/environment. --peer only queries the local route; it sends no probes.
Default output: a fresh /tmp/dualpd-hardware-... directory plus .tar.gz.
Commands are read-only; only report files are written. No sudo, installation,
GPU computation, RDMA test traffic, network changes, or stopping existing services.
Reports include hostnames, IPs, PCI addresses and GPU UUIDs. They do not collect
the full environment, credentials, SSH keys, or arbitrary process command lines.
Missing tools/permissions and timeouts are reported, not silently treated as OK.
GNU timeout is required. Python, NVIDIA, RDMA and topology tools are optional.
EOF
}

output_dir=''
command_timeout=20
python_bin=python3
peers=()
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --output|--peer|--timeout|--python)
      (($# >= 2)) || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
      case "$1" in
        --output) output_dir=$2 ;;
        --peer)
          [[ "$2" =~ ^[0-9A-Fa-f:.]+$ && "$2" == *[.:]* ]] || {
            printf 'Use a numeric IPv4/IPv6 address for --peer\n' >&2; exit 2;
          }
          peers+=("$2") ;;
        --timeout) command_timeout=$2 ;;
        --python) python_bin=$2 ;;
      esac
      shift 2 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$command_timeout" =~ ^[1-9][0-9]*$ ]] || {
  printf 'Timeout must be a positive integer\n' >&2; exit 2;
}
[[ $(uname -s) == Linux ]] || { printf 'Linux is required\n' >&2; exit 2; }
command -v timeout >/dev/null || { printf 'GNU timeout is required\n' >&2; exit 2; }
command -v tar >/dev/null || { printf 'tar is required\n' >&2; exit 2; }

if [[ -z "$output_dir" ]]; then
  output_dir=$(mktemp -d "${TMPDIR:-/tmp}/dualpd-hardware-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
else
  [[ ! -e "$output_dir" && ! -L "$output_dir" && ! -e "${output_dir%/}.tar.gz" ]] || {
    printf 'Refusing to overwrite an existing report: %s\n' "$output_dir" >&2; exit 2;
  }
  mkdir -- "$output_dir"
fi
output_dir=$(cd -- "$output_dir" && pwd -P)
status_file="$output_dir/status.tsv"
printf 'section\texit_code\tstatus\n' > "$status_file"

collect() {
  local section=$1 rc result
  shift
  printf '[collect] %s\n' "$section"
  printf '%q ' "$@" > "$output_dir/${section}.command.txt"
  printf '\n' >> "$output_dir/${section}.command.txt"
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Optional command missing: %s\n' "$1" > "$output_dir/${section}.txt"
    rc=127
    result=MISSING
  elif timeout --signal=TERM --kill-after=3s "${command_timeout}s" "$@" > "$output_dir/${section}.txt" 2>&1; then
    rc=0
    result=OK
  else
    rc=$?
    result=ERROR_OR_UNSUPPORTED
    [[ $rc != 124 && $rc != 137 ]] || result=TIMEOUT
  fi
  printf '%s\t%s\t%s\n' "$section" "$rc" "$result" >> "$status_file"
}

# Run sysfs enumeration in a bounded subprocess too. No recursive filesystem scan.
read_attr() {
  local path=$1 value
  if [[ -r "$path" ]]; then
    if value=$(cat -- "$path" 2>/dev/null); then printf '%s' "$value"; else printf '?'; fi
  else
    printf '?'
  fi
}

sysfs_topology() {
  local dev vendor class port gid value gid_type netdev node
  printf 'PCI devices (display/GPU and network):\n'
  printf 'BDF\tvendor\tdevice\tclass\tNUMA\tlocal_cpus\tlink_speed\tlink_width\tmax_speed\tmax_width\n'
  for dev in /sys/bus/pci/devices/*; do
    [[ -d "$dev" ]] || continue
    vendor=$(read_attr "$dev/vendor")
    class=$(read_attr "$dev/class")
    [[ "$class" == 0x02* || "$class" == 0x03* || "$vendor" == 0x10de ]] || continue
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${dev##*/}" "$vendor" "$(read_attr "$dev/device")" "$class" \
      "$(read_attr "$dev/numa_node")" "$(read_attr "$dev/local_cpulist")" \
      "$(read_attr "$dev/current_link_speed")" "$(read_attr "$dev/current_link_width")" \
      "$(read_attr "$dev/max_link_speed")" "$(read_attr "$dev/max_link_width")"
  done
  printf '\nNetwork interface -> PCI -> NUMA:\n'
  for dev in /sys/class/net/*; do
    [[ -e "$dev" ]] || continue
    printf '%s device=%s NUMA=%s state=%s mtu=%s\n' "${dev##*/}" \
      "$(readlink -f "$dev/device" 2>/dev/null || true)" \
      "$(read_attr "$dev/device/numa_node")" "$(read_attr "$dev/operstate")" "$(read_attr "$dev/mtu")"
  done
  printf '\nRDMA device -> PCI -> NUMA, port state, GID/netdev/type:\n'
  for dev in /sys/class/infiniband/*; do
    [[ -e "$dev" ]] || continue
    printf '%s device=%s NUMA=%s firmware=%s\n' "${dev##*/}" \
      "$(readlink -f "$dev/device" 2>/dev/null || true)" \
      "$(read_attr "$dev/device/numa_node")" "$(read_attr "$dev/fw_ver")"
    for port in "$dev"/ports/*; do
      [[ -d "$port" ]] || continue
      printf '  port=%s state=%s physical=%s layer=%s rate=%s\n' "${port##*/}" \
        "$(read_attr "$port/state")" "$(read_attr "$port/phys_state")" \
        "$(read_attr "$port/link_layer")" "$(read_attr "$port/rate")"
      for gid in "$port"/gids/*; do
        [[ -f "$gid" ]] || continue
        value=$(read_attr "$gid")
        [[ "$value" != 0000:0000:0000:0000:0000:0000:0000:0000 ]] || continue
        gid_type=$(read_attr "$port/gid_attrs/types/${gid##*/}")
        netdev=$(read_attr "$port/gid_attrs/ndevs/${gid##*/}")
        printf '    gid_index=%s gid=%s type=%s netdev=%s\n' "${gid##*/}" "$value" "$gid_type" "$netdev"
      done
    done
  done
  printf '\nNUMA nodes:\n'
  for node in /sys/devices/system/node/node[0-9]*; do
    [[ -d "$node" ]] || continue
    printf '%s cpus=%s distances=%s\n%s\n' "${node##*/}" \
      "$(read_attr "$node/cpulist")" "$(read_attr "$node/distance")" "$(read_attr "$node/meminfo")"
  done
}

collect identity bash -c 'date -u --iso-8601=seconds; hostname; uname -a; id; cat /etc/os-release'
collect cpu lscpu
collect numa numactl --hardware
collect memory free -h
collect meminfo cat /proc/meminfo
collect process_limits cat /proc/self/limits
collect cpuset bash -c 'grep -E "^(Cpus_allowed_list|Mems_allowed_list):" /proc/self/status; cat /proc/self/cgroup'
collect cgroup_limits bash -c 'for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current /sys/fs/cgroup/cpu.max /sys/fs/cgroup/cpuset.cpus.effective /sys/fs/cgroup/memory/memory.limit_in_bytes; do if test -r "$f"; then printf "%s: " "$f"; cat "$f"; fi; done'
collect container systemd-detect-virt --container
collect shm df -hT /dev/shm
collect shm_mount findmnt -T /dev/shm
collect pci_tree lspci -Dtv
collect pci_devices lspci -Dnnk
collect sysfs_topology bash -c "$(declare -f read_attr sysfs_topology); sysfs_topology"

collect gpu_list nvidia-smi -L
collect gpu_summary nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total,memory.used,utilization.gpu --format=csv
collect gpu_details nvidia-smi -q
collect gpu_topology nvidia-smi topo -m
collect gpu_p2p_read nvidia-smi topo -p2p r
collect gpu_p2p_write nvidia-smi topo -p2p w
collect gpu_nvlink nvidia-smi nvlink -s
collect cuda_compiler nvcc --version
collect driver_modules lsmod
collect device_access bash -c 'ls -l /dev/infiniband /dev/nvidia* 2>&1'

collect network_addresses ip -brief address show
collect network_links ip -details link show
collect network_routes ip route show table all
collect network_routes_ipv6 ip -6 route show table all
for dev in /sys/class/net/*; do
  [[ -e "$dev/device" ]] || continue
  iface=${dev##*/}
  label=${iface//[^a-zA-Z0-9_.-]/_}
  collect "nic_${label}" ethtool "$iface"
  collect "nic_${label}_driver" ethtool -i "$iface"
done
for index in "${!peers[@]}"; do
  collect "peer_${index}_route" ip route get "${peers[$index]}"
done
collect rdma_devices ibv_devices
collect rdma_details ibv_devinfo -v
collect rdma_ports ibstat
collect rdma_netdev ibdev2netdev
collect rdma_links rdma link show
collect rdma_system rdma system show
collect ucx_version ucx_info -v
collect ofed_version ofed_info -s

# Metadata only: do NOT import torch or initialize a CUDA context.
collect python_packages "$python_bin" -c '
import importlib.metadata as m
import importlib.util
import platform
import sys
print("executable:", sys.executable)
print("python:", platform.python_version())
for name in ("sglang", "slime", "torch", "transformers", "nixl", "nixl-cu12", "sglang-kernel", "flashinfer-python", "cuda-python", "sglang-router"):
    try:
        print(name + ":", m.version(name))
    except m.PackageNotFoundError:
        print(name + ": NOT INSTALLED")
for name in ("sglang", "slime"):
    spec = importlib.util.find_spec(name)
    print(name + " source:", spec.origin if spec else "NOT FOUND")
'

{
  printf '# DualPD hardware inventory\n\n'
  printf 'Collected (UTC): %s\n\n' "$(date -u --iso-8601=seconds)"
  printf '## GPU inventory\n\n```text\n'; cat "$output_dir/gpu_summary.txt"; printf '\n```\n'
  printf '\n## GPU/NUMA topology\n\n```text\n'; cat "$output_dir/gpu_topology.txt"; printf '\n```\n'
  printf '\n## RDMA to network mapping\n\n```text\n'; cat "$output_dir/rdma_netdev.txt"; printf '\n```\n'
  printf '\n## Probe status\n\n```text\n'; cat "$status_file"; printf '```\n'
  printf '\nSee sysfs_topology.txt for PCI/NUMA mappings, PCIe link rates and RDMA GIDs.\n'
  printf 'NUMA=-1 or ? means unknown, not node 0. Missing tools do not prove missing hardware.\n'
  printf 'Link rates and P2P capability are not measured bandwidth. This report does not prove\n'
  printf 'peer connectivity, GPUDirect RDMA correctness, or multi-node KV transfer correctness.\n'
  printf 'NVIDIA CUDA Version (if shown) is driver capability, not the installed CUDA runtime.\n'
  printf 'Compare node reports before selecting NICs/GPUs; use separate authorized tests for bandwidth.\n'
} > "$output_dir/SUMMARY.md"

archive_path="${output_dir}.tar.gz"
[[ ! -e "$archive_path" ]] || { printf 'Archive already exists: %s\n' "$archive_path" >&2; exit 2; }
tar -czf "$archive_path" -C "$(dirname -- "$output_dir")" "$(basename -- "$output_dir")"
printf '\nReport: %s\nArchive: %s\nSend one archive per node. Review IP/UUID information before sharing.\n' \
  "$output_dir/SUMMARY.md" "$archive_path"
