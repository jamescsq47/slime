# Bandwidth microbenchmarks

These tests isolate data paths and never launch the mixed Agent workload.

- `run_gpu_memory_paths.sh`: GPU-to-GPU NVLink/PCIe paths.
- `run_shared_host_arena.sh`: D GPU → shared P Host arena → P GPU.
- `run_mooncake_store.sh`: Mooncake Put/Get paths.
- `benchmark_agentic_host_h2d.py`: Host-to-P HBM agentic recovery.
- `benchmark_d_hbm_via_p_staging_to_host.py`: staged D-HBM-to-Host path.
- `benchmark_nixl_d_hbm_to_p_host.py`: NIXL D-HBM-to-P-Host path.
- `benchmark_p_host_bidirectional.py`: simultaneous P Host ingress/egress.
- `stream_memory_bandwidth.c`: CPU DRAM STREAM helper.

The three `run_*.sh` wrappers select the modified `pd` environment.  Direct
Python benchmarks expose their own `--help` and should be run from the repo
root.  Store generated results outside this source directory.
