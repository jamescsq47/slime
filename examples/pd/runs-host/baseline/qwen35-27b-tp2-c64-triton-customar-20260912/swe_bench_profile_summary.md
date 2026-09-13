# SWE-bench full-run profile

## Outcome

| Metric | Value |
|---|---:|
| requests | 500 |
| completed | 500 |
| failed | 0 |
| raw_failed | 0 |
| truncated | 0 |
| resolved | 308 |
| unresolved | 192 |
| verifier_infrastructure_errors | 0 |
| warm_image_requests | 500 |
| mini_swe_trajectories | 0 |
| openenv_trajectories | 500 |
| run_wall_seconds | 7762.3544 |
| request_per_second | 0.0644 |
| agent_per_second | 0.0644 |

## Timing and length distributions

| Metric | Mean | P50 | P90 | P95 | P99 | Max | Sum |
|---|---:|---:|---:|---:|---:|---:|---:|
| agent_latency_seconds | 560.575 | 521.787 | 891.071 | 1017.547 | 1384.131 | 4968.174 | 280287.655 |
| sample_time_seconds | 560.575 | 521.787 | 891.070 | 1017.546 | 1384.130 | 4968.173 | 280287.490 |
| model_seconds | 469.067 | 434.707 | 783.893 | 904.268 | 1234.180 | 1459.749 | 234533.395 |
| tool_seconds | 66.122 | 42.514 | 92.907 | 124.323 | 623.245 | 2204.592 | 33061.147 |
| verifier_queue_seconds | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.014 |
| verifier_seconds | 18.741 | 9.584 | 20.385 | 59.656 | 119.677 | 2400.226 | 9370.709 |
| docker_image_inspect_seconds | 0.097 | 0.076 | 0.161 | 0.189 | 0.231 | 0.268 | 48.403 |
| docker_container_start_seconds | 0.358 | 0.249 | 0.572 | 1.242 | 1.391 | 1.561 | 178.983 |
| docker_container_close_seconds | 0.334 | 0.306 | 0.451 | 0.495 | 0.658 | 0.732 | 166.820 |
| docker_exec_seconds | 88.170 | 55.568 | 125.597 | 185.508 | 679.543 | 4607.067 | 44085.232 |
| docker_upload_seconds | 0.108 | 0.093 | 0.155 | 0.178 | 0.235 | 0.391 | 53.767 |
| docker_accounted_seconds | 89.066 | 56.527 | 126.530 | 186.324 | 680.633 | 4607.666 | 44533.205 |
| docker_agent_tool_seconds | 66.121 | 42.514 | 92.906 | 124.322 | 623.244 | 2204.590 | 33060.732 |
| docker_baseline_seconds | 1.936 | 1.643 | 3.712 | 4.013 | 4.322 | 4.451 | 967.906 |
| docker_patch_capture_seconds | 1.224 | 1.395 | 1.566 | 1.633 | 1.771 | 2.354 | 612.243 |
| docker_verifier_exec_seconds | 18.609 | 9.472 | 20.270 | 59.420 | 119.566 | 2400.147 | 9304.692 |
| docker_setup_seconds | 0.279 | 0.248 | 0.379 | 0.437 | 0.677 | 0.820 | 139.659 |
| docker_exec_calls | 60.436 | 70.000 | 73.000 | 74.000 | 74.000 | 74.000 | 30218.000 |
| docker_upload_calls | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 500.000 |
| turns | 51.808 | 62.000 | 64.000 | 64.000 | 64.000 | 64.000 | 25904.000 |
| shell_calls | 51.286 | 61.000 | 64.000 | 64.000 | 64.000 | 64.000 | 25643.000 |
| cumulative_model_input_tokens | 917985.836 | 969078.500 | 1527873.400 | 1681450.550 | 1893903.940 | 2329954.000 | 458992918.000 |
| model_output_tokens | 10139.410 | 9173.000 | 17190.300 | 19446.700 | 26314.120 | 31838.000 | 5069705.000 |
| tool_observation_tokens | 18553.436 | 18361.500 | 28152.500 | 31246.950 | 35200.970 | 47845.000 | 9276718.000 |
| cached_input_tokens | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| trajectory_tokens | 10139.410 | 9173.000 | 17190.300 | 19446.700 | 26314.120 | 31838.000 | 5069705.000 |
| patch_chars | 11926.460 | 1514.500 | 6925.500 | 16535.750 | 350933.290 | 1163505.000 | 5963230.000 |
| patch_bytes | 11928.576 | 1514.500 | 6925.500 | 16537.650 | 350956.540 | 1163811.000 | 5964288.000 |

## Categories

- status: `{"completed": 500}`
- raw status: `{"completed": 500}`
- stop reason: `{"task_complete": 219, "max_turns": 236, "max_tokens_per_turn": 17, "no_command": 23, "command_timeout": 3, "tool_format_error": 1, "final_answer": 1}`
- verifier status: `{"completed": 499, "timeout": 1}`
- mini-SWE-agent version: `{"None": 500}`
