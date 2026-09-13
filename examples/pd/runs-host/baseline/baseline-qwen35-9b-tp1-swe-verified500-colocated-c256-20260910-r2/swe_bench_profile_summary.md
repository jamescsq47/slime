# SWE-bench full-run profile

## Outcome

| Metric | Value |
|---|---:|
| requests | 500 |
| completed | 496 |
| failed | 4 |
| raw_failed | 4 |
| truncated | 0 |
| resolved | 158 |
| unresolved | 338 |
| verifier_infrastructure_errors | 0 |
| warm_image_requests | 500 |
| mini_swe_trajectories | 0 |
| openenv_trajectories | 500 |
| run_wall_seconds | 3262.2812 |
| request_per_second | 0.1533 |
| agent_per_second | 0.1520 |

## Timing and length distributions

| Metric | Mean | P50 | P90 | P95 | P99 | Max | Sum |
|---|---:|---:|---:|---:|---:|---:|---:|
| agent_latency_seconds | 849.904 | 786.587 | 1472.320 | 1709.745 | 2022.286 | 3261.896 | 424952.229 |
| sample_time_seconds | 849.904 | 786.587 | 1472.320 | 1709.745 | 2022.285 | 3261.895 | 424951.894 |
| model_seconds | 770.688 | 706.293 | 1370.258 | 1630.274 | 1949.315 | 3206.558 | 385344.197 |
| tool_seconds | 46.459 | 37.703 | 63.752 | 82.930 | 289.129 | 949.478 | 23229.351 |
| verifier_queue_seconds | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.006 |
| verifier_seconds | 15.877 | 10.268 | 27.435 | 52.737 | 106.298 | 326.537 | 7874.880 |
| docker_image_inspect_seconds | 0.396 | 0.152 | 1.296 | 1.475 | 1.604 | 1.788 | 197.936 |
| docker_container_start_seconds | 2.165 | 1.109 | 5.564 | 5.880 | 6.064 | 11.755 | 1082.413 |
| docker_container_close_seconds | 0.670 | 0.379 | 1.036 | 1.619 | 7.736 | 12.196 | 334.792 |
| docker_exec_seconds | 69.552 | 55.806 | 98.464 | 139.905 | 353.702 | 1069.863 | 34776.105 |
| docker_upload_seconds | 0.140 | 0.111 | 0.234 | 0.296 | 0.470 | 1.234 | 70.097 |
| docker_accounted_seconds | 72.923 | 59.900 | 103.505 | 140.879 | 354.802 | 1071.093 | 36461.343 |
| docker_agent_tool_seconds | 46.458 | 37.702 | 63.751 | 82.930 | 289.129 | 949.478 | 23229.088 |
| docker_baseline_seconds | 5.497 | 7.033 | 11.139 | 11.466 | 11.781 | 12.004 | 2748.637 |
| docker_patch_capture_seconds | 1.257 | 1.432 | 1.770 | 2.004 | 2.313 | 4.106 | 628.477 |
| docker_verifier_exec_seconds | 15.593 | 10.106 | 27.066 | 50.900 | 106.007 | 326.286 | 7796.322 |
| docker_setup_seconds | 0.747 | 0.593 | 1.337 | 1.487 | 2.413 | 2.468 | 373.582 |
| docker_exec_calls | 52.872 | 57.000 | 73.000 | 73.000 | 74.000 | 74.000 | 26436.000 |
| docker_upload_calls | 0.992 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 496.000 |
| turns | 44.188 | 49.000 | 64.000 | 64.000 | 64.000 | 64.000 | 22094.000 |
| shell_calls | 43.746 | 48.000 | 64.000 | 64.000 | 64.000 | 64.000 | 21873.000 |
| cumulative_model_input_tokens | 554462.938 | 495758.000 | 1095377.400 | 1266805.400 | 1564107.790 | 1925212.000 | 277231469.000 |
| model_output_tokens | 13158.660 | 11516.500 | 24478.600 | 29717.050 | 39866.640 | 71899.000 | 6579330.000 |
| tool_observation_tokens | 14761.448 | 14098.000 | 25515.100 | 29034.250 | 34514.070 | 62465.000 | 7380724.000 |
| cached_input_tokens | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| trajectory_tokens | 13158.660 | 11516.500 | 24478.600 | 29717.050 | 39866.640 | 71899.000 | 6579330.000 |
| patch_chars | 2707.837 | 553.000 | 2790.000 | 4912.250 | 28228.150 | 572767.000 | 1343087.000 |
| patch_bytes | 2686.318 | 550.000 | 2762.800 | 4881.650 | 28154.820 | 572780.000 | 1343159.000 |

## Categories

- status: `{"completed": 496, "failed": 4}`
- raw status: `{"completed": 496, "failed": 4}`
- stop reason: `{"max_tokens_per_turn": 85, "task_complete": 103, "max_turns": 171, "repeated_command_outcome": 106, "no_command": 29, "environment_error:TimeoutError": 3, "command_timeout": 2, "environment_error:RuntimeError": 1}`
- verifier status: `{"completed": 496, "None": 4}`
- mini-SWE-agent version: `{"None": 500}`
