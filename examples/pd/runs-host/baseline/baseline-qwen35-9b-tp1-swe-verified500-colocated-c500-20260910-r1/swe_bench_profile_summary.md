# SWE-bench full-run profile

## Outcome

| Metric | Value |
|---|---:|
| requests | 500 |
| completed | 496 |
| failed | 4 |
| raw_failed | 4 |
| truncated | 0 |
| resolved | 157 |
| unresolved | 339 |
| verifier_infrastructure_errors | 0 |
| warm_image_requests | 500 |
| mini_swe_trajectories | 0 |
| openenv_trajectories | 500 |
| run_wall_seconds | 3902.4215 |
| request_per_second | 0.1281 |
| agent_per_second | 0.1271 |

## Timing and length distributions

| Metric | Mean | P50 | P90 | P95 | P99 | Max | Sum |
|---|---:|---:|---:|---:|---:|---:|---:|
| agent_latency_seconds | 1805.416 | 1918.701 | 2541.404 | 2660.147 | 2926.438 | 3900.740 | 902708.064 |
| sample_time_seconds | 1805.415 | 1918.700 | 2541.403 | 2660.147 | 2926.437 | 3900.740 | 902707.716 |
| model_seconds | 1586.966 | 1708.763 | 2334.186 | 2428.165 | 2694.349 | 2864.521 | 793483.238 |
| tool_seconds | 151.429 | 147.271 | 194.707 | 216.488 | 711.952 | 823.194 | 75714.579 |
| verifier_queue_seconds | 1.726 | 0.000 | 0.000 | 16.576 | 41.275 | 58.916 | 855.956 |
| verifier_seconds | 24.553 | 11.744 | 38.516 | 77.716 | 128.630 | 2401.035 | 12178.482 |
| docker_image_inspect_seconds | 1.981 | 2.390 | 3.344 | 3.490 | 3.726 | 3.790 | 990.505 |
| docker_container_start_seconds | 13.034 | 5.572 | 26.025 | 26.463 | 27.063 | 27.762 | 6516.810 |
| docker_container_close_seconds | 1.781 | 0.682 | 3.964 | 6.511 | 14.019 | 50.849 | 890.390 |
| docker_exec_seconds | 192.843 | 179.676 | 242.369 | 290.818 | 754.579 | 3080.898 | 96421.558 |
| docker_upload_seconds | 0.808 | 0.148 | 2.745 | 4.500 | 9.285 | 11.767 | 404.217 |
| docker_accounted_seconds | 210.447 | 194.103 | 260.141 | 318.742 | 762.833 | 3089.180 | 105223.480 |
| docker_agent_tool_seconds | 151.429 | 147.270 | 194.706 | 216.487 | 711.951 | 823.193 | 75714.278 |
| docker_baseline_seconds | 11.825 | 12.057 | 16.021 | 16.675 | 16.860 | 17.931 | 5912.251 |
| docker_patch_capture_seconds | 2.511 | 1.506 | 5.580 | 10.645 | 17.506 | 24.615 | 1255.525 |
| docker_verifier_exec_seconds | 23.535 | 10.911 | 37.836 | 77.342 | 127.235 | 2400.130 | 11767.553 |
| docker_setup_seconds | 3.544 | 3.200 | 5.676 | 5.714 | 5.737 | 5.743 | 1771.951 |
| docker_exec_calls | 51.718 | 53.000 | 73.000 | 74.000 | 74.000 | 74.000 | 25859.000 |
| docker_upload_calls | 0.992 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 496.000 |
| turns | 43.080 | 45.000 | 64.000 | 64.000 | 64.000 | 64.000 | 21540.000 |
| shell_calls | 42.592 | 44.000 | 64.000 | 64.000 | 64.000 | 64.000 | 21296.000 |
| cumulative_model_input_tokens | 540451.782 | 456693.500 | 1088724.400 | 1245307.350 | 1565614.300 | 2079363.000 | 270225891.000 |
| model_output_tokens | 13071.520 | 11542.000 | 24352.000 | 29046.100 | 36191.910 | 47384.000 | 6535760.000 |
| tool_observation_tokens | 14786.634 | 13320.500 | 26242.200 | 29952.450 | 36265.550 | 72826.000 | 7393317.000 |
| cached_input_tokens | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| trajectory_tokens | 13071.520 | 11542.000 | 24352.000 | 29046.100 | 36191.910 | 47384.000 | 6535760.000 |
| patch_chars | 2250.700 | 563.000 | 3264.500 | 7062.750 | 44700.600 | 111834.000 | 1116347.000 |
| patch_bytes | 2232.848 | 560.000 | 3240.100 | 7037.350 | 44265.000 | 111837.000 | 1116424.000 |

## Categories

- status: `{"completed": 496, "failed": 4}`
- raw status: `{"completed": 496, "failed": 4}`
- stop reason: `{"max_tokens_per_turn": 95, "max_turns": 158, "no_command": 37, "task_complete": 108, "repeated_command_outcome": 92, "environment_error:TimeoutError": 4, "command_timeout": 6}`
- verifier status: `{"completed": 495, "None": 4, "timeout": 1}`
- mini-SWE-agent version: `{"None": 500}`
