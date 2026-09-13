# SWE-bench full-run profile

## Outcome

| Metric | Value |
|---|---:|
| requests | 500 |
| completed | 495 |
| failed | 5 |
| raw_failed | 5 |
| truncated | 0 |
| resolved | 162 |
| unresolved | 333 |
| verifier_infrastructure_errors | 0 |
| warm_image_requests | 500 |
| mini_swe_trajectories | 0 |
| openenv_trajectories | 500 |
| run_wall_seconds | 5375.1562 |
| request_per_second | 0.0930 |
| agent_per_second | 0.0921 |

## Timing and length distributions

| Metric | Mean | P50 | P90 | P95 | P99 | Max | Sum |
|---|---:|---:|---:|---:|---:|---:|---:|
| agent_latency_seconds | 696.048 | 649.037 | 1224.406 | 1427.063 | 1933.893 | 3661.315 | 348024.038 |
| sample_time_seconds | 696.047 | 649.036 | 1224.406 | 1427.063 | 1933.893 | 3661.315 | 348023.340 |
| model_seconds | 623.906 | 555.653 | 1141.502 | 1347.566 | 1852.125 | 3618.430 | 311953.127 |
| tool_seconds | 42.603 | 35.762 | 62.071 | 78.445 | 314.148 | 1007.488 | 21301.616 |
| verifier_queue_seconds | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.008 |
| verifier_seconds | 15.974 | 10.216 | 26.873 | 67.097 | 103.838 | 347.216 | 7907.254 |
| docker_image_inspect_seconds | 0.114 | 0.075 | 0.257 | 0.328 | 0.409 | 0.444 | 57.095 |
| docker_container_start_seconds | 0.987 | 0.417 | 1.938 | 2.614 | 9.609 | 17.998 | 493.279 |
| docker_container_close_seconds | 0.729 | 0.307 | 1.139 | 2.289 | 12.049 | 17.398 | 364.282 |
| docker_exec_seconds | 62.691 | 50.523 | 96.847 | 135.884 | 412.453 | 1026.717 | 31345.693 |
| docker_upload_seconds | 0.103 | 0.087 | 0.146 | 0.177 | 0.362 | 0.664 | 51.747 |
| docker_accounted_seconds | 64.624 | 52.057 | 98.867 | 138.384 | 423.307 | 1029.922 | 32312.097 |
| docker_agent_tool_seconds | 42.603 | 35.761 | 62.071 | 78.444 | 314.147 | 1007.488 | 21301.322 |
| docker_baseline_seconds | 2.773 | 2.321 | 5.361 | 5.732 | 6.386 | 6.765 | 1386.747 |
| docker_patch_capture_seconds | 1.190 | 1.388 | 1.687 | 1.824 | 2.318 | 2.878 | 594.762 |
| docker_verifier_exec_seconds | 15.695 | 10.077 | 26.239 | 66.431 | 103.687 | 347.087 | 7847.551 |
| docker_setup_seconds | 0.431 | 0.280 | 0.813 | 1.017 | 2.249 | 5.524 | 215.311 |
| docker_exec_calls | 52.932 | 57.000 | 73.000 | 73.000 | 74.000 | 74.000 | 26466.000 |
| docker_upload_calls | 0.990 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 495.000 |
| turns | 44.258 | 48.000 | 64.000 | 64.000 | 64.000 | 64.000 | 22129.000 |
| shell_calls | 43.812 | 48.000 | 64.000 | 64.000 | 64.000 | 64.000 | 21906.000 |
| cumulative_model_input_tokens | 565611.796 | 494933.000 | 1114013.900 | 1273407.100 | 1807297.130 | 3508474.000 | 282805898.000 |
| model_output_tokens | 13382.522 | 11430.000 | 24806.200 | 28953.550 | 42038.650 | 91936.000 | 6691261.000 |
| tool_observation_tokens | 14771.750 | 13610.500 | 25807.300 | 31170.500 | 36687.220 | 46362.000 | 7385875.000 |
| cached_input_tokens | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| trajectory_tokens | 13382.522 | 11430.000 | 24806.200 | 28953.550 | 42038.650 | 91936.000 | 6691261.000 |
| patch_chars | 3607.523 | 536.000 | 2783.200 | 5472.900 | 17839.640 | 543453.000 | 1785724.000 |
| patch_bytes | 3571.844 | 531.500 | 2759.700 | 5414.650 | 17374.220 | 543599.000 | 1785922.000 |

## Categories

- status: `{"completed": 495, "failed": 5}`
- raw status: `{"completed": 495, "failed": 5}`
- stop reason: `{"max_tokens_per_turn": 82, "repeated_command_outcome": 97, "max_turns": 178, "task_complete": 106, "no_command": 30, "environment_error:TimeoutError": 4, "command_timeout": 2, "environment_error:RuntimeError": 1}`
- verifier status: `{"completed": 495, "None": 5}`
- mini-SWE-agent version: `{"None": 500}`
