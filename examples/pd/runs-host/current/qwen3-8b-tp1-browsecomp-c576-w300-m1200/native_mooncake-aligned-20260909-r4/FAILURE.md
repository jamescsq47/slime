# Invalid run: search endpoint port mismatch

This run is not a throughput result. The BrowseComp workload resolved
`search_url=http://127.0.0.1:8750`, while the search service was accidentally
launched on port `8710`. Consequently 6,767 requests aborted with
`search_backend_error`, all recorded trajectories had one model call, and the
apparent Decode throughput measured the failure path rather than BrowseComp.

PD transport and model serving remained healthy throughout the 300+1200 second
window. The failure was in experiment launch configuration, not the native
Mooncake data path.
