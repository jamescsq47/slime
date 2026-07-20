"""Optional, failure-isolated observability for slime training runs.

The implementation is adapted from the dashboard proposed in
https://github.com/radixark/miles/pull/1654 at commit
d9189010bc3ba407cf0189389015032096a7c725.  It intentionally keeps the
collector and all producer hooks optional so monitoring cannot become a
training dependency.
"""

COLLECTOR_NAME_PREFIX = "slime_dashboard_collector"
