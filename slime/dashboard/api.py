from __future__ import annotations

from contextlib import nullcontext


class _NoopSpan:
    def set(self, key, value):
        return self

    def update(self, attrs):
        return self


def span(args, target, name: str, attrs: dict | None = None):
    if not getattr(args, "use_slime_dashboard", False):
        return nullcontext(_NoopSpan())
    from slime.utils.trace_utils import trace_span

    return trace_span(target, name, attrs=attrs)


def event(args, target, name: str, attrs: dict | None = None) -> None:
    if not getattr(args, "use_slime_dashboard", False):
        return
    from slime.utils.trace_utils import trace_event

    trace_event(target, name, attrs=attrs)


def metrics(args, values: dict, step: int | None = None, step_key: str | None = None) -> None:
    if not getattr(args, "use_slime_dashboard", False):
        return
    from slime.dashboard.backend import log_metrics

    log_metrics(values, step=step, step_key=step_key)
