from __future__ import annotations

import logging
import time


class RateLimitedWarner:
    def __init__(self, logger: logging.Logger, interval_seconds: float = 60.0):
        self.logger = logger
        self.interval_seconds = interval_seconds
        self._last_warning = 0.0

    def warn(self, message: str, *args) -> None:
        now = time.monotonic()
        if now - self._last_warning < self.interval_seconds:
            return
        self._last_warning = now
        self.logger.warning(message, *args)
