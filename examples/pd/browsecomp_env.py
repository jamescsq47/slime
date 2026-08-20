"""Compatibility alias for :mod:`data.browsecomp.env`."""

import sys

from data.browsecomp import env as _implementation


sys.modules[__name__] = _implementation
