"""Compatibility alias for :mod:`data.browsecomp.agent`."""

import sys

from data.browsecomp import agent as _implementation


sys.modules[__name__] = _implementation
