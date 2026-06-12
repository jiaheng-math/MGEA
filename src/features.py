"""Feature definitions for the public review package.

The full query/probe feature implementation is withheld from this anonymous
review artifact. It will be released with the final public code package.
"""

from __future__ import annotations


WITHHELD_MESSAGE = (
    "MGEA feature extraction is withheld in this anonymous review artifact. "
    "This file intentionally preserves the module path without exposing the "
    "full implementation."
)


def query_feature_names() -> list[str]:
    raise RuntimeError(WITHHELD_MESSAGE)


def probe_feature_names() -> list[str]:
    raise RuntimeError(WITHHELD_MESSAGE)


def extract_query_features(*args, **kwargs):
    raise RuntimeError(WITHHELD_MESSAGE)


def extract_probe_features(*args, **kwargs):
    raise RuntimeError(WITHHELD_MESSAGE)
