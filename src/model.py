"""Model utilities for the public review package.

The training and cross-validation utilities used by the MGEA router are
withheld from this anonymous review artifact.
"""

from __future__ import annotations


WITHHELD_MESSAGE = (
    "MGEA router training utilities are withheld in this anonymous review "
    "artifact. This file intentionally preserves the module path without "
    "exposing the full implementation."
)


def make_split(*args, **kwargs):
    raise RuntimeError(WITHHELD_MESSAGE)


def train_and_evaluate(*args, **kwargs):
    raise RuntimeError(WITHHELD_MESSAGE)


def make_cv_folds(*args, **kwargs):
    raise RuntimeError(WITHHELD_MESSAGE)
