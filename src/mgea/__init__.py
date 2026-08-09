"""MGEA: conditional marginal allocation of Graph RAG evidence slots."""

from .schema import Example, Passage
from .slot_allocator import SlotAllocator, SlotConfig
from .triage import DenseProbeTriage, TriageConfig

__all__ = [
    "DenseProbeTriage",
    "Example",
    "Passage",
    "SlotAllocator",
    "SlotConfig",
    "TriageConfig",
]
