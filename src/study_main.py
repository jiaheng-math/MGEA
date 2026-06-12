"""Entry point placeholder for the anonymous review package.

The full MGEA experiment runner is withheld from this anonymous review
artifact. The repository keeps dataset preparation, environment setup, and
non-sensitive evaluation utilities, but the main routing/retrieval orchestration
will be released with the final public code package.
"""

from __future__ import annotations


WITHHELD_MESSAGE = (
    "The full MGEA study runner is withheld in this anonymous review artifact. "
    "It will be released with the final public code package."
)


def main() -> None:
    raise RuntimeError(WITHHELD_MESSAGE)


if __name__ == "__main__":
    main()
