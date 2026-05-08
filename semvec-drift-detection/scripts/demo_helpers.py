"""Pure-Python render helpers shared across interactive_demo scenarios.

Kept separate from ``interactive_demo.py`` so they can be unit-tested
without spinning up the demo's environment guard (which requires a
fully populated ``.env`` and a reachable Neo4j).
"""

from __future__ import annotations

# ANSI colour codes used by the demo. Duplicated here on purpose so the
# helpers stay importable even before the main demo module loads.
_GREEN = "\x1b[32m"
_YELLOW = "\x1b[33m"
_RED = "\x1b[31m"
_RESET = "\x1b[0m"

_FILLED = "█"
_EMPTY = "░"


def sim_bar(sim: float, width: int = 20) -> str:
    """Render a cosine-similarity bar with three semantic colour bands.

    Bands (matching the demo's existing visual language):

    - ``sim > 0.4``     → green  (on-topic)
    - ``0.2 < sim ≤ 0.4`` → yellow (shifting)
    - ``sim ≤ 0.2``     → red    (off-topic / drifted)

    Cosine similarity can drift slightly outside ``[0, 1]`` due to
    floating-point normalisation; the bar clamps so the layout never
    breaks.
    """
    clamped = max(0.0, min(sim, 1.0))
    filled = int(clamped * width)
    bar = f"{_FILLED * filled}{_EMPTY * (width - filled)}"
    if sim > 0.4:
        colour = _GREEN
    elif sim > 0.2:
        colour = _YELLOW
    else:
        colour = _RED
    return f"{colour}{bar}{_RESET}"
