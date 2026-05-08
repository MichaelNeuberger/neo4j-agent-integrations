"""Tests for the small render helpers shared across interactive_demo scenarios.

The demo currently inlines the same ``_sim_bar`` closure five times in
five different scenarios. This file pins the contract of the extracted
module-level helper so any future tweak to the colour bands or width
defaults gets caught immediately.
"""

from __future__ import annotations

import re

import pytest

from scripts.demo_helpers import sim_bar


@pytest.fixture(autouse=True)
def clean_test_data():
    """Override the conftest autouse fixture — pure-Python helpers, no Neo4j."""
    yield


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _strip(s: str) -> str:
    return _ANSI.sub("", s)


def _filled_count(s: str) -> int:
    return _strip(s).count("█")


def _empty_count(s: str) -> int:
    return _strip(s).count("░")


class TestSimBarShape:
    def test_default_width_is_20(self):
        bar = sim_bar(0.5)
        assert _filled_count(bar) + _empty_count(bar) == 20

    def test_custom_width_respected(self):
        bar = sim_bar(0.5, width=40)
        assert _filled_count(bar) + _empty_count(bar) == 40

    def test_zero_similarity_renders_empty(self):
        bar = sim_bar(0.0)
        assert _filled_count(bar) == 0
        assert _empty_count(bar) == 20

    def test_full_similarity_renders_full(self):
        bar = sim_bar(1.0)
        assert _filled_count(bar) == 20
        assert _empty_count(bar) == 0


class TestSimBarClamp:
    """Cosine similarities >1.0 or <0 must not break the layout."""

    def test_above_one_clamps_to_full(self):
        bar = sim_bar(1.7, width=10)
        assert _filled_count(bar) == 10
        assert _empty_count(bar) == 0

    def test_below_zero_clamps_to_empty(self):
        bar = sim_bar(-0.3, width=10)
        assert _filled_count(bar) == 0
        assert _empty_count(bar) == 10


class TestSimBarColours:
    """Three colour bands: green > 0.4, yellow (0.2, 0.4], red <= 0.2."""

    def test_high_similarity_is_green(self):
        bar = sim_bar(0.6)
        assert "\x1b[32m" in bar       # GREEN

    def test_mid_similarity_is_yellow(self):
        bar = sim_bar(0.3)
        assert "\x1b[33m" in bar       # YELLOW

    def test_low_similarity_is_red(self):
        bar = sim_bar(0.1)
        assert "\x1b[31m" in bar       # RED

    def test_boundary_at_0_4_is_yellow(self):
        # > 0.4 is strict; sim == 0.4 falls into the yellow band.
        bar = sim_bar(0.4)
        assert "\x1b[33m" in bar
        assert "\x1b[32m" not in bar

    def test_boundary_at_0_2_is_red(self):
        # > 0.2 is strict; sim == 0.2 falls into the red band.
        bar = sim_bar(0.2)
        assert "\x1b[31m" in bar
        assert "\x1b[33m" not in bar
