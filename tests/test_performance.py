"""Tests for historical performance comparison (Issue #2)."""

from __future__ import annotations

import numpy as np

from k_cube_analytics import performance
from tests.conftest import FakeProvider, make_history


def test_total_return_and_cagr():
    hist = make_history(np.linspace(100, 200, 253))  # ~1 year of B days
    stats = performance.stats_from_history("X", hist)
    assert abs(stats.total_return - 1.0) < 1e-9
    assert stats.cagr > 0


def test_max_drawdown_detects_trough():
    # up to 120, down to 60 (-50% peak-to-trough), back to 90
    closes = [100, 110, 120, 90, 60, 75, 90]
    hist = make_history(closes)
    stats = performance.stats_from_history("X", hist)
    assert abs(stats.max_drawdown - (-0.5)) < 1e-9


def test_compare_ranks_best_and_worst():
    fp = FakeProvider()
    fp.set("WIN", make_history(np.linspace(100, 300, 100)))
    fp.set("LOSE", make_history(np.linspace(100, 80, 100)))
    res = performance.compare(["WIN", "LOSE"], fp, period="1y")
    assert res["best"] == "WIN"
    assert res["worst"] == "LOSE"
    assert list(res["stats"].index) == ["WIN", "LOSE"]


def test_normalised_curves_start_at_100():
    fp = FakeProvider()
    h1 = make_history(np.linspace(50, 100, 50))
    h2 = make_history(np.linspace(200, 400, 50))
    curves = performance.normalised_curves({"A": h1, "B": h2})
    assert abs(curves["A"].iloc[0] - 100.0) < 1e-9
    assert abs(curves["B"].iloc[0] - 100.0) < 1e-9
