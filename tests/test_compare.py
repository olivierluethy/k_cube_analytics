"""Tests for stock-vs-stock ranking by mode (Issue #4)."""

from __future__ import annotations

import numpy as np
import pytest

from k_cube_analytics import compare
from tests.conftest import FakeProvider, make_history


@pytest.fixture
def provider():
    fp = FakeProvider()
    # GROWTH: strong price gain, no dividends.
    fp.set("GROWTH", make_history(np.linspace(100, 250, 250)))
    # INCOME: modest price gain, generous dividends.
    fp.set(
        "INCOME",
        make_history(
            np.linspace(100, 110, 250),
            dividends=[0] * 200 + [3.0] + [0] * 24 + [3.0] + [0] * 24,
        ),
    )
    return fp


def test_price_mode_prefers_growth(provider):
    res = compare.rank(["GROWTH", "INCOME"], provider, mode="price", period="1y")
    assert res["winner"] == "GROWTH"
    assert res["score_column"] == "price_cagr"


def test_dividend_mode_prefers_income(provider):
    res = compare.rank(["GROWTH", "INCOME"], provider, mode="dividend",
                       period="1y")
    assert res["winner"] == "INCOME"
    assert res["score_column"] == "dividend_cagr"


def test_both_mode_uses_total(provider):
    res = compare.rank(["GROWTH", "INCOME"], provider, mode="both", period="1y")
    assert res["score_column"] == "total_cagr"
    assert set(res["ranking"].index) == {"GROWTH", "INCOME"}
    assert list(res["ranking"]["rank"]) == [1, 2]


def test_invalid_mode_raises(provider):
    with pytest.raises(ValueError):
        compare.rank(["GROWTH"], provider, mode="nonsense")
