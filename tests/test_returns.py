"""Tests for price- vs dividend-return analysis (Issue #1)."""

from __future__ import annotations

import numpy as np

from k_cube_analytics import returns
from tests.conftest import FakeProvider, make_history


def test_pure_price_appreciation_has_no_dividend_return():
    hist = make_history(np.linspace(100, 200, 250))
    b = returns.breakdown_from_history("UP", hist)
    # 200/100 - 1 = 1.0
    assert abs(b.price_return - 1.0) < 1e-9
    assert b.dividend_return == 0.0
    assert abs(b.total_return - 1.0) < 1e-9


def test_flat_price_with_dividends_isolates_dividend_return():
    # Price flat at 100, pays 2.0 dividend four times => 8 on a 100 base.
    closes = [100.0] * 8
    divs = [0, 0, 2.0, 0, 2.0, 0, 2.0, 2.0]
    hist = make_history(closes, dividends=divs)
    b = returns.breakdown_from_history("DIV", hist)
    assert abs(b.price_return) < 1e-9
    # 8.0 of dividends on a 100 base, slightly more once each payout is
    # reinvested into additional shares that then pay out themselves.
    assert 0.08 <= b.dividend_return < 0.09
    assert b.dividend_return > 0


def test_relation_correlation_present_for_multiple_symbols():
    fp = FakeProvider()
    fp.set("A", make_history(np.linspace(100, 150, 250)))
    fp.set("B", make_history(np.linspace(100, 120, 250),
                             dividends=[0] * 249 + [5.0]))
    res = returns.relation(["A", "B"], fp, period="1y")
    assert "breakdowns" in res
    assert set(res["breakdowns"].index) == {"A", "B"}
    # correlation is defined (may be nan if variance is zero, but here it is not)
    assert "price_vs_dividend_cagr_corr" in res


def test_insufficient_data_is_safe():
    hist = make_history([100.0])
    b = returns.breakdown_from_history("X", hist)
    assert b.total_return == 0.0
