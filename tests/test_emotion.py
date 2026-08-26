"""Tests for emotion-vs-fundamentals analysis (Issue #7)."""

from __future__ import annotations

import numpy as np

from k_cube_analytics import emotion
from tests.conftest import FakeProvider, make_history


def test_rsi_maxes_out_on_pure_uptrend():
    close = make_history(np.linspace(100, 200, 60))["Close"]
    r = emotion.rsi(close)
    assert abs(r.iloc[-1] - 100.0) < 1e-6


def test_sharp_oversold_decline_reads_as_emotional():
    closes = np.concatenate([np.full(200, 100.0), np.linspace(100, 60, 50)])
    hist = make_history(closes)
    report = emotion.analyse_from_history(
        "PANIC", hist, info={"trailingEps": 5.0, "profitMargins": 0.2}
    )
    assert report.drawdown_from_high < -0.3
    assert report.rsi < 30
    assert report.emotion_index > 0.6
    assert "emotional" in report.verdict
    # Fair floor sits above the panicked last price.
    assert report.hypothetical_fair_floor > report.last_close


def test_healthy_uptrend_is_not_emotional():
    hist = make_history(np.linspace(100, 200, 250))
    report = emotion.analyse_from_history("UP", hist)
    assert report.emotion_index < 0.2
    assert "grounded" in report.verdict


def test_analyse_uses_provider_and_info():
    fp = FakeProvider()
    closes = np.concatenate([np.full(200, 100.0), np.linspace(100, 65, 50)])
    fp.set("Z", make_history(closes), info={"trailingEps": 3.0})
    report = emotion.analyse("Z", fp)
    assert report.symbol == "Z"
    assert report.fundamentals_stable is True
