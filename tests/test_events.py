"""Tests for swing detection and emotional assessment (Issue #3)."""

from __future__ import annotations

from k_cube_analytics import events
from tests.conftest import FakeProvider, make_history


def test_detects_only_moves_above_threshold():
    # A 2% wobble then a 10% drop; threshold 5% should catch only the drop.
    closes = [100] * 25 + [98, 90, 90, 90, 90, 90]
    opens = [100] * 25 + [98, 90, 90, 90, 90, 90]
    hist = make_history(closes, opens=opens)
    evs = events.detect_from_history(hist, threshold=0.05)
    assert len(evs) == 1
    assert evs[0].daily_return < -0.05


def test_reversal_reads_as_emotional():
    # Intraday panic to 90 (opened flat at 100), fully recovers to 100.
    closes = [100] * 25 + [90, 92, 94, 96, 98, 100]
    opens = [100] * 25 + [100, 92, 94, 96, 98, 100]
    hist = make_history(closes, opens=opens)
    evs = events.detect_from_history(hist, threshold=0.05, follow_window=5)
    assert len(evs) == 1
    assert evs[0].reversal > 0.9
    assert evs[0].emotional_score >= 0.66
    assert "emotional" in evs[0].verdict


def test_gap_and_volume_hold_reads_as_justified():
    # Gaps down 10% on 5x volume and stays there -> justified repricing.
    closes = [100] * 25 + [90] * 6
    opens = [100] * 25 + [90] * 6
    volumes = [1_000_000.0] * 25 + [5_000_000.0] + [1_000_000.0] * 5
    hist = make_history(closes, opens=opens, volumes=volumes)
    evs = events.detect_from_history(hist, threshold=0.05, follow_window=5)
    assert len(evs) == 1
    assert evs[0].reversal < 0.1
    assert evs[0].volume_ratio > 2.0
    assert evs[0].emotional_score <= 0.33
    assert "justified" in evs[0].verdict


def test_detect_via_provider_returns_dataframe():
    fp = FakeProvider()
    closes = [100] * 25 + [90, 92, 94, 96, 98, 100]
    fp.set("X", make_history(closes))
    df = events.detect("X", fp, threshold=0.05)
    assert "emotional_score" in df.columns
    assert len(df) >= 1
