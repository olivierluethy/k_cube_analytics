"""Tests for event -> connected stocks (Issue #6)."""

from __future__ import annotations

import numpy as np

from k_cube_analytics import event_stocks
from tests.conftest import FakeProvider, make_history


def test_matches_war_theme():
    themes = event_stocks.match_themes("Ukraine war escalates")
    names = [t.name for t in themes]
    assert "armed conflict / war" in names
    assert "LMT" in themes[0].beneficiaries


def test_matches_oil_theme_for_venezuela():
    themes = event_stocks.match_themes("Venezuela sanctions on oil exports")
    names = [t.name for t in themes]
    assert "oil supply shock" in names


def test_no_match_returns_empty():
    assert event_stocks.match_themes("quiet uneventful tuesday") == []


def test_ranking_by_reaction_puts_beneficiary_first():
    fp = FakeProvider()
    # Defense name rallies, airline falls after the event date.
    fp.set("LMT", make_history(np.linspace(100, 160, 300), start="2022-01-03"))
    fp.set("DAL", make_history(np.linspace(100, 70, 300), start="2022-01-03"))
    res = event_stocks.analyse_event(
        "Ukraine war", fp, event_date="2022-02-24", window=20
    )
    assert res["matched_themes"] == ["armed conflict / war"]
    assert res["top_beneficiary"] == "LMT"
    ranking = res["ranking"]
    assert ranking.loc["LMT", "reaction"] > 0
    assert ranking.loc["LMT", "side"] == "beneficiary"


def test_no_date_returns_baskets_only():
    fp = FakeProvider()
    res = event_stocks.analyse_event("Ukraine war", fp)
    assert res["connected_beneficiaries"]
    assert len(res["ranking"]) == 0


def test_extra_candidates_included():
    fp = FakeProvider()
    fp.set("XYZ", make_history(np.linspace(100, 200, 300), start="2022-01-03"))
    res = event_stocks.analyse_event(
        "AI boom", fp, event_date="2022-02-24",
        extra_candidates=["XYZ"], window=20,
    )
    assert "XYZ" in res["connected_beneficiaries"]
    assert "XYZ" in res["ranking"].index
