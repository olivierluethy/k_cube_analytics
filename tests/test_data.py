"""Tests for the data layer normalisation (Issue #5)."""

from __future__ import annotations

import pandas as pd

from k_cube_analytics.data import (
    REQUIRED_COLUMNS,
    DataProvider,
    _normalise_history,
)
from tests.conftest import FakeProvider, make_history


def test_normalise_fills_missing_dividends_column():
    df = pd.DataFrame(
        {"Open": [1], "High": [1], "Low": [1], "Close": [1], "Volume": [1]},
        index=pd.date_range("2020-01-01", periods=1, freq="B"),
    )
    out = _normalise_history(df)
    assert "Dividends" in out.columns
    assert out["Dividends"].iloc[0] == 0.0


def test_normalise_empty_returns_required_columns():
    out = _normalise_history(pd.DataFrame())
    for col in REQUIRED_COLUMNS:
        assert col in out.columns
    assert len(out) == 0


def test_normalise_sorts_index():
    df = make_history([3, 1, 2])
    shuffled = df.iloc[[2, 0, 1]]
    out = _normalise_history(shuffled)
    assert list(out.index) == sorted(out.index)


def test_fake_provider_is_a_dataprovider():
    fp = FakeProvider()
    assert isinstance(fp, DataProvider)
