"""Shared test fixtures: synthetic data and a fake data provider."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_history(
    closes,
    dividends=None,
    volumes=None,
    opens=None,
    start="2020-01-01",
    freq="B",
):
    """Build an OHLCV+Dividends frame from a list of closing prices."""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    index = pd.date_range(start=start, periods=n, freq=freq)
    if dividends is None:
        dividends = np.zeros(n)
    if volumes is None:
        volumes = np.full(n, 1_000_000.0)
    if opens is None:
        opens = closes.copy()
    return pd.DataFrame(
        {
            "Open": np.asarray(opens, dtype=float),
            "High": np.maximum(opens, closes),
            "Low": np.minimum(opens, closes),
            "Close": closes,
            "Volume": np.asarray(volumes, dtype=float),
            "Dividends": np.asarray(dividends, dtype=float),
        },
        index=index,
    )


class FakeProvider:
    """In-memory :class:`DataProvider` for deterministic tests."""

    def __init__(self, histories=None, infos=None):
        self._histories = histories or {}
        self._infos = infos or {}

    def set(self, symbol, history, info=None):
        self._histories[symbol] = history
        if info is not None:
            self._infos[symbol] = info

    def get_history(self, symbol, period="5y", interval="1d"):
        if symbol not in self._histories:
            raise KeyError(symbol)
        return self._histories[symbol]

    def get_info(self, symbol):
        return self._infos.get(symbol, {})


@pytest.fixture
def fake_provider():
    return FakeProvider()
