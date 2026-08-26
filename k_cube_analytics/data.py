"""Data access layer.

Resolves Issue #5 (yfinance 1.0): use the free, open-source ``yfinance``
library as the market-data source.

The analytics modules never talk to ``yfinance`` directly. They depend on the
:class:`DataProvider` protocol, which keeps the numeric code pure and testable
(a fake provider can feed it deterministic frames) and makes it trivial to swap
the data source later if Yahoo's scraping breaks -- the documented risk in the
issue.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

# Columns every provider must return from ``get_history``.
REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume", "Dividends")


@runtime_checkable
class DataProvider(Protocol):
    """Abstract source of market data used by every analytics module."""

    def get_history(
        self, symbol: str, period: str = "5y", interval: str = "1d"
    ) -> pd.DataFrame:
        """Return an OHLCV+Dividends frame indexed by date (ascending)."""
        ...

    def get_info(self, symbol: str) -> dict:
        """Return fundamental / descriptive metadata for ``symbol``."""
        ...


def _normalise_history(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee the columns the analytics layer relies on exist and are clean."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS))

    out = df.copy()
    # yfinance sometimes omits Dividends when there were none in the window.
    if "Dividends" not in out.columns:
        out["Dividends"] = 0.0
    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
    out["Dividends"] = out["Dividends"].fillna(0.0)
    out = out.sort_index()
    return out


class YFinanceProvider:
    """Live :class:`DataProvider` backed by ``yfinance``.

    ``yfinance`` is imported lazily so the rest of the package (and its test
    suite) can run without the dependency or a network connection.
    """

    def __init__(self, auto_adjust: bool = True) -> None:
        self.auto_adjust = auto_adjust
        self._info_cache: dict[str, dict] = {}

    def _ticker(self, symbol: str):
        import yfinance as yf

        return yf.Ticker(symbol)

    def get_history(
        self, symbol: str, period: str = "5y", interval: str = "1d"
    ) -> pd.DataFrame:
        hist = self._ticker(symbol).history(
            period=period, interval=interval, auto_adjust=self.auto_adjust
        )
        return _normalise_history(hist)

    def get_info(self, symbol: str) -> dict:
        if symbol not in self._info_cache:
            try:
                self._info_cache[symbol] = dict(self._ticker(symbol).info or {})
            except Exception:  # pragma: no cover - network / scraping failure
                self._info_cache[symbol] = {}
        return self._info_cache[symbol]


def default_provider() -> DataProvider:
    """Return the standard live provider."""
    return YFinanceProvider()
