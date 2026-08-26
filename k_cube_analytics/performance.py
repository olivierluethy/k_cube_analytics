"""Historical performance comparison.

Resolves Issue #2: compare the performance of one or more tickers over their
price history -- normalised growth curves plus the headline risk/return metrics
that make two names comparable on the same footing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import DataProvider

TRADING_DAYS = 252


@dataclass
class PerformanceStats:
    """Summary performance metrics for one ticker over the loaded window."""

    symbol: str
    start: object
    end: object
    total_return: float
    cagr: float
    annual_volatility: float
    sharpe: float  # excess-return Sharpe, risk-free assumed 0
    max_drawdown: float  # most negative peak-to-trough, as a fraction

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "start": self.start,
            "end": self.end,
            "total_return": self.total_return,
            "cagr": self.cagr,
            "annual_volatility": self.annual_volatility,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
        }


def max_drawdown(close: pd.Series) -> float:
    """Largest peak-to-trough decline in ``close`` as a negative fraction."""
    if len(close) < 2:
        return 0.0
    running_max = close.cummax()
    drawdown = close / running_max - 1.0
    return float(drawdown.min())


def _years_span(index: pd.Index) -> float:
    if len(index) < 2:
        return 0.0
    delta = index[-1] - index[0]
    days = getattr(delta, "days", None)
    if days is None:
        return max(len(index) - 1, 1) / TRADING_DAYS
    return max(days, 1) / 365.25


def stats_from_history(symbol: str, history: pd.DataFrame) -> PerformanceStats:
    """Compute performance metrics from a single OHLCV history frame."""
    hist = history.dropna(subset=["Close"])
    hist = hist[hist["Close"] > 0]
    if len(hist) < 2:
        return PerformanceStats(symbol, None, None, 0.0, 0.0, 0.0, 0.0, 0.0)

    close = hist["Close"]
    total_return = float(close.iloc[-1] / close.iloc[0] - 1.0)
    years = _years_span(hist.index)
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    daily = close.pct_change().dropna()
    vol = float(daily.std() * np.sqrt(TRADING_DAYS)) if len(daily) else 0.0
    mean_annual = float(daily.mean() * TRADING_DAYS) if len(daily) else 0.0
    sharpe = mean_annual / vol if vol > 0 else 0.0

    return PerformanceStats(
        symbol=symbol,
        start=hist.index[0],
        end=hist.index[-1],
        total_return=total_return,
        cagr=cagr,
        annual_volatility=vol,
        sharpe=sharpe,
        max_drawdown=max_drawdown(close),
    )


def normalised_curves(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return each ticker's Close rebased to 100 at its first common date.

    Rebasing to a shared start makes growth curves directly comparable on one
    chart regardless of absolute price.
    """
    series = {}
    for symbol, hist in histories.items():
        close = hist.dropna(subset=["Close"])["Close"]
        close = close[close > 0]
        if len(close):
            series[symbol] = close / close.iloc[0] * 100.0
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index()


def compare(
    symbols: list[str], provider: DataProvider, period: str = "5y"
) -> dict:
    """Compare historical performance across ``symbols``.

    Returns a metrics table (ranked by total return), the rebased growth curves,
    and the best/worst performer over the window.
    """
    histories = {s: provider.get_history(s, period=period) for s in symbols}
    rows = [stats_from_history(s, h) for s, h in histories.items()]
    table = pd.DataFrame([r.as_dict() for r in rows]).set_index("symbol")
    table = table.sort_values("total_return", ascending=False)

    ranked = table.index.tolist()
    return {
        "period": period,
        "stats": table,
        "curves": normalised_curves(histories),
        "best": ranked[0] if ranked else None,
        "worst": ranked[-1] if ranked else None,
    }
