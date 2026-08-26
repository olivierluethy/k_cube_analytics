"""Price-return vs. dividend-return analysis.

Resolves Issue #1: the relationship between past *price* returns and the return
that comes purely from *dividends* -- looking at the dividend component on its
own, independent of price appreciation.

Total shareholder return splits into two parts:

* **price return**  -- appreciation of the quote itself, and
* **dividend return** -- cash paid out, expressed as a yield on the price.

This module measures each separately and quantifies how they relate across a
basket of tickers (do high price-return names also pay well, or is it a
trade-off?).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import DataProvider

TRADING_DAYS = 252


@dataclass
class ReturnBreakdown:
    """Decomposition of a single ticker's return over the loaded window."""

    symbol: str
    years: float
    price_return: float  # cumulative, from price appreciation only
    dividend_return: float  # cumulative, from dividends only (reinvested)
    total_return: float
    price_cagr: float
    dividend_cagr: float
    avg_dividend_yield: float  # mean annual dividend / price

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "years": round(self.years, 3),
            "price_return": self.price_return,
            "dividend_return": self.dividend_return,
            "total_return": self.total_return,
            "price_cagr": self.price_cagr,
            "dividend_cagr": self.dividend_cagr,
            "avg_dividend_yield": self.avg_dividend_yield,
        }


def _years_span(index: pd.Index) -> float:
    if len(index) < 2:
        return 0.0
    delta = index[-1] - index[0]
    days = getattr(delta, "days", None)
    if days is None:  # non-datetime index (e.g. plain ints in tests)
        return max(len(index) - 1, 1) / TRADING_DAYS
    return max(days, 1) / 365.25


def _cagr(cumulative_return: float, years: float) -> float:
    if years <= 0:
        return 0.0
    return (1.0 + cumulative_return) ** (1.0 / years) - 1.0


def breakdown_from_history(symbol: str, history: pd.DataFrame) -> ReturnBreakdown:
    """Split a single history frame into its price and dividend components.

    * Price return uses only the Close column.
    * Dividend return treats each payout as cash reinvested at that day's close,
      isolating the yield component with **no** price appreciation counted.
    """
    hist = history.dropna(subset=["Close"])
    hist = hist[hist["Close"] > 0]
    if len(hist) < 2:
        return ReturnBreakdown(symbol, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    close = hist["Close"]
    dividends = hist.get("Dividends", pd.Series(0.0, index=hist.index)).fillna(0.0)

    price_return = float(close.iloc[-1] / close.iloc[0] - 1.0)

    # Dividend-only growth: reinvest each payout in more shares; track how the
    # share count (and thus payout stream) compounds, ignoring price moves.
    shares = 1.0
    dividend_cash = 0.0
    for price, div in zip(close, dividends):
        if div > 0 and price > 0:
            payout = shares * div
            dividend_cash += payout
            shares += payout / price  # reinvest at the current price
    # Value from dividends alone, relative to the initial one-share stake.
    dividend_return = float(dividend_cash / close.iloc[0])

    total_return = float((1.0 + price_return) * (1.0 + dividend_return) - 1.0)

    years = _years_span(hist.index)
    # Average annual dividend yield across the window.
    annual_div = dividends.sum() / years if years > 0 else 0.0
    avg_yield = float(annual_div / close.mean()) if close.mean() > 0 else 0.0

    return ReturnBreakdown(
        symbol=symbol,
        years=years,
        price_return=price_return,
        dividend_return=dividend_return,
        total_return=total_return,
        price_cagr=_cagr(price_return, years),
        dividend_cagr=_cagr(dividend_return, years),
        avg_dividend_yield=avg_yield,
    )


def breakdown(
    symbol: str, provider: DataProvider, period: str = "5y"
) -> ReturnBreakdown:
    """Load ``symbol`` via ``provider`` and return its return breakdown."""
    history = provider.get_history(symbol, period=period)
    return breakdown_from_history(symbol, history)


def relation(
    symbols: list[str], provider: DataProvider, period: str = "5y"
) -> dict:
    """Quantify the relationship between price returns and dividend returns.

    Returns per-symbol breakdowns plus the cross-sectional correlation between
    price CAGR and dividend CAGR -- the core question in Issue #1.
    """
    rows = [breakdown(s, provider, period=period) for s in symbols]
    table = pd.DataFrame([r.as_dict() for r in rows]).set_index("symbol")

    correlation = np.nan
    if len(table) >= 2:
        pc = table["price_cagr"]
        dc = table["dividend_cagr"]
        if pc.std() > 0 and dc.std() > 0:
            correlation = float(pc.corr(dc))

    return {
        "period": period,
        "breakdowns": table,
        "price_vs_dividend_cagr_corr": correlation,
    }
