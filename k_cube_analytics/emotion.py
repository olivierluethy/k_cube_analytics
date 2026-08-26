"""Emotion-vs-fundamentals analysis of a decline.

Resolves Issue #7: when a stock is falling, gauge how much of the drop is
emotionally driven versus grounded in a real, logical reason -- and estimate,
hypothetically, how far sentiment alone may have pushed it below a
fundamentally defensible level.

As with :mod:`events`, this is a transparent scoring of measurable signals, not
a claim of certainty:

* **RSI** -- a deeply oversold RSI is the classic fingerprint of panic selling.
* **Stretch below trend** -- how far price sits below its long moving average,
  in units of its own volatility; extreme stretch is unsustainable and tends to
  mean-revert, a sign of emotional overshoot.
* **Fundamental stability** -- if valuation multiples / earnings have *not*
  deteriorated while the price collapsed, the move is harder to justify on
  fundamentals and reads as more emotional.

These blend into an ``emotion_index`` in ``[0, 1]`` (1 = the decline looks
almost entirely emotional).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import DataProvider


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder-style Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # When there are no losses at all, RSI is 100.
    out = out.where(avg_loss != 0.0, 100.0)
    return out


@dataclass
class EmotionReport:
    symbol: str
    last_close: float
    drawdown_from_high: float  # current decline vs. window peak
    rsi: float
    stretch_below_trend: float  # negative = below MA, in volatility units
    fundamentals_stable: bool | None
    emotion_index: float  # 0 = fundamentally driven, 1 = emotional
    verdict: str
    hypothetical_fair_floor: float  # rough level once emotion is stripped out

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "last_close": self.last_close,
            "drawdown_from_high": self.drawdown_from_high,
            "rsi": self.rsi,
            "stretch_below_trend": self.stretch_below_trend,
            "fundamentals_stable": self.fundamentals_stable,
            "emotion_index": self.emotion_index,
            "verdict": self.verdict,
            "hypothetical_fair_floor": self.hypothetical_fair_floor,
        }


def _verdict(index: float) -> str:
    if index >= 0.66:
        return "decline looks largely emotional"
    if index <= 0.33:
        return "decline looks fundamentally grounded"
    return "mixed: partly emotional, partly justified"


def _fundamentals_stable(info: dict | None) -> bool | None:
    """Best-effort read of whether fundamentals still look healthy.

    Uses positive trailing earnings / a non-stretched forward multiple as a
    proxy. Returns ``None`` when there is not enough info to judge.
    """
    if not info:
        return None
    signals = []
    eps = info.get("trailingEps")
    if eps is not None:
        signals.append(eps > 0)
    fwd_pe = info.get("forwardPE")
    if fwd_pe is not None:
        signals.append(0 < fwd_pe < 40)
    margins = info.get("profitMargins")
    if margins is not None:
        signals.append(margins > 0)
    if not signals:
        return None
    return sum(signals) >= (len(signals) / 2.0)


def analyse_from_history(
    symbol: str,
    history: pd.DataFrame,
    info: dict | None = None,
    trend_window: int = 200,
) -> EmotionReport:
    """Score how emotional the current decline in ``history`` looks."""
    hist = history.dropna(subset=["Close"])
    hist = hist[hist["Close"] > 0]
    if len(hist) < 2:
        return EmotionReport(
            symbol, 0.0, 0.0, float("nan"), 0.0, None, 0.0,
            "insufficient data", 0.0,
        )

    close = hist["Close"]
    last = float(close.iloc[-1])
    peak = float(close.cummax().iloc[-1])
    drawdown = last / peak - 1.0 if peak > 0 else 0.0

    rsi_series = rsi(close)
    last_rsi = float(rsi_series.iloc[-1]) if not rsi_series.isna().all() else 50.0
    if np.isnan(last_rsi):
        last_rsi = 50.0

    window = min(trend_window, len(close))
    ma = close.rolling(window, min_periods=1).mean()
    daily_std = close.pct_change().std() or 0.0
    if daily_std > 0 and ma.iloc[-1] > 0:
        stretch = float((last / ma.iloc[-1] - 1.0) / daily_std)
    else:
        stretch = 0.0

    fundamentals_stable = _fundamentals_stable(info)

    # --- blend into an emotion index -------------------------------------
    # RSI: 30 -> emotional, 70 -> not. Map to 0..1 (lower RSI = more emotional).
    rsi_c = float(np.clip((50.0 - last_rsi) / 40.0, 0.0, 1.0))
    # Stretch below trend: -3 std or lower reads as heavy overshoot.
    stretch_c = float(np.clip(-stretch / 3.0, 0.0, 1.0)) if stretch < 0 else 0.0
    # Fundamentals still healthy while price collapses -> more emotional.
    if fundamentals_stable is True:
        fundamental_c = 1.0
    elif fundamentals_stable is False:
        fundamental_c = 0.0
    else:
        fundamental_c = 0.5  # unknown -> neutral

    # Emotion signals only matter when the stock is actually down.
    down_gate = float(np.clip(-drawdown / 0.1, 0.0, 1.0))  # full weight by -10%
    raw = 0.45 * rsi_c + 0.30 * stretch_c + 0.25 * fundamental_c
    emotion_index = float(np.clip(raw * down_gate, 0.0, 1.0))

    # Hypothetical "fair floor": if part of the drop is emotional, a defensible
    # level sits above the current price by the emotional fraction of the
    # drawdown from the peak.
    emotional_drop = emotion_index * (peak - last)
    fair_floor = float(last + emotional_drop)

    return EmotionReport(
        symbol=symbol,
        last_close=last,
        drawdown_from_high=float(drawdown),
        rsi=last_rsi,
        stretch_below_trend=stretch,
        fundamentals_stable=fundamentals_stable,
        emotion_index=emotion_index,
        verdict=_verdict(emotion_index),
        hypothetical_fair_floor=fair_floor,
    )


def analyse(
    symbol: str,
    provider: DataProvider,
    period: str = "2y",
    use_fundamentals: bool = True,
) -> EmotionReport:
    """Load ``symbol`` and score how emotional its current decline looks."""
    history = provider.get_history(symbol, period=period)
    info = provider.get_info(symbol) if use_fundamentals else None
    return analyse_from_history(symbol, history, info=info)
