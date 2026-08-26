"""Large-move detection and justified-vs-emotional assessment.

Resolves Issue #3: on big price swings, surface *when* something happened and
judge whether the move looks justified or is an exaggerated, emotionally driven
overreaction.

There is no oracle for "was this justified", so the judgment is a transparent,
rule-based read of measurable market behaviour around the move rather than a
black box. The signals:

* **Volume confirmation** -- a real, news-driven repricing tends to trade on
  heavy volume as the market digests information. A move on ordinary volume is
  more likely noise/sentiment.
* **Follow-through vs. reversal** -- if the move largely reverses over the next
  few sessions, the market itself decided it overreacted (emotional). If it
  holds, the repricing stuck (justified).
* **Gap** -- a large opening gap points to a discrete overnight catalyst
  (earnings, news), i.e. a fundamental trigger.

These combine into an ``emotional_score`` in ``[0, 1]`` (1 = looks purely
emotional) and a human-readable ``verdict``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import DataProvider


@dataclass
class SwingEvent:
    date: object
    daily_return: float  # the move that triggered detection
    z_score: float  # move in units of the ticker's own daily volatility
    volume_ratio: float  # volume / trailing average volume
    gap: float  # open vs. prior close, as a fraction
    reversal: float  # fraction of the move reversed over the follow window
    emotional_score: float  # 0 = justified/fundamental, 1 = emotional
    verdict: str

    def as_dict(self) -> dict:
        return {
            "date": self.date,
            "daily_return": self.daily_return,
            "z_score": self.z_score,
            "volume_ratio": self.volume_ratio,
            "gap": self.gap,
            "reversal": self.reversal,
            "emotional_score": self.emotional_score,
            "verdict": self.verdict,
        }


def _verdict(score: float) -> str:
    if score >= 0.66:
        return "likely emotional overreaction"
    if score <= 0.33:
        return "likely justified repricing"
    return "mixed / inconclusive"


def _emotional_score(volume_ratio: float, reversal: float, gap: float) -> float:
    """Blend the signals into a 0..1 emotionality score.

    High reversal pushes toward *emotional*; strong volume confirmation and a
    large catalyst gap push toward *justified*.
    """
    # Each component is squashed into 0..1.
    reversal_c = float(np.clip(reversal, 0.0, 1.0))  # more reversal -> emotional
    volume_c = float(np.clip((volume_ratio - 1.0) / 3.0, 0.0, 1.0))  # conviction
    gap_c = float(np.clip(abs(gap) / 0.05, 0.0, 1.0))  # discrete catalyst

    # Weighted: reversal dominates, volume & gap temper it toward "justified".
    score = 0.6 * reversal_c + 0.4 * (1.0 - volume_c) * 0.5 + 0.4 * (
        1.0 - gap_c
    ) * 0.5
    return float(np.clip(score, 0.0, 1.0))


def detect_from_history(
    history: pd.DataFrame,
    threshold: float = 0.05,
    follow_window: int = 5,
    volume_window: int = 20,
) -> list[SwingEvent]:
    """Find days whose move exceeds ``threshold`` and assess each.

    Parameters
    ----------
    threshold:
        Minimum absolute daily return to flag as a swing (e.g. 0.05 = 5%).
    follow_window:
        Sessions after the move used to measure reversal / follow-through.
    volume_window:
        Trailing window for the average-volume baseline.
    """
    hist = history.dropna(subset=["Close"])
    hist = hist[hist["Close"] > 0]
    if len(hist) < 3:
        return []

    close = hist["Close"]
    daily = close.pct_change()
    vol_avg = (
        hist["Volume"].rolling(volume_window, min_periods=1).mean()
        if "Volume" in hist
        else pd.Series(np.nan, index=hist.index)
    )
    daily_std = daily.std() or 1.0

    events: list[SwingEvent] = []
    n = len(hist)
    for i in range(1, n):
        move = daily.iloc[i]
        if pd.isna(move) or abs(move) < threshold:
            continue

        price = close.iloc[i]
        prior_close = close.iloc[i - 1]
        open_px = hist["Open"].iloc[i] if "Open" in hist else prior_close
        gap = (open_px / prior_close - 1.0) if prior_close > 0 else 0.0

        base_vol = vol_avg.iloc[i]
        cur_vol = hist["Volume"].iloc[i] if "Volume" in hist else np.nan
        volume_ratio = (
            float(cur_vol / base_vol)
            if base_vol and base_vol > 0 and not pd.isna(cur_vol)
            else 1.0
        )

        # Reversal: how much of the move is undone over the follow window.
        end = min(i + follow_window, n - 1)
        future_price = close.iloc[end]
        move_size = price - prior_close
        if abs(move_size) > 0:
            recovered = (price - future_price) / move_size
            reversal = float(np.clip(recovered, 0.0, 1.0))
        else:
            reversal = 0.0

        score = _emotional_score(volume_ratio, reversal, gap)
        events.append(
            SwingEvent(
                date=hist.index[i],
                daily_return=float(move),
                z_score=float(move / daily_std),
                volume_ratio=volume_ratio,
                gap=float(gap),
                reversal=reversal,
                emotional_score=score,
                verdict=_verdict(score),
            )
        )
    return events


def detect(
    symbol: str,
    provider: DataProvider,
    period: str = "2y",
    threshold: float = 0.05,
    follow_window: int = 5,
) -> pd.DataFrame:
    """Detect and assess swings for ``symbol``; return one row per event."""
    history = provider.get_history(symbol, period=period)
    events = detect_from_history(
        history, threshold=threshold, follow_window=follow_window
    )
    if not events:
        return pd.DataFrame(
            columns=list(SwingEvent.__annotations__.keys())
        )
    return pd.DataFrame([e.as_dict() for e in events])
