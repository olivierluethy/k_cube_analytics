"""Map a real-world event to the stocks connected to it.

Resolves Issue #6: for an event (e.g. the war in Ukraine, or the Venezuela
incident) identify which stocks are connected -- who benefits and who is at
risk -- and rank them by how much they actually moved around the event.

Two pieces:

* a small, extensible **theme library** mapping event keywords to baskets of
  tickers that are typically beneficiaries or at risk, and
* a **reaction ranking** that measures each candidate's price move over a
  window starting at the event date, so the "who profited most" answer is
  grounded in real data rather than assertion.

Callers can rely on the built-in library, pass their own candidate baskets, or
both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from .data import DataProvider


@dataclass
class Theme:
    name: str
    keywords: list[str]
    beneficiaries: list[str] = field(default_factory=list)
    at_risk: list[str] = field(default_factory=list)


# Built-in, deliberately small starter library. It is a starting point, not an
# exhaustive market map -- extend via ``register_theme`` or per-call baskets.
EVENT_LIBRARY: list[Theme] = [
    Theme(
        name="armed conflict / war",
        keywords=["war", "krieg", "conflict", "invasion", "ukraine", "military"],
        beneficiaries=["LMT", "RTX", "NOC", "GD", "BA", "HII"],  # defense
        at_risk=["EXPE", "DAL", "CCL", "RCL"],  # travel / discretionary
    ),
    Theme(
        name="oil supply shock",
        keywords=["oil", "opec", "venezuela", "sanction", "embargo", "energy"],
        beneficiaries=["XOM", "CVX", "COP", "OXY", "SLB"],  # oil & gas
        at_risk=["DAL", "UAL", "LUV"],  # airlines (fuel cost)
    ),
    Theme(
        name="pandemic / lockdown",
        keywords=["pandemic", "covid", "virus", "lockdown", "outbreak"],
        beneficiaries=["ZM", "MRNA", "PFE", "NFLX", "PTON"],
        at_risk=["CCL", "RCL", "MAR", "AAL"],
    ),
    Theme(
        name="semiconductor / AI demand",
        keywords=["ai", "chip", "semiconductor", "gpu", "artificial intelligence"],
        beneficiaries=["NVDA", "AMD", "TSM", "ASML", "AVGO"],
        at_risk=[],
    ),
]


def register_theme(theme: Theme) -> None:
    """Add a theme to the built-in library."""
    EVENT_LIBRARY.append(theme)


def match_themes(event_text: str) -> list[Theme]:
    """Return library themes with a whole-word keyword hit in ``event_text``.

    Whole-word matching (not substring) avoids false hits like the keyword
    ``"ai"`` matching inside "ukr**ai**ne".
    """
    text = event_text.lower()
    matched = []
    for theme in EVENT_LIBRARY:
        if any(
            re.search(rf"\b{re.escape(kw)}\b", text) for kw in theme.keywords
        ):
            matched.append(theme)
    return matched


def _reaction(
    symbol: str, provider: DataProvider, event_date: str, window: int
) -> float | None:
    """Price return of ``symbol`` from ``event_date`` over ``window`` sessions."""
    try:
        hist = provider.get_history(symbol, period="5y")
    except Exception:
        return None
    hist = hist.dropna(subset=["Close"])
    hist = hist[hist["Close"] > 0]
    if len(hist) < 2:
        return None

    ts = pd.Timestamp(event_date)
    idx = hist.index
    # Tolerate tz-aware indices from yfinance.
    try:
        pos = idx.searchsorted(ts)
    except TypeError:
        idx = idx.tz_localize(None)
        hist = hist.copy()
        hist.index = idx
        pos = idx.searchsorted(ts)
    if pos >= len(hist):
        return None
    end = min(pos + window, len(hist) - 1)
    start_price = hist["Close"].iloc[pos]
    end_price = hist["Close"].iloc[end]
    if start_price <= 0:
        return None
    return float(end_price / start_price - 1.0)


def analyse_event(
    event_text: str,
    provider: DataProvider,
    event_date: str | None = None,
    window: int = 20,
    extra_candidates: list[str] | None = None,
) -> dict:
    """Identify and rank stocks connected to ``event_text``.

    Parameters
    ----------
    event_text:
        Free-text description; matched against the theme library's keywords.
    event_date:
        Date the event began (``YYYY-MM-DD``). Required to rank by real price
        reaction; without it only the connected baskets are returned.
    window:
        Sessions after the event over which to measure the reaction.
    extra_candidates:
        Additional tickers to score alongside the library matches.
    """
    themes = match_themes(event_text)
    beneficiaries: list[str] = []
    at_risk: list[str] = []
    for theme in themes:
        beneficiaries.extend(theme.beneficiaries)
        at_risk.extend(theme.at_risk)
    if extra_candidates:
        beneficiaries.extend(extra_candidates)

    # De-duplicate, preserve order.
    beneficiaries = list(dict.fromkeys(beneficiaries))
    at_risk = list(dict.fromkeys(at_risk))

    result = {
        "event": event_text,
        "event_date": event_date,
        "matched_themes": [t.name for t in themes],
        "connected_beneficiaries": beneficiaries,
        "connected_at_risk": at_risk,
        "ranking": pd.DataFrame(),
    }

    if not event_date:
        return result

    rows = []
    for symbol in beneficiaries + at_risk:
        r = _reaction(symbol, provider, event_date, window)
        if r is None:
            continue
        rows.append(
            {
                "symbol": symbol,
                "side": "beneficiary" if symbol in beneficiaries else "at_risk",
                "reaction": r,
            }
        )
    if rows:
        table = pd.DataFrame(rows).sort_values("reaction", ascending=False)
        table["rank"] = range(1, len(table) + 1)
        result["ranking"] = table.set_index("symbol")
        result["top_beneficiary"] = table.iloc[0]["symbol"]

    return result
