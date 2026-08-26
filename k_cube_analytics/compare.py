"""Stock-vs-stock ranking by return philosophy.

Resolves Issue #4: decide between stocks based on *only* price returns, *only*
dividends, or *both* combined. The caller picks the lens (``mode``) and gets a
ranked verdict on the same set of names.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from .data import DataProvider
from .returns import breakdown

Mode = Literal["price", "dividend", "both"]

_SCORE_COLUMN = {
    "price": "price_cagr",
    "dividend": "dividend_cagr",
    "both": "total_cagr",
}


def rank(
    symbols: list[str],
    provider: DataProvider,
    mode: Mode = "both",
    period: str = "5y",
) -> dict:
    """Rank ``symbols`` under the chosen decision ``mode``.

    * ``"price"``    -- judge only on price appreciation (CAGR).
    * ``"dividend"`` -- judge only on the dividend component (CAGR).
    * ``"both"``     -- judge on combined total-return CAGR.

    Returns the ranked table, the winning symbol, and the score column used.
    """
    if mode not in _SCORE_COLUMN:
        raise ValueError(
            f"mode must be one of {list(_SCORE_COLUMN)}, got {mode!r}"
        )

    rows = []
    for symbol in symbols:
        b = breakdown(symbol, provider, period=period)
        d = b.as_dict()
        d["total_cagr"] = (
            (1.0 + b.price_cagr) * (1.0 + b.dividend_cagr) - 1.0
        )
        rows.append(d)

    table = pd.DataFrame(rows).set_index("symbol")
    score_col = _SCORE_COLUMN[mode]
    table = table.sort_values(score_col, ascending=False)
    table["rank"] = range(1, len(table) + 1)

    return {
        "mode": mode,
        "period": period,
        "score_column": score_col,
        "ranking": table,
        "winner": table.index[0] if len(table) else None,
    }
