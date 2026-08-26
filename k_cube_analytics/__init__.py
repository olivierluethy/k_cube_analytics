"""k_cube_analytics -- stock analytics toolkit.

Each analytics module maps to a GitHub issue:

* :mod:`k_cube_analytics.data`         -- yfinance data source            (#5)
* :mod:`k_cube_analytics.returns`      -- price- vs dividend-return relation (#1)
* :mod:`k_cube_analytics.performance`  -- historical performance comparison (#2)
* :mod:`k_cube_analytics.compare`      -- stocks vs stocks by price/div/both (#4)
* :mod:`k_cube_analytics.events`       -- swing detection & emotion verdict  (#3)
* :mod:`k_cube_analytics.emotion`      -- emotion-vs-fundamentals of a drop   (#7)
* :mod:`k_cube_analytics.event_stocks` -- event -> connected stocks           (#6)
"""

from __future__ import annotations

from . import (
    compare,
    data,
    emotion,
    event_stocks,
    events,
    performance,
    returns,
)
from .data import DataProvider, YFinanceProvider, default_provider

__version__ = "0.1.0"

__all__ = [
    "compare",
    "data",
    "emotion",
    "event_stocks",
    "events",
    "performance",
    "returns",
    "DataProvider",
    "YFinanceProvider",
    "default_provider",
]
