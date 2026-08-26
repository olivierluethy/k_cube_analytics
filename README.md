# k_cube_analytics

A stock-analytics toolkit backed by the free, open-source
[`yfinance`](https://pypi.org/project/yfinance/) data source. Each analytics
module implements one of the project's feature requests (GitHub issues).

## Features (issue → module)

| Issue | Feature | Module | CLI |
|------:|---------|--------|-----|
| #5 | yfinance data source | `data` | — |
| #1 | Relationship between past **price** returns and **dividend** returns (dividend component on its own) | `returns` | `returns` |
| #2 | Compare historical **performance** across tickers | `performance` | `performance` |
| #4 | **Stock vs. stock** ranking by price-only / dividend-only / both | `compare` | `compare` |
| #3 | Detect **large swings**, flag when they happened, judge justified vs. emotional | `events` | `events` |
| #7 | **Emotion vs. facts**: how much of a drop is emotional vs. fundamentally grounded | `emotion` | `emotion` |
| #6 | Map an **event** (e.g. Ukraine war) to the stocks connected to it and rank them | `event_stocks` | `event` |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .          # or: pip install pandas numpy yfinance
```

## Command-line usage

```bash
# #2 – performance comparison over history
python -m k_cube_analytics performance AAPL MSFT --period 5y

# #1 – price-return vs dividend-return relationship
python -m k_cube_analytics returns AAPL KO JNJ

# #4 – rank stocks by price / dividend / both
python -m k_cube_analytics compare AAPL KO --mode dividend

# #3 – detect and assess large swings
python -m k_cube_analytics events TSLA --threshold 0.07

# #7 – emotion-vs-fundamentals of the current decline
python -m k_cube_analytics emotion PYPL

# #6 – which stocks are connected to an event
python -m k_cube_analytics event "Ukraine war" --date 2022-02-24
```

## Library usage

```python
from k_cube_analytics import default_provider, performance, emotion

provider = default_provider()                      # live yfinance provider
res = performance.compare(["AAPL", "MSFT"], provider, period="5y")
print(res["stats"])

report = emotion.analyse("PYPL", provider)
print(report.verdict, report.emotion_index)
```

Every analytics function accepts a `DataProvider`, so you can plug in your own
data source (or a fake one in tests) instead of `yfinance`.

## On the "AI judgment" features (#3, #6, #7)

Issues #3, #6 and #7 ask the system to *judge* whether a move is justified or
emotional. There is no oracle for that, so rather than a black box these modules
use **transparent, rule-based scoring of measurable market behaviour**:

- **#3 / #7** blend signals such as RSI (panic/oversold), how far price is
  stretched below its trend, volume confirmation, gap size, and how much of a
  move later reverses, into an interpretable score in `[0, 1]` plus a verdict.
- **#6** combines a small, extensible *theme library* (event keywords → baskets
  of connected tickers) with a **real price-reaction ranking** measured around
  the event date, so "who profited most" is grounded in data.

Each score exposes its component signals, so the reasoning is auditable and the
weights are easy to tune. The `DataProvider` seam also makes it straightforward
to layer an LLM-based news/sentiment signal on top later.

## Development

```bash
pip install -e ".[dev]"
pytest            # 30 tests, no network required (uses a fake data provider)
```

## Data-source caveat

`yfinance` scrapes Yahoo Finance and can break when Yahoo changes its site
(noted in issue #5). The `DataProvider` abstraction in `data.py` isolates that
risk: the analytics never call `yfinance` directly, so swapping the source
touches one class.
