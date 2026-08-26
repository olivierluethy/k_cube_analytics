"""Command-line interface for the analytics toolkit.

Each subcommand maps to one issue's feature. Data comes from the live
``yfinance`` provider by default::

    python -m k_cube_analytics performance AAPL MSFT --period 5y
    python -m k_cube_analytics returns AAPL KO JNJ
    python -m k_cube_analytics compare AAPL MSFT --mode dividend
    python -m k_cube_analytics events TSLA --threshold 0.07
    python -m k_cube_analytics emotion PYPL
    python -m k_cube_analytics event "Ukraine war" --date 2022-02-24
"""

from __future__ import annotations

import argparse

import pandas as pd

from . import compare, emotion, event_stocks, events, performance, returns
from .data import default_provider


def _print_df(df: pd.DataFrame) -> None:
    if df is None or len(df) == 0:
        print("(no rows)")
        return
    with pd.option_context(
        "display.max_rows", None,
        "display.width", 200,
        "display.float_format", lambda x: f"{x:,.4f}",
    ):
        print(df)


def _cmd_performance(args, provider) -> None:
    res = performance.compare(args.symbols, provider, period=args.period)
    print(f"Performance over {res['period']}  (best: {res['best']}, "
          f"worst: {res['worst']})")
    _print_df(res["stats"])


def _cmd_returns(args, provider) -> None:
    res = returns.relation(args.symbols, provider, period=args.period)
    corr = res["price_vs_dividend_cagr_corr"]
    print(f"Price vs dividend returns over {res['period']}")
    _print_df(res["breakdowns"])
    print(f"\nPrice-CAGR vs Dividend-CAGR correlation: {corr:.4f}"
          if corr == corr else
          "\nPrice-CAGR vs Dividend-CAGR correlation: n/a")


def _cmd_compare(args, provider) -> None:
    res = compare.rank(args.symbols, provider, mode=args.mode, period=args.period)
    print(f"Ranking by mode={res['mode']} (score={res['score_column']}), "
          f"winner: {res['winner']}")
    _print_df(res["ranking"])


def _cmd_events(args, provider) -> None:
    df = events.detect(
        args.symbol, provider, period=args.period,
        threshold=args.threshold, follow_window=args.follow,
    )
    print(f"Swing events for {args.symbol} (|move| >= {args.threshold:.0%})")
    _print_df(df)


def _cmd_emotion(args, provider) -> None:
    report = emotion.analyse(args.symbol, provider, period=args.period)
    print(f"Emotion analysis for {args.symbol}")
    for k, v in report.as_dict().items():
        print(f"  {k}: {v}")


def _cmd_event(args, provider) -> None:
    res = event_stocks.analyse_event(
        args.text, provider, event_date=args.date,
        window=args.window, extra_candidates=args.candidates,
    )
    print(f"Event: {res['event']}")
    print(f"Matched themes: {', '.join(res['matched_themes']) or '(none)'}")
    print(f"Connected beneficiaries: {', '.join(res['connected_beneficiaries'])}")
    print(f"Connected at-risk:       {', '.join(res['connected_at_risk'])}")
    if len(res["ranking"]):
        print("\nReaction ranking:")
        _print_df(res["ranking"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="k_cube_analytics",
        description="Stock analytics toolkit (yfinance-backed).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("performance", help="compare historical performance (#2)")
    p.add_argument("symbols", nargs="+")
    p.add_argument("--period", default="5y")
    p.set_defaults(func=_cmd_performance)

    p = sub.add_parser("returns", help="price- vs dividend-return relation (#1)")
    p.add_argument("symbols", nargs="+")
    p.add_argument("--period", default="5y")
    p.set_defaults(func=_cmd_returns)

    p = sub.add_parser("compare", help="rank stocks by price/dividend/both (#4)")
    p.add_argument("symbols", nargs="+")
    p.add_argument("--mode", choices=["price", "dividend", "both"], default="both")
    p.add_argument("--period", default="5y")
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser("events", help="detect & assess large swings (#3)")
    p.add_argument("symbol")
    p.add_argument("--period", default="2y")
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--follow", type=int, default=5)
    p.set_defaults(func=_cmd_events)

    p = sub.add_parser("emotion", help="emotion-vs-fundamentals of a drop (#7)")
    p.add_argument("symbol")
    p.add_argument("--period", default="2y")
    p.set_defaults(func=_cmd_emotion)

    p = sub.add_parser("event", help="event -> connected stocks (#6)")
    p.add_argument("text", help="event description, e.g. 'Ukraine war'")
    p.add_argument("--date", default=None, help="event date YYYY-MM-DD")
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--candidates", nargs="*", default=None)
    p.set_defaults(func=_cmd_event)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    provider = default_provider()
    args.func(args, provider)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
