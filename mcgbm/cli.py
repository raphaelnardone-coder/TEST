"""Command line interface for the :mod:`mcgbm` package.

Two sub-commands are available::

    python -m mcgbm simulate --s0 100 --mu 0.08 --sigma 0.2 --horizon 1 \
        --steps 252 --paths 10000 --seed 42 --outdir output

    python -m mcgbm price --s0 100 --strike 105 --rate 0.03 --sigma 0.2 \
        --maturity 1 --kind call --paths 200000 --seed 7
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .export import write_fan_chart_svg, write_paths_csv, write_report_json
from .params import GBMParams
from .pricing import black_scholes_greeks, mc_asian_price, mc_european_price
from .simulation import simulate_gbm
from .stats import summarize

__all__ = ["build_parser", "main"]


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--s0", type=float, default=100.0, help="initial spot price (default: 100)")
    parser.add_argument("--sigma", type=float, default=0.2, help="annual volatility (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument(
        "--paths", type=int, default=10_000, help="number of simulated paths (default: 10000)"
    )
    parser.add_argument(
        "--no-antithetic",
        action="store_true",
        help="disable antithetic variates (they are on by default)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``mcgbm`` command line tool."""
    parser = argparse.ArgumentParser(
        prog="mcgbm",
        description="Monte Carlo simulation of geometric Brownian motion (GBM) paths.",
    )
    parser.add_argument("--version", action="version", version="mcgbm 1.0.0")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---------------------------------------------------------------- simulate
    sim = sub.add_parser("simulate", help="simulate GBM paths and export statistics")
    _add_common(sim)
    sim.add_argument(
        "--mu", type=float, default=0.08, help="annual expected return (default: 0.08)"
    )
    sim.add_argument(
        "--q", type=float, default=0.0, help="continuous dividend yield (default: 0)"
    )
    sim.add_argument("--horizon", type=float, default=1.0, help="horizon in years (default: 1.0)")
    sim.add_argument("--steps", type=int, default=252, help="time steps per path (default: 252)")
    sim.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="directory for the CSV/JSON/SVG outputs (default: no files written)",
    )
    sim.add_argument(
        "--csv-paths",
        type=int,
        default=50,
        help="number of path columns written to the CSV (default: 50)",
    )
    sim.add_argument("--json", action="store_true", help="print the summary as JSON")

    # ------------------------------------------------------------------- price
    price = sub.add_parser("price", help="price a European or Asian option by Monte Carlo")
    _add_common(price)
    price.add_argument("--strike", type=float, required=True, help="option strike")
    price.add_argument("--rate", type=float, default=0.03, help="risk-free rate (default: 0.03)")
    price.add_argument(
        "--q", type=float, default=0.0, help="continuous dividend yield (default: 0)"
    )
    price.add_argument(
        "--maturity", type=float, default=1.0, help="maturity in years (default: 1.0)"
    )
    price.add_argument("--kind", choices=("call", "put"), default="call", help="option type")
    price.add_argument(
        "--style",
        choices=("european", "asian"),
        default="european",
        help="exercise style (default: european)",
    )
    price.add_argument(
        "--steps", type=int, default=52, help="monitoring dates for Asian options (default: 52)"
    )
    price.add_argument("--json", action="store_true", help="print the result as JSON")

    return parser


def _run_simulate(args: argparse.Namespace) -> int:
    params = GBMParams(
        s0=args.s0,
        mu=args.mu,
        sigma=args.sigma,
        horizon=args.horizon,
        steps=args.steps,
        n_paths=args.paths,
        seed=args.seed,
        antithetic=not args.no_antithetic,
        dividend_yield=args.q,
    )
    result = simulate_gbm(params)
    summary = summarize(
        result.terminal_values(),
        s0=params.s0,
        expected_value=params.expected_terminal_value,
    )

    if args.json:
        print(json.dumps({"params": params.to_dict(), "summary": summary.to_dict()}, indent=2))
    else:
        print(params.describe())
        print()
        print(f"{'statistic':<18}{'value':>14}")
        print("-" * 32)
        for label, value in summary.as_rows():
            if label == "paths":
                print(f"{label:<18}{int(value):>14,}")
            elif label in ("expected return", "prob. of loss"):
                print(f"{label:<18}{value:>13.2%}")
            else:
                print(f"{label:<18}{value:>14,.4f}")
        print()
        print(
            f"analytic E[S_T] = {params.expected_terminal_value:,.4f} | "
            f"MC mean = {summary.mean:,.4f} | "
            f"std error = {summary.stderr:,.4f}"
        )

    if args.outdir is not None:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        csv_path = write_paths_csv(result, outdir / "paths.csv", max_paths=args.csv_paths)
        json_path = write_report_json(result, summary, outdir / "summary.json")
        svg_path = write_fan_chart_svg(result, outdir / "fan_chart.svg")
        print()
        print(f"wrote {csv_path}")
        print(f"wrote {json_path}")
        print(f"wrote {svg_path}")
    return 0


def _run_price(args: argparse.Namespace) -> int:
    if args.style == "asian":
        estimate = mc_asian_price(
            args.s0,
            args.strike,
            args.rate,
            args.sigma,
            args.maturity,
            kind=args.kind,
            dividend_yield=args.q,
            steps=args.steps,
            n_paths=args.paths,
            seed=args.seed,
            antithetic=not args.no_antithetic,
        )
    else:
        estimate = mc_european_price(
            args.s0,
            args.strike,
            args.rate,
            args.sigma,
            args.maturity,
            kind=args.kind,
            dividend_yield=args.q,
            n_paths=args.paths,
            seed=args.seed,
            antithetic=not args.no_antithetic,
        )

    greeks = black_scholes_greeks(
        args.s0,
        args.strike,
        args.rate,
        args.sigma,
        args.maturity,
        kind=args.kind,
        dividend_yield=args.q,
    )

    if args.json:
        print(json.dumps({"estimate": estimate.to_dict(), "greeks": greeks}, indent=2))
        return 0

    label = f"{args.style} {args.kind}"
    print(
        f"{label}: S0={args.s0:g} K={args.strike:g} r={args.rate:g} "
        f"sigma={args.sigma:g} T={args.maturity:g}y paths={args.paths:,}"
    )
    print()
    print(f"Monte Carlo price : {estimate.price:,.4f}")
    print(f"standard error    : {estimate.stderr:,.4f}")
    print(f"95% confidence    : [{estimate.ci_low:,.4f}, {estimate.ci_high:,.4f}]")
    if estimate.analytic is not None:
        print(f"Black-Scholes     : {estimate.analytic:,.4f}")
        print(f"MC - analytic     : {estimate.error_vs_analytic:+,.4f}")
    else:
        print("Black-Scholes     : n/a (no closed form for arithmetic-average Asian)")
    print()
    print("Black-Scholes Greeks (analytic):")
    for name, value in greeks.items():
        print(f"  {name:<6}{value:>12.6f}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the command line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "simulate":
            return _run_simulate(args)
        if args.command == "price":
            return _run_price(args)
    except ValueError as exc:  # invalid parameters
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
