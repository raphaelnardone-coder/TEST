"""End-to-end example: simulate GBM paths, summarise them and export the results.

Run it from the repository root::

    python examples/run_example.py

It writes ``paths.csv``, ``summary.json`` and ``fan_chart.svg`` into the
``output/`` directory (created if missing) and prints a short report.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when the script is run directly from a checkout
# (``python examples/run_example.py``), without installing it first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcgbm import (  # noqa: E402
    GBMParams,
    black_scholes_price,
    mc_european_price,
    simulate_gbm,
    summarize,
    write_fan_chart_svg,
    write_paths_csv,
    write_report_json,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def main() -> int:
    # 1. Describe the asset: 100 EUR spot, 8 % expected return, 20 % volatility,
    #    one year horizon sampled daily (252 trading days).
    params = GBMParams(
        s0=100.0,
        mu=0.08,
        sigma=0.2,
        horizon=1.0,
        steps=252,
        n_paths=20_000,
        seed=42,
        antithetic=True,
    )
    print(params.describe())

    # 2. Simulate the paths.
    result = simulate_gbm(params)

    # 3. Summarise the terminal distribution.
    summary = summarize(
        result.terminal_values(),
        s0=params.s0,
        expected_value=params.expected_terminal_value,
    )
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
        f"MC mean = {summary.mean:,.4f} | std error = {summary.stderr:,.4f}"
    )

    # 4. Price a European call on the same asset and compare with Black-Scholes.
    estimate = mc_european_price(
        s0=params.s0,
        strike=105.0,
        rate=0.03,
        sigma=params.sigma,
        maturity=params.horizon,
        kind="call",
        n_paths=200_000,
        seed=7,
    )
    analytic = black_scholes_price(params.s0, 105.0, 0.03, params.sigma, params.horizon)
    print()
    print(f"European call (K=105, r=3 %): MC {estimate.price:,.4f} "
          f"+/- {1.96 * estimate.stderr:,.4f} (95 %) | Black-Scholes {analytic:,.4f}")

    # 5. Export the results.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = write_paths_csv(result, OUTPUT_DIR / "paths.csv", max_paths=100)
    json_path = write_report_json(result, summary, OUTPUT_DIR / "summary.json")
    svg_path = write_fan_chart_svg(result, OUTPUT_DIR / "fan_chart.svg")
    print()
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
