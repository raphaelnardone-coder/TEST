"""Monte Carlo simulation of geometric Brownian motion (GBM) paths.

A small, dependency-free toolkit to simulate asset price paths under the GBM
model, summarise the resulting distribution and price European / Asian options
against the closed-form Black-Scholes benchmark.

Quick start
-----------
>>> from mcgbm import GBMParams, simulate_gbm, summarize
>>> params = GBMParams(s0=100, mu=0.08, sigma=0.2, horizon=1.0,
    ...                    steps=252, n_paths=10_000, seed=42)
>>> result = simulate_gbm(params)
>>> summary = summarize(result.terminal_values(), s0=params.s0,
    ...                     expected_value=params.expected_terminal_value)
>>> round(summary.median, 2) > 0
True
"""

from __future__ import annotations

from .export import (
    write_fan_chart_svg,
    write_paths_csv,
    write_report_json,
    write_summary_json,
)
from .params import GBMParams
from .pricing import (
    PriceEstimate,
    black_scholes_greeks,
    black_scholes_price,
    mc_asian_price,
    mc_european_price,
    norm_cdf,
    norm_pdf,
)
from .simulation import SimulationResult, simulate_gbm, simulate_terminal_values
from .stats import (
    Summary,
    conditional_value_at_risk,
    mean,
    percentile,
    quantile,
    stdev,
    summarize,
    value_at_risk,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # parameters and simulation
    "GBMParams",
    "SimulationResult",
    "simulate_gbm",
    "simulate_terminal_values",
    # statistics
    "Summary",
    "summarize",
    "mean",
    "stdev",
    "quantile",
    "percentile",
    "value_at_risk",
    "conditional_value_at_risk",
    # pricing
    "PriceEstimate",
    "black_scholes_price",
    "black_scholes_greeks",
    "mc_european_price",
    "mc_asian_price",
    "norm_cdf",
    "norm_pdf",
    # export
    "write_paths_csv",
    "write_summary_json",
    "write_report_json",
    "write_fan_chart_svg",
]
