# mcgbm — Monte Carlo GBM simulations

A small, **dependency-free** Python toolkit to simulate asset price paths with
geometric Brownian motion (GBM), summarise the resulting distribution, and price
European / Asian options against the closed-form Black-Scholes benchmark.

Only the Python standard library is used (Python 3.10+), so it runs anywhere —
no NumPy, SciPy or matplotlib required.

## The model

GBM assumes the price `S_t` follows

```
dS_t = (mu - q) * S_t dt + sigma * S_t dW_t
```

whose exact solution over a step `dt` is

```
S_{t+dt} = S_t * exp((mu - q - sigma^2 / 2) * dt + sigma * sqrt(dt) * Z),   Z ~ N(0, 1)
```

Because this is the **exact** solution of the SDE, the simulation carries no
time-discretisation bias: the only error is Monte Carlo sampling error, which
shrinks like `1 / sqrt(n_paths)` and is reported explicitly as a standard error
and a 95 % confidence interval.

| Symbol | Meaning | Field |
| --- | --- | --- |
| `S_0` | initial spot price | `s0` |
| `mu` | annualised expected return | `mu` |
| `sigma` | annualised volatility | `sigma` |
| `q` | continuous dividend yield | `dividend_yield` |
| `T` | horizon in years | `horizon` |
| — | time steps per path | `steps` |
| — | number of paths | `n_paths` |

## Install

```bash
pip install -e .          # optional: installs the `mcgbm` command
```

Or simply run from the repository root — the package is importable as is.

## Quick start

```python
from mcgbm import GBMParams, simulate_gbm, summarize

params = GBMParams(s0=100, mu=0.08, sigma=0.2, horizon=1.0,
                   steps=252, n_paths=20_000, seed=42)

result  = simulate_gbm(params)
summary = summarize(result.terminal_values(),
                    s0=params.s0,
                    expected_value=params.expected_terminal_value)

print(summary.median, summary.p05, summary.p95)
print(summary.var_95, summary.cvar_95)
```

The same parameters and seed always produce the same paths, so results are
reproducible.

## Command line

Simulate paths and export CSV / JSON / SVG:

```bash
python -m mcgbm simulate --s0 100 --mu 0.08 --sigma 0.2 --horizon 1 \
    --steps 252 --paths 20000 --seed 42 --outdir output
```

Price an option by Monte Carlo and compare with Black-Scholes:

```bash
python -m mcgbm price --s0 100 --strike 105 --rate 0.03 --sigma 0.2 \
    --maturity 1 --kind call --paths 200000 --seed 7
```

Add `--json` to either command for machine-readable output, and
`--no-antithetic` to disable antithetic variates.

## What you get

| Module | Contents |
| --- | --- |
| `mcgbm.params` | `GBMParams` — validated, immutable simulation parameters |
| `mcgbm.simulation` | `simulate_gbm`, `simulate_terminal_values`, `SimulationResult` |
| `mcgbm.stats` | `summarize`, `Summary`, `quantile`, `percentile`, `value_at_risk`, `conditional_value_at_risk` |
| `mcgbm.pricing` | `black_scholes_price`, `black_scholes_greeks`, `mc_european_price`, `mc_asian_price`, `PriceEstimate` |
| `mcgbm.export` | `write_paths_csv`, `write_summary_json`, `write_report_json`, `write_fan_chart_svg` |
| `mcgbm.cli` | `python -m mcgbm simulate` / `python -m mcgbm price` |

### Risk measures

`value_at_risk` and `conditional_value_at_risk` report a **loss** measured
against a reference price (normally `S_0`), floored at zero:

* `VaR 95 %` — the loss not exceeded in 95 % of the simulated scenarios;
* `CVaR 95 %` (expected shortfall) — the average loss in the worst 5 % tail.

### Exports

* `paths.csv` — one row per observation date, one column per path;
* `summary.json` — parameters, summary statistics and fan-chart quantiles;
* `fan_chart.svg` — median path, 5–95 % and 25–75 % bands, and the analytic
  expectation `S_0 * exp((mu - q) * T)` as a dashed line.

## Example script

```bash
python examples/run_example.py
```

Runs a full simulation, prints a report, prices a European call and writes the
three output files into `output/`.

## Tests

```bash
python -m pytest          # 103 tests
ruff check mcgbm tests    # lint
mypy mcgbm                # type check
```

The suite checks the mathematics rather than just the plumbing: the analytic
expectation `E[S_T] = S_0 * exp((mu - q) * T)`, the moments of the simulated log
returns, put-call parity, Black-Scholes Greeks against finite differences, the
Monte Carlo price falling inside its own confidence interval around the analytic
price, antithetic variance reduction, reproducibility, and the validity of the
exported files.

## Notes and limitations

* GBM assumes constant drift and volatility and log-normal prices: it cannot
  reproduce fat tails, volatility clustering or jumps.
* `simulate_terminal_values` draws one normal per path and is much cheaper than
  `simulate_gbm` when the whole path is not needed; the two therefore consume
  different random sequences (they agree exactly when `steps=1`).
* The Asian option price uses a discretely monitored arithmetic average, for
  which no closed form exists, so no analytic benchmark is reported.
* A memory guard rejects simulations above 100 million stored prices.

## License

MIT
