# MEXTYL.md — project memory for AI coding agents

## What this is

`mcgbm` is a small, **dependency-free** Python 3.10+ toolkit for quantitative
finance: it simulates asset price paths with geometric Brownian motion (GBM)
using the exact solution of the SDE, summarises the terminal distribution
(quantiles, VaR/CVaR), prices European and Asian options by Monte Carlo against
a closed-form Black-Scholes benchmark, and exports CSV / JSON / SVG. Only the
standard library is used — no NumPy, SciPy or matplotlib. Full user
documentation: @README.md

## Commands (verified in this sandbox, Python 3.10.12)

```bash
# Install (optional: the package is importable from the repo root as is)
pip install -e . --no-build-isolation   # plain `pip install -e .` needs network
                                        # (build isolation pulls setuptools>=68)
pip install -e ".[dev]"                 # pytest>=7, ruff>=0.4, mypy>=1.8 (not run: no network)

# Tests — 103 tests, ~5 s
python -m pytest                                  # whole suite
python -m pytest tests/test_stats.py              # one file
python -m pytest tests/test_pricing.py::test_put_call_parity   # one test

# Lint / type-check
ruff check mcgbm tests     # clean; do NOT use `ruff check .` (see pitfalls)
mypy mcgbm                 # clean; bare `mypy` also checks examples/ (files = ["mcgbm", "examples"])

# Run
python -m mcgbm simulate --steps 252 --paths 20000 --seed 42 --outdir output
python -m mcgbm price --strike 105 --rate 0.03 --paths 200000 --seed 7 --json
python examples/run_example.py     # end-to-end demo, writes gitignored output/
```

There is no build step and no CI configuration in the repository.
`ruff format` is **not** enforced: `ruff format --check mcgbm tests` reports
5 files that would be reformatted, so do not reformat the tree wholesale.

## Architecture

| Path | Role |
| --- | --- |
| `mcgbm/params.py` | `GBMParams` — frozen dataclass, all validation in `__post_init__`, derived properties (`dt`, `times`, `log_drift`, `expected_terminal_value`), `MAX_CELLS` memory guard |
| `mcgbm/simulation.py` | `simulate_gbm`, `simulate_terminal_values`, `SimulationResult` (frozen; `terminal_values`, `mean_path`, `percentile_path`, `fan_chart_data`, `to_rows`, `to_dict`) |
| `mcgbm/stats.py` | `mean`, `stdev`, `quantile`, `percentile`, `value_at_risk`, `conditional_value_at_risk`, `Summary`, `summarize` |
| `mcgbm/pricing.py` | `norm_cdf`/`norm_pdf`, `black_scholes_price`, `black_scholes_greeks`, `mc_european_price`, `mc_asian_price`, `PriceEstimate` |
| `mcgbm/export.py` | `write_paths_csv`, `write_summary_json`, `write_report_json`, `write_fan_chart_svg` (SVG built as plain strings) |
| `mcgbm/cli.py` | argparse CLI: `build_parser()`, `main(argv)`; `__main__.py` wires `python -m mcgbm` |
| `mcgbm/__init__.py` | public API re-exports + `__all__` + `__version__` |
| `tests/` | one module per source module (`test_params`, `test_simulation`, `test_stats`, `test_pricing`, `test_export`, `test_cli`) |
| `examples/run_example.py` | runnable demo; inserts the repo root on `sys.path` so it works uninstalled |

## Conventions

- Every module starts with `from __future__ import annotations` and declares `__all__`.
- Value objects are `@dataclass(frozen=True)`; mutation goes through `replace()`
  (`GBMParams.with_`, `GBMParams.risk_neutral`).
- Invalid input raises `ValueError` with an f-string that echoes the offending
  value using `{value!r}` (e.g. `f"sigma must be finite and >= 0, got {sigma!r}"`).
  The CLI catches `ValueError` in `main`, prints `error: ...` to stderr and
  returns exit code 2.
- Optional flags are keyword-only (after `*`): `rng=`, `kind=`, `dividend_yield=`,
  `sample=`, `reference=`, `max_paths=`.
- Private helpers are `_`-prefixed (`_build_path`, `_estimate`, `_validate_option`,
  `_nice_ticks`, `_escape`, `_ensure_parent`, `_add_common`, `_run_*`).
- Numerics use `math.fsum` for summation; `quantile` follows the NumPy "linear"
  definition (Hyndman & Fan type 7).
- Style: line length 100, ruff rules `E,F,W,I,UP,B,C4,SIM,RET,ARG,PTH`
  (`ARG` ignored under `tests/`). Docstrings are numpydoc-style with
  `Parameters` / `Returns` / `Examples` sections.
- Tests are plain functions annotated `-> None`, using `pytest.approx`,
  `@pytest.mark.parametrize`, `tmp_path` and `capsys`. Statistical assertions
  always fix a seed and use tolerances of several standard errors.

## Pitfalls

- `test_demo.py` at the repository root is **not valid Python** (a placeholder).
  `pyproject.toml` sets `testpaths = ["tests"]` so pytest skips it. Never run
  `pytest test_demo.py` (collection error) or `ruff check .` (3 errors from that
  file); do not add it to the test paths.
- `filterwarnings = ["error"]` in the pytest config: any warning fails the suite.
- `dependencies = []` is a hard constraint. Do not add third-party packages
  (NumPy/SciPy/matplotlib); the SVG chart is hand-written markup for this reason.
- `simulate_gbm` and `simulate_terminal_values` consume different random
  sequences (one normal per step vs one per path); they agree only when
  `steps=1`. Do not assert equality between them for multi-step runs.
- `GBMParams` rejects `n_paths * (steps + 1) > MAX_CELLS` (100 million) and
  rejects `bool` where an `int` is expected.
- `seed=None` means non-reproducible; the CLI defaults to `--seed 42`.
- `output/` and `*.svg` are gitignored — generated artefacts must not be committed.
- mypy only checks `mcgbm` and `examples`; `tests/` is not type-checked.
