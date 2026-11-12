"""Tests for the Monte Carlo GBM engine.

The statistical assertions use a fixed seed and generous tolerances (several
standard errors) so they are deterministic but still meaningful.
"""

from __future__ import annotations

import math
import random

import pytest

from mcgbm import GBMParams, simulate_gbm, simulate_terminal_values


def test_shape_of_result() -> None:
    params = GBMParams(steps=12, n_paths=100, seed=1)
    result = simulate_gbm(params)
    assert result.n_paths == 100
    assert result.n_steps == 12
    assert len(result.times) == 13
    assert len(result.paths) == 100
    assert all(len(p) == 13 for p in result.paths)
    assert len(result) == 100


def test_every_path_starts_at_s0() -> None:
    result = simulate_gbm(GBMParams(s0=57.5, steps=5, n_paths=50, seed=2))
    assert all(p[0] == 57.5 for p in result.paths)


def test_prices_stay_positive() -> None:
    result = simulate_gbm(GBMParams(sigma=0.8, steps=252, n_paths=200, seed=3))
    assert all(price > 0.0 for path in result.paths for price in path)


def test_same_seed_is_reproducible() -> None:
    params = GBMParams(steps=20, n_paths=100, seed=123)
    assert simulate_gbm(params).paths == simulate_gbm(params).paths


def test_different_seeds_differ() -> None:
    a = simulate_gbm(GBMParams(steps=20, n_paths=100, seed=1))
    b = simulate_gbm(GBMParams(steps=20, n_paths=100, seed=2))
    assert a.paths != b.paths


def test_antithetic_pairs_are_mirrored() -> None:
    """With antithetic variates, path 2k and 2k+1 use opposite shocks."""
    params = GBMParams(steps=10, n_paths=4, seed=5, antithetic=True)
    result = simulate_gbm(params)
    drift = params.log_drift * params.dt
    vol = params.sigma * math.sqrt(params.dt)
    for k in range(0, 4, 2):
        first, second = result.paths[k], result.paths[k + 1]
        for i in range(1, 11):
            # Recover the shocks and check they are exact opposites.
            z1 = (math.log(first[i] / first[i - 1]) - drift) / vol
            z2 = (math.log(second[i] / second[i - 1]) - drift) / vol
            assert z1 == pytest.approx(-z2, abs=1e-9)


def test_antithetic_reduces_variance_of_the_mean() -> None:
    base = {"steps": 1, "n_paths": 4000, "seed": 11, "sigma": 0.3}
    plain = simulate_gbm(GBMParams(**base, antithetic=False)).terminal_values()
    anti = simulate_gbm(GBMParams(**base, antithetic=True)).terminal_values()
    mean_plain = math.fsum(plain) / len(plain)
    mean_anti = math.fsum(anti) / len(anti)
    target = GBMParams(**base).expected_terminal_value
    # Antithetic sampling should land closer to the analytic expectation.
    assert abs(mean_anti - target) < abs(mean_plain - target)


def test_odd_number_of_paths_with_antithetic() -> None:
    result = simulate_gbm(GBMParams(steps=5, n_paths=7, seed=4, antithetic=True))
    assert result.n_paths == 7


def test_terminal_values_match_last_step_of_paths() -> None:
    """With a single step, both entry points draw the same shocks."""
    params = GBMParams(steps=1, n_paths=200, seed=8)
    result = simulate_gbm(params)
    direct = simulate_terminal_values(params)
    assert result.terminal_values() == pytest.approx(direct)


def test_terminal_values_agree_in_distribution_with_multi_step_paths() -> None:
    """With several steps the draws differ, but the distribution must match."""
    params = GBMParams(s0=100.0, mu=0.05, sigma=0.2, horizon=1.0, steps=50, n_paths=50_000, seed=8)
    from_paths = simulate_gbm(params).terminal_values()
    direct = simulate_terminal_values(params)
    mean_paths = math.fsum(from_paths) / len(from_paths)
    mean_direct = math.fsum(direct) / len(direct)
    target = params.expected_terminal_value
    assert mean_paths == pytest.approx(target, rel=0.01)
    assert mean_direct == pytest.approx(target, rel=0.01)
    assert mean_paths == pytest.approx(mean_direct, rel=0.01)


def test_terminal_values_are_reproducible() -> None:
    params = GBMParams(steps=1, n_paths=500, seed=99)
    assert simulate_terminal_values(params) == simulate_terminal_values(params)


def test_mean_converges_to_analytic_expectation() -> None:
    params = GBMParams(
        s0=100.0, mu=0.08, sigma=0.2, horizon=1.0, steps=1, n_paths=200_000, seed=42
    )
    values = simulate_terminal_values(params)
    sample_mean = math.fsum(values) / len(values)
    # 4 standard errors of the mean is a very safe band.
    sd = params.s0 * math.exp(params.mu) * math.sqrt(math.exp(params.sigma**2) - 1.0)
    stderr = sd / math.sqrt(len(values))
    assert abs(sample_mean - params.expected_terminal_value) < 4.0 * stderr


def test_log_returns_are_normal_with_expected_moments() -> None:
    params = GBMParams(
        s0=100.0, mu=0.05, sigma=0.25, horizon=1.0, steps=1, n_paths=100_000, seed=17
    )
    values = simulate_terminal_values(params)
    logs = [math.log(v / params.s0) for v in values]
    n = len(logs)
    sample_mean = math.fsum(logs) / n
    variance = math.fsum((x - sample_mean) ** 2 for x in logs) / (n - 1)
    assert sample_mean == pytest.approx(params.log_drift * params.horizon, abs=0.005)
    assert math.sqrt(variance) == pytest.approx(params.sigma * math.sqrt(params.horizon), abs=0.005)


def test_zero_volatility_is_deterministic() -> None:
    """With sigma = 0 every path is the deterministic curve S0 * exp(mu * t)."""
    params = GBMParams(s0=100.0, mu=0.05, sigma=0.0, horizon=2.0, steps=10, n_paths=5, seed=1)
    result = simulate_gbm(params)
    for path in result.paths:
        for t, price in zip(result.times, path, strict=True):
            assert price == pytest.approx(100.0 * math.exp(0.05 * t))
    assert result.paths[0] == pytest.approx(result.paths[-1])


def test_mean_and_percentile_paths() -> None:
    result = simulate_gbm(GBMParams(steps=10, n_paths=500, seed=6))
    mean_path = result.mean_path()
    assert len(mean_path) == 11
    assert mean_path[0] == pytest.approx(result.params.s0)
    p05 = result.percentile_path(5.0)
    p95 = result.percentile_path(95.0)
    assert all(lo <= hi for lo, hi in zip(p05, p95, strict=True))
    assert all(lo <= m <= hi for lo, m, hi in zip(p05, mean_path, p95, strict=True))


def test_fan_chart_data_keys() -> None:
    result = simulate_gbm(GBMParams(steps=5, n_paths=50, seed=1))
    data = result.fan_chart_data()
    assert set(data) == {"mean", "p05", "p25", "p50", "p75", "p95"}
    assert all(len(series) == 6 for series in data.values())


def test_to_rows_and_to_dict() -> None:
    result = simulate_gbm(GBMParams(steps=3, n_paths=4, seed=1))
    rows = list(result.to_rows())
    assert len(rows) == 4
    assert len(rows[0]) == 5  # time + 4 paths
    payload = result.to_dict(max_paths=2)
    assert payload["n_paths"] == 4
    assert len(payload["paths"]) == 2  # type: ignore[arg-type]
    assert payload["paths_truncated"] is True


def test_external_rng_is_used() -> None:
    params = GBMParams(steps=5, n_paths=10, seed=None)
    a = simulate_gbm(params, rng=random.Random(1))
    b = simulate_gbm(params, rng=random.Random(1))
    assert a.paths == b.paths
