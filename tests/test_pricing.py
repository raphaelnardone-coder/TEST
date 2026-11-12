"""Tests for the pricing module: Black-Scholes benchmarks and Monte Carlo prices."""

from __future__ import annotations

import math

import pytest

from mcgbm import (
    black_scholes_greeks,
    black_scholes_price,
    mc_asian_price,
    mc_european_price,
    norm_cdf,
    norm_pdf,
)


def test_norm_cdf_and_pdf_reference_values() -> None:
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
    assert norm_cdf(-1.96) == pytest.approx(0.025, abs=1e-3)
    assert norm_pdf(0.0) == pytest.approx(1.0 / math.sqrt(2.0 * math.pi))
    assert norm_cdf(0.3) + norm_cdf(-0.3) == pytest.approx(1.0)


def test_black_scholes_known_call_price() -> None:
    """Textbook case: S=K=100, r=5 %, sigma=20 %, T=1 -> 10.4506."""
    assert black_scholes_price(100, 100, 0.05, 0.2, 1.0) == pytest.approx(10.4506, abs=1e-4)


def test_put_call_parity() -> None:
    s0, k, r, sigma, t, q = 100.0, 95.0, 0.03, 0.25, 1.5, 0.01
    call = black_scholes_price(s0, k, r, sigma, t, kind="call", dividend_yield=q)
    put = black_scholes_price(s0, k, r, sigma, t, kind="put", dividend_yield=q)
    lhs = call - put
    rhs = s0 * math.exp(-q * t) - k * math.exp(-r * t)
    assert lhs == pytest.approx(rhs, abs=1e-9)


def test_deep_in_the_money_call_approaches_intrinsic() -> None:
    price = black_scholes_price(200.0, 100.0, 0.05, 0.2, 1.0)
    assert price == pytest.approx(200.0 - 100.0 * math.exp(-0.05), abs=0.5)


def test_zero_volatility_price_is_discounted_intrinsic() -> None:
    call = black_scholes_price(100.0, 90.0, 0.05, 0.0, 1.0)
    assert call == pytest.approx(max(0.0, 100.0 - 90.0 * math.exp(-0.05)))
    put = black_scholes_price(100.0, 120.0, 0.05, 0.0, 1.0, kind="put")
    assert put == pytest.approx(120.0 * math.exp(-0.05) - 100.0)


def test_call_price_increases_with_volatility() -> None:
    prices = [black_scholes_price(100, 100, 0.05, s, 1.0) for s in (0.1, 0.2, 0.4)]
    assert prices[0] < prices[1] < prices[2]


@pytest.mark.parametrize("kind", ["straddle", "", "CALLS"])
def test_invalid_option_kind_raises(kind: str) -> None:
    with pytest.raises(ValueError):
        black_scholes_price(100, 100, 0.05, 0.2, 1.0, kind=kind)


def test_kind_is_case_insensitive() -> None:
    assert black_scholes_price(100, 100, 0.05, 0.2, 1.0, kind="CALL") == pytest.approx(
        black_scholes_price(100, 100, 0.05, 0.2, 1.0, kind="call")
    )


@pytest.mark.parametrize(
    "kwargs",
    [{"strike": 0.0}, {"strike": -5.0}, {"maturity": 0.0}, {"maturity": -1.0}],
)
def test_invalid_option_inputs_raise(kwargs: dict[str, float]) -> None:
    base = {"s0": 100.0, "strike": 100.0, "rate": 0.05, "sigma": 0.2, "maturity": 1.0}
    base.update(kwargs)
    with pytest.raises(ValueError):
        black_scholes_price(**base)  # type: ignore[arg-type]


def test_greeks_signs_and_parity() -> None:
    call = black_scholes_greeks(100, 100, 0.05, 0.2, 1.0, kind="call")
    put = black_scholes_greeks(100, 100, 0.05, 0.2, 1.0, kind="put")
    assert 0.0 < call["delta"] < 1.0
    assert -1.0 < put["delta"] < 0.0
    assert call["gamma"] == pytest.approx(put["gamma"])
    assert call["vega"] == pytest.approx(put["vega"])
    assert call["vega"] > 0.0
    assert call["theta"] < 0.0
    # Put-call delta parity: delta_call - delta_put = exp(-q T)
    assert call["delta"] - put["delta"] == pytest.approx(1.0)


def test_greeks_match_finite_differences() -> None:
    s0, k, r, sigma, t = 100.0, 105.0, 0.04, 0.22, 0.75
    greeks = black_scholes_greeks(s0, k, r, sigma, t)
    h = 1e-4
    delta_fd = (
        black_scholes_price(s0 + h, k, r, sigma, t) - black_scholes_price(s0 - h, k, r, sigma, t)
    ) / (2 * h)
    vega_fd = (
        black_scholes_price(s0, k, r, sigma + h, t) - black_scholes_price(s0, k, r, sigma - h, t)
    ) / (2 * h)
    assert greeks["delta"] == pytest.approx(delta_fd, abs=1e-5)
    assert greeks["vega"] == pytest.approx(vega_fd, abs=1e-3)


def test_mc_european_price_is_close_to_black_scholes() -> None:
    estimate = mc_european_price(100, 100, 0.05, 0.2, 1.0, n_paths=200_000, seed=7)
    assert estimate.analytic is not None
    assert abs(estimate.error_vs_analytic or 0.0) < 4.0 * estimate.stderr
    assert estimate.ci_low < estimate.analytic < estimate.ci_high


def test_mc_european_put_matches_black_scholes() -> None:
    estimate = mc_european_price(100, 110, 0.03, 0.25, 0.5, kind="put", n_paths=200_000, seed=11)
    assert abs(estimate.error_vs_analytic or 0.0) < 4.0 * estimate.stderr


def test_mc_price_is_reproducible() -> None:
    a = mc_european_price(100, 100, 0.05, 0.2, 1.0, n_paths=5_000, seed=3)
    b = mc_european_price(100, 100, 0.05, 0.2, 1.0, n_paths=5_000, seed=3)
    assert a.price == b.price


def test_mc_confidence_interval_shrinks_with_more_paths() -> None:
    small = mc_european_price(100, 100, 0.05, 0.2, 1.0, n_paths=2_000, seed=5)
    large = mc_european_price(100, 100, 0.05, 0.2, 1.0, n_paths=200_000, seed=5)
    assert large.ci_width < small.ci_width


def test_price_estimate_fields() -> None:
    estimate = mc_european_price(100, 100, 0.05, 0.2, 1.0, n_paths=1_000, seed=1)
    assert estimate.n_paths == 1_000
    assert estimate.ci_low <= estimate.price <= estimate.ci_high
    payload = estimate.to_dict()
    assert payload["n_paths"] == 1_000
    assert "error_vs_analytic" in payload


def test_asian_price_is_below_european_for_a_call() -> None:
    """Averaging reduces volatility, so an arithmetic Asian call is cheaper."""
    asian = mc_asian_price(100, 100, 0.05, 0.2, 1.0, steps=52, n_paths=20_000, seed=9)
    european = mc_european_price(100, 100, 0.05, 0.2, 1.0, n_paths=20_000, seed=9)
    assert asian.price < european.price
    assert asian.analytic is None
    assert asian.price > 0.0


def test_asian_put_is_positive() -> None:
    asian = mc_asian_price(100, 100, 0.05, 0.2, 1.0, kind="put", steps=12, n_paths=5_000, seed=2)
    assert asian.price > 0.0
