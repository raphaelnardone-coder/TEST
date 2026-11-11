"""Option pricing with Monte Carlo GBM and closed-form Black-Scholes benchmarks.

The Monte Carlo prices are computed under the risk-neutral measure (drift equal
to the risk-free rate minus the dividend yield) and reported with a standard
error and a 95 % confidence interval, so the statistical error is explicit.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

from .params import GBMParams
from .simulation import simulate_gbm, simulate_terminal_values
from .stats import mean, stdev

__all__ = [
    "norm_cdf",
    "norm_pdf",
    "black_scholes_price",
    "black_scholes_greeks",
    "PriceEstimate",
    "mc_european_price",
    "mc_asian_price",
]

_NORMAL = NormalDist()
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return _NORMAL.cdf(x)


def _validate_option(kind: str, strike: float, maturity: float) -> str:
    kind = kind.lower()
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    if not math.isfinite(strike) or strike <= 0.0:
        raise ValueError(f"strike must be a finite positive number, got {strike!r}")
    if not math.isfinite(maturity) or maturity <= 0.0:
        raise ValueError(f"maturity must be a finite positive number, got {maturity!r}")
    return kind


def black_scholes_price(
    s0: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    *,
    kind: str = "call",
    dividend_yield: float = 0.0,
) -> float:
    """Closed-form Black-Scholes-Merton price of a European option.

    Parameters
    ----------
    s0, strike, rate, sigma, maturity:
        Spot, strike, continuously compounded risk-free rate, volatility and
        time to maturity in years.
    kind:
        ``"call"`` or ``"put"``.
    dividend_yield:
        Continuous dividend yield ``q``.

    Examples
    --------
    >>> round(black_scholes_price(100, 100, 0.05, 0.2, 1.0), 4)
    10.4506
    """
    kind = _validate_option(kind, strike, maturity)
    if not math.isfinite(s0) or s0 <= 0.0:
        raise ValueError(f"s0 must be a finite positive number, got {s0!r}")
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError(f"sigma must be finite and >= 0, got {sigma!r}")

    df_r = math.exp(-rate * maturity)
    df_q = math.exp(-dividend_yield * maturity)
    if sigma == 0.0:
        forward = s0 * df_q / df_r
        intrinsic = max(0.0, forward - strike) if kind == "call" else max(0.0, strike - forward)
        return df_r * intrinsic

    vol_sqrt_t = sigma * math.sqrt(maturity)
    d1 = (math.log(s0 / strike) + (rate - dividend_yield + 0.5 * sigma**2) * maturity) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    if kind == "call":
        return s0 * df_q * norm_cdf(d1) - strike * df_r * norm_cdf(d2)
    return strike * df_r * norm_cdf(-d2) - s0 * df_q * norm_cdf(-d1)


def black_scholes_greeks(
    s0: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    *,
    kind: str = "call",
    dividend_yield: float = 0.0,
) -> dict[str, float]:
    """Black-Scholes Greeks: ``delta``, ``gamma``, ``vega``, ``theta``, ``rho``.

    ``vega`` is per 1.00 of volatility and ``theta`` per year; divide by 100 and
    252 respectively for the usual per-point and per-day conventions.
    """
    kind = _validate_option(kind, strike, maturity)
    df_r = math.exp(-rate * maturity)
    df_q = math.exp(-dividend_yield * maturity)
    vol_sqrt_t = sigma * math.sqrt(maturity)
    if vol_sqrt_t == 0.0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    d1 = (math.log(s0 / strike) + (rate - dividend_yield + 0.5 * sigma**2) * maturity) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    pdf_d1 = norm_pdf(d1)
    gamma = df_q * pdf_d1 / (s0 * vol_sqrt_t)
    vega = s0 * df_q * pdf_d1 * math.sqrt(maturity)

    if kind == "call":
        delta = df_q * norm_cdf(d1)
        theta = (
            -s0 * df_q * pdf_d1 * sigma / (2.0 * math.sqrt(maturity))
            - rate * strike * df_r * norm_cdf(d2)
            + dividend_yield * s0 * df_q * norm_cdf(d1)
        )
        rho = strike * maturity * df_r * norm_cdf(d2)
    else:
        delta = -df_q * norm_cdf(-d1)
        theta = (
            -s0 * df_q * pdf_d1 * sigma / (2.0 * math.sqrt(maturity))
            + rate * strike * df_r * norm_cdf(-d2)
            - dividend_yield * s0 * df_q * norm_cdf(-d1)
        )
        rho = -strike * maturity * df_r * norm_cdf(-d2)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


@dataclass(frozen=True)
class PriceEstimate:
    """A Monte Carlo price together with its statistical error."""

    price: float
    stderr: float
    ci_low: float
    ci_high: float
    n_paths: int
    analytic: float | None = None

    @property
    def ci_width(self) -> float:
        """Width of the 95 % confidence interval."""
        return self.ci_high - self.ci_low

    @property
    def error_vs_analytic(self) -> float | None:
        """Signed difference between the Monte Carlo and analytic price."""
        if self.analytic is None:
            return None
        return self.price - self.analytic

    def to_dict(self) -> dict[str, float | int | None]:
        """Plain ``dict`` representation, suitable for JSON."""
        return {
            "price": self.price,
            "stderr": self.stderr,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_paths": self.n_paths,
            "analytic": self.analytic,
            "error_vs_analytic": self.error_vs_analytic,
        }


def _estimate(payoffs: Sequence[float], discount: float, analytic: float | None) -> PriceEstimate:
    n = len(payoffs)
    price = discount * mean(payoffs)
    stderr = discount * stdev(payoffs) / math.sqrt(n) if n > 1 else 0.0
    return PriceEstimate(
        price=price,
        stderr=stderr,
        ci_low=price - 1.96 * stderr,
        ci_high=price + 1.96 * stderr,
        n_paths=n,
        analytic=analytic,
    )


def mc_european_price(
    s0: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    *,
    kind: str = "call",
    dividend_yield: float = 0.0,
    n_paths: int = 100_000,
    seed: int | None = None,
    antithetic: bool = True,
) -> PriceEstimate:
    """Price a European option by Monte Carlo under the risk-neutral measure.

    Only the terminal price is simulated, which is exact for GBM. The analytic
    Black-Scholes price is attached to the result for comparison.
    """
    kind = _validate_option(kind, strike, maturity)
    params = GBMParams(
        s0=s0,
        mu=rate,
        sigma=sigma,
        horizon=maturity,
        steps=1,
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
        dividend_yield=dividend_yield,
    )
    terminal = simulate_terminal_values(params, rng=random.Random(seed))
    if kind == "call":
        payoffs = [max(0.0, s - strike) for s in terminal]
    else:
        payoffs = [max(0.0, strike - s) for s in terminal]
    analytic = black_scholes_price(
        s0, strike, rate, sigma, maturity, kind=kind, dividend_yield=dividend_yield
    )
    return _estimate(payoffs, math.exp(-rate * maturity), analytic)


def mc_asian_price(
    s0: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    *,
    kind: str = "call",
    dividend_yield: float = 0.0,
    steps: int = 52,
    n_paths: int = 20_000,
    seed: int | None = None,
    antithetic: bool = True,
) -> PriceEstimate:
    """Price a discretely monitored arithmetic-average Asian option.

    The average is taken over the ``steps`` observation dates of each path
    (excluding ``t=0``). No closed form exists for the arithmetic average, so
    ``analytic`` is left as ``None``.
    """
    kind = _validate_option(kind, strike, maturity)
    params = GBMParams(
        s0=s0,
        mu=rate,
        sigma=sigma,
        horizon=maturity,
        steps=steps,
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
        dividend_yield=dividend_yield,
    )
    result = simulate_gbm(params)
    payoffs: list[float] = []
    for path in result.paths:
        observations = path[1:]
        average = math.fsum(observations) / len(observations)
        if kind == "call":
            payoffs.append(max(0.0, average - strike))
        else:
            payoffs.append(max(0.0, strike - average))
    return _estimate(payoffs, math.exp(-rate * maturity), None)
