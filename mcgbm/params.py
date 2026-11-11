"""Parameter objects for Monte Carlo geometric Brownian motion (GBM) simulations.

The GBM model describes an asset price ``S_t`` following

.. math::

    dS_t = (\\mu - q) S_t dt + \\sigma S_t dW_t

whose exact solution between two dates is

.. math::

    S_{t+\\Delta t} = S_t \\exp\\left(
        \\left(\\mu - q - \\tfrac{1}{2}\\sigma^2\\right)\\Delta t
        + \\sigma\\sqrt{\\Delta t}\\, Z
    \\right), \\qquad Z \\sim \\mathcal{N}(0, 1).

Because the scheme above is the exact solution of the SDE, simulating it
introduces no discretisation bias: only Monte Carlo (statistical) error.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

__all__ = ["GBMParams", "MAX_CELLS"]

#: Upper bound on ``n_paths * (steps + 1)`` stored floats, to fail fast instead
#: of exhausting memory on an accidental huge request.
MAX_CELLS = 100_000_000


@dataclass(frozen=True)
class GBMParams:
    """Immutable description of a GBM simulation.

    Parameters
    ----------
    s0:
        Initial spot price, strictly positive.
    mu:
        Annualised expected (arithmetic) return of the asset, e.g. ``0.08``
        for 8 % per year. Under the risk-neutral measure this is the
        risk-free rate.
    sigma:
        Annualised volatility, e.g. ``0.2`` for 20 % per year.
    horizon:
        Time to maturity in years, e.g. ``1.0`` or ``0.25`` for one quarter.
    steps:
        Number of time steps per path (``252`` is a typical trading year).
    n_paths:
        Number of simulated paths.
    seed:
        Seed for the pseudo random number generator. ``None`` means a
        non-reproducible run; any integer makes the run reproducible.
    antithetic:
        If ``True``, each block of Gaussian draws is reused with the opposite
        sign, which cancels part of the sampling error and lowers the
        variance of the estimator.
    dividend_yield:
        Continuous dividend yield ``q``, e.g. ``0.02`` for 2 % per year.
    """

    s0: float = 100.0
    mu: float = 0.08
    sigma: float = 0.2
    horizon: float = 1.0
    steps: int = 252
    n_paths: int = 10_000
    seed: int | None = None
    antithetic: bool = True
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.s0) or self.s0 <= 0.0:
            raise ValueError(f"s0 must be a finite positive number, got {self.s0!r}")
        if not math.isfinite(self.mu):
            raise ValueError(f"mu must be finite, got {self.mu!r}")
        if not math.isfinite(self.sigma) or self.sigma < 0.0:
            raise ValueError(f"sigma must be finite and >= 0, got {self.sigma!r}")
        if not math.isfinite(self.horizon) or self.horizon <= 0.0:
            raise ValueError(f"horizon must be a finite positive number, got {self.horizon!r}")
        if not math.isfinite(self.dividend_yield):
            raise ValueError(f"dividend_yield must be finite, got {self.dividend_yield!r}")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError(f"steps must be an integer >= 1, got {self.steps!r}")
        if isinstance(self.n_paths, bool) or not isinstance(self.n_paths, int) or self.n_paths < 1:
            raise ValueError(f"n_paths must be an integer >= 1, got {self.n_paths!r}")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise ValueError(f"seed must be an integer or None, got {self.seed!r}")
        if self.n_paths * (self.steps + 1) > MAX_CELLS:
            raise ValueError(
                f"n_paths * (steps + 1) = {self.n_paths * (self.steps + 1):,} exceeds the "
                f"limit of {MAX_CELLS:,} stored values; reduce n_paths or steps"
            )

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    @property
    def dt(self) -> float:
        """Length of one time step in years."""
        return self.horizon / self.steps

    @property
    def times(self) -> tuple[float, ...]:
        """Observation dates of a path, from ``0.0`` to ``horizon`` inclusive."""
        dt = self.dt
        return tuple(i * dt for i in range(self.steps + 1))

    @property
    def log_drift(self) -> float:
        """Annualised drift of ``log S``: ``mu - q - sigma**2 / 2``."""
        return self.mu - self.dividend_yield - 0.5 * self.sigma**2

    @property
    def annual_vol(self) -> float:
        """Annualised volatility (alias of :attr:`sigma`)."""
        return self.sigma

    @property
    def expected_terminal_value(self) -> float:
        """Analytic expectation ``E[S_T] = S_0 * exp((mu - q) * T)``."""
        return self.s0 * math.exp((self.mu - self.dividend_yield) * self.horizon)

    @property
    def n_cells(self) -> int:
        """Number of stored prices (``n_paths * (steps + 1)``)."""
        return self.n_paths * (self.steps + 1)

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------
    def risk_neutral(self, rate: float) -> GBMParams:
        """Return a copy drifting at the risk-free ``rate`` (pricing measure)."""
        if not math.isfinite(rate):
            raise ValueError(f"rate must be finite, got {rate!r}")
        return replace(self, mu=rate)

    def with_(self, **changes: Any) -> GBMParams:
        """Return a copy with the given fields replaced (validated again)."""
        return replace(self, **changes)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Plain ``dict`` representation, suitable for JSON."""
        return {
            "s0": self.s0,
            "mu": self.mu,
            "sigma": self.sigma,
            "horizon": self.horizon,
            "steps": self.steps,
            "n_paths": self.n_paths,
            "seed": self.seed,
            "antithetic": self.antithetic,
            "dividend_yield": self.dividend_yield,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GBMParams:
        """Build parameters from a mapping, ignoring unknown keys."""
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    def describe(self) -> str:
        """One-line human readable summary."""
        return (
            f"GBM(S0={self.s0:g}, mu={self.mu:g}, sigma={self.sigma:g}, "
            f"q={self.dividend_yield:g}, T={self.horizon:g}y, "
            f"steps={self.steps}, paths={self.n_paths:,}, "
            f"antithetic={self.antithetic}, seed={self.seed})"
        )
