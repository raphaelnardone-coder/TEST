"""Monte Carlo engine for geometric Brownian motion paths.

The engine uses the *exact* solution of the GBM stochastic differential
equation, so a path is sampled without time-discretisation bias::

    S_{t+dt} = S_t * exp((mu - q - sigma**2 / 2) * dt + sigma * sqrt(dt) * Z)

with ``Z`` a standard normal draw. Only the Monte Carlo error remains, and it
shrinks like ``1 / sqrt(n_paths)``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from .params import GBMParams
from .stats import percentile

__all__ = ["SimulationResult", "simulate_gbm", "simulate_terminal_values"]


@dataclass(frozen=True)
class SimulationResult:
    """Immutable container for a set of simulated GBM paths.

    Attributes
    ----------
    params:
        The parameters that produced the paths.
    times:
        Observation dates in years, length ``steps + 1``.
    paths:
        One tuple of prices per path, each of length ``steps + 1`` and
        starting at ``params.s0``.
    """

    params: GBMParams
    times: tuple[float, ...]
    paths: tuple[tuple[float, ...], ...]

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------
    @property
    def n_paths(self) -> int:
        """Number of simulated paths."""
        return len(self.paths)

    @property
    def n_steps(self) -> int:
        """Number of time steps per path."""
        return len(self.times) - 1

    def path(self, index: int) -> tuple[float, ...]:
        """Return path number ``index``."""
        return self.paths[index]

    def terminal_values(self) -> list[float]:
        """Final price ``S_T`` of every path."""
        return [p[-1] for p in self.paths]

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[tuple[float, ...]]:
        return iter(self.paths)

    # ------------------------------------------------------------------
    # Cross-sectional statistics along the time axis
    # ------------------------------------------------------------------
    def mean_path(self) -> list[float]:
        """Average price across paths at each observation date."""
        n = self.n_paths
        return [math.fsum(column) / n for column in zip(*self.paths, strict=True)]

    def percentile_path(self, q: float) -> list[float]:
        """``q``-th percentile (0-100) across paths at each date."""
        return [percentile(column, q) for column in zip(*self.paths, strict=True)]

    def percentile_paths(self, quantiles: Sequence[float]) -> dict[float, list[float]]:
        """Several percentile paths at once, keyed by quantile."""
        return {q: self.percentile_path(q) for q in quantiles}

    def fan_chart_data(
        self, quantiles: Sequence[float] = (5, 25, 50, 75, 95)
    ) -> dict[str, list[float]]:
        """Data ready for a fan chart: mean plus one band per quantile."""
        data = {"mean": self.mean_path()}
        for q in quantiles:
            data[f"p{int(q):02d}"] = self.percentile_path(q)
        return data

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------
    def to_rows(self) -> Iterator[tuple[float, ...]]:
        """Yield ``(time, price_path_0, ..., price_path_n)`` rows."""
        for i, t in enumerate(self.times):
            yield (t, *(p[i] for p in self.paths))

    def to_dict(self, max_paths: int | None = 100) -> dict[str, object]:
        """JSON-friendly representation (paths optionally truncated)."""
        paths = self.paths if max_paths is None else self.paths[:max_paths]
        return {
            "params": self.params.to_dict(),
            "times": list(self.times),
            "n_paths": self.n_paths,
            "n_steps": self.n_steps,
            "paths": [list(p) for p in paths],
            "paths_truncated": len(paths) < self.n_paths,
        }


def _build_path(s0: float, drift: float, vol: float, shocks: Sequence[float]) -> tuple[float, ...]:
    """Turn a sequence of standard normal shocks into a price path."""
    prices = [s0]
    price = s0
    for z in shocks:
        price *= math.exp(drift + vol * z)
        prices.append(price)
    return tuple(prices)


def simulate_gbm(params: GBMParams, *, rng: random.Random | None = None) -> SimulationResult:
    """Simulate ``params.n_paths`` GBM paths.

    Parameters
    ----------
    params:
        Simulation parameters (see :class:`mcgbm.params.GBMParams`).
    rng:
        Optional pre-seeded :class:`random.Random`. When omitted a fresh
        generator seeded with ``params.seed`` is used, so the same parameters
        always produce the same paths.

    Returns
    -------
    SimulationResult
        The simulated paths together with their observation dates.

    Examples
    --------
    >>> from mcgbm import GBMParams, simulate_gbm
    >>> res = simulate_gbm(GBMParams(s0=100, mu=0.05, sigma=0.2, steps=12,
    ...                              n_paths=1000, seed=42))
    >>> len(res), res.n_steps
    (1000, 12)
    """
    if rng is None:
        rng = random.Random(params.seed)

    dt = params.dt
    drift = params.log_drift * dt
    vol = params.sigma * math.sqrt(dt)
    s0 = params.s0
    steps = params.steps
    n_paths = params.n_paths

    paths: list[tuple[float, ...]] = []
    if params.antithetic:
        n_pairs = n_paths // 2
        for _ in range(n_pairs):
            shocks = [rng.gauss(0.0, 1.0) for _ in range(steps)]
            paths.append(_build_path(s0, drift, vol, shocks))
            paths.append(_build_path(s0, drift, vol, [-z for z in shocks]))
        if n_paths % 2:  # odd number of paths: one unpaired draw
            shocks = [rng.gauss(0.0, 1.0) for _ in range(steps)]
            paths.append(_build_path(s0, drift, vol, shocks))
    else:
        for _ in range(n_paths):
            shocks = [rng.gauss(0.0, 1.0) for _ in range(steps)]
            paths.append(_build_path(s0, drift, vol, shocks))

    return SimulationResult(params=params, times=params.times, paths=tuple(paths))


def simulate_terminal_values(params: GBMParams, *, rng: random.Random | None = None) -> list[float]:
    """Simulate only the terminal price ``S_T`` of each path.

    This is much cheaper than :func:`simulate_gbm` when the whole path is not
    needed (for example to price a European option), because it draws a single
    normal per path instead of one per step.
    """
    if rng is None:
        rng = random.Random(params.seed)

    drift = params.log_drift * params.horizon
    vol = params.sigma * math.sqrt(params.horizon)
    s0 = params.s0
    n_paths = params.n_paths

    values: list[float] = []
    if params.antithetic:
        for _ in range(n_paths // 2):
            z = rng.gauss(0.0, 1.0)
            values.append(s0 * math.exp(drift + vol * z))
            values.append(s0 * math.exp(drift - vol * z))
        if n_paths % 2:
            values.append(s0 * math.exp(drift + vol * rng.gauss(0.0, 1.0)))
    else:
        for _ in range(n_paths):
            values.append(s0 * math.exp(drift + vol * rng.gauss(0.0, 1.0)))
    return values
