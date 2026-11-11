"""Descriptive statistics and risk measures for simulated price samples.

Everything here is implemented with the standard library only, so the package
runs in any Python 3.10+ environment without third-party dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

__all__ = [
    "mean",
    "stdev",
    "quantile",
    "percentile",
    "value_at_risk",
    "conditional_value_at_risk",
    "Summary",
    "summarize",
]


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean, using an accurate compensated summation."""
    if not values:
        raise ValueError("mean() requires at least one value")
    return math.fsum(values) / len(values)


def stdev(values: Sequence[float], *, sample: bool = True) -> float:
    """Standard deviation.

    ``sample=True`` (default) uses the ``n - 1`` denominator (unbiased
    estimator); ``sample=False`` uses the population denominator ``n``.
    """
    n = len(values)
    if n == 0:
        raise ValueError("stdev() requires at least one value")
    if n == 1:
        return 0.0
    mu = mean(values)
    ss = math.fsum((v - mu) ** 2 for v in values)
    denominator = n - 1 if sample else n
    return math.sqrt(ss / denominator)


def quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile of ``values`` for ``q`` in ``[0, 1]``.

    Uses the same definition as ``numpy.quantile`` with ``method="linear"``
    (Hyndman & Fan type 7), so results are comparable with common tooling.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q!r}")
    if not values:
        raise ValueError("quantile() requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def percentile(values: Sequence[float], p: float) -> float:
    """Percentile ``p`` (0-100) of ``values``."""
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"p must be in [0, 100], got {p!r}")
    return quantile(values, p / 100.0)


def value_at_risk(
    values: Sequence[float], level: float = 0.95, *, reference: float = 0.0
) -> float:
    """Value at Risk as a *loss* amount (a positive number means a loss).

    The loss is measured against ``reference`` (typically the initial price
    ``S_0``): ``level=0.95`` returns
    ``max(0, reference - quantile(values, 0.05))``, the loss that is not
    exceeded in 95 % of the simulated scenarios. The result is floored at zero,
    so a sample whose 5 % quantile is above ``reference`` reports no loss.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level!r}")
    return max(0.0, reference - quantile(values, 1.0 - level))


def conditional_value_at_risk(
    values: Sequence[float], level: float = 0.95, *, reference: float = 0.0
) -> float:
    """Expected shortfall: mean loss in the worst ``1 - level`` tail.

    Like :func:`value_at_risk`, the loss is measured against ``reference``.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level!r}")
    threshold = quantile(values, 1.0 - level)
    tail = [v for v in values if v <= threshold]
    if not tail:
        return max(0.0, reference - threshold)
    return max(0.0, reference - mean(tail))


@dataclass(frozen=True)
class Summary:
    """Summary statistics of a terminal-value sample."""

    n: int
    mean: float
    stdev: float
    stderr: float
    minimum: float
    maximum: float
    median: float
    p05: float
    p25: float
    p75: float
    p95: float
    expected_value: float
    expected_return: float
    prob_loss: float
    var_95: float
    cvar_95: float

    def to_dict(self) -> dict[str, float | int]:
        """Plain ``dict`` representation, suitable for JSON."""
        return asdict(self)

    def as_rows(self) -> list[tuple[str, float]]:
        """``(label, value)`` pairs for tabular display."""
        return [
            ("paths", float(self.n)),
            ("mean", self.mean),
            ("std dev", self.stdev),
            ("std error", self.stderr),
            ("minimum", self.minimum),
            ("p05", self.p05),
            ("p25", self.p25),
            ("median", self.median),
            ("p75", self.p75),
            ("p95", self.p95),
            ("maximum", self.maximum),
            ("expected value", self.expected_value),
            ("expected return", self.expected_return),
            ("prob. of loss", self.prob_loss),
            ("VaR 95%", self.var_95),
            ("CVaR 95%", self.cvar_95),
        ]


def summarize(
    values: Iterable[float],
    *,
    s0: float | None = None,
    expected_value: float | None = None,
) -> Summary:
    """Compute a :class:`Summary` from a sample of terminal prices.

    Parameters
    ----------
    values:
        Terminal prices ``S_T`` (one per simulated path).
    s0:
        Initial price, used to express the expected return. When omitted the
        expected return is not computed (reported as ``nan``).
    expected_value:
        Analytic expectation ``E[S_T]``, when known, so the Monte Carlo mean
        can be compared with the theoretical value.
    """
    sample = list(values)
    if not sample:
        raise ValueError("summarize() requires at least one value")

    n = len(sample)
    mu = mean(sample)
    sd = stdev(sample)
    stderr = sd / math.sqrt(n) if n > 1 else 0.0
    if expected_value is None:
        expected_value = mu
    expected_return = expected_value / s0 - 1.0 if s0 is not None and s0 > 0.0 else float("nan")

    return Summary(
        n=n,
        mean=mu,
        stdev=sd,
        stderr=stderr,
        minimum=min(sample),
        maximum=max(sample),
        median=quantile(sample, 0.5),
        p05=quantile(sample, 0.05),
        p25=quantile(sample, 0.25),
        p75=quantile(sample, 0.75),
        p95=quantile(sample, 0.95),
        expected_value=expected_value,
        expected_return=expected_return,
        prob_loss=sum(1 for v in sample if v < (s0 if s0 is not None else 0.0)) / n,
        var_95=value_at_risk(sample, 0.95, reference=s0 if s0 is not None else 0.0),
        cvar_95=conditional_value_at_risk(sample, 0.95, reference=s0 if s0 is not None else 0.0),
    )
