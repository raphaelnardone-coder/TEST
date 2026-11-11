"""Tests for the GBM parameter object."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from mcgbm import GBMParams


def test_defaults_are_valid() -> None:
    params = GBMParams()
    assert params.s0 == 100.0
    assert params.steps == 252
    assert params.n_paths == 10_000


def test_derived_quantities() -> None:
    params = GBMParams(s0=100.0, mu=0.08, sigma=0.2, horizon=1.0, steps=4)
    assert params.dt == pytest.approx(0.25)
    assert params.times == pytest.approx((0.0, 0.25, 0.5, 0.75, 1.0))
    assert params.log_drift == pytest.approx(0.08 - 0.5 * 0.2**2)
    assert params.expected_terminal_value == pytest.approx(100.0 * math.exp(0.08))
    assert params.n_cells == 5 * params.n_paths


def test_dividend_yield_reduces_expected_value() -> None:
    without = GBMParams(mu=0.08, dividend_yield=0.0)
    with_div = GBMParams(mu=0.08, dividend_yield=0.03)
    assert with_div.expected_terminal_value < without.expected_terminal_value
    assert with_div.log_drift == pytest.approx(without.log_drift - 0.03)


def test_risk_neutral_sets_drift_to_rate() -> None:
    params = GBMParams(mu=0.15, sigma=0.25)
    neutral = params.risk_neutral(0.03)
    assert neutral.mu == 0.03
    assert neutral.sigma == params.sigma
    assert params.mu == 0.15  # original untouched (frozen dataclass)


def test_params_are_immutable() -> None:
    params = GBMParams()
    with pytest.raises(FrozenInstanceError):
        params.s0 = 50.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"s0": 0.0},
        {"s0": -10.0},
        {"s0": float("nan")},
        {"sigma": -0.1},
        {"sigma": float("inf")},
        {"horizon": 0.0},
        {"horizon": -1.0},
        {"steps": 0},
        {"steps": -5},
        {"steps": 2.5},
        {"n_paths": 0},
        {"n_paths": -1},
        {"mu": float("nan")},
        {"dividend_yield": float("nan")},
    ],
)
def test_invalid_parameters_raise(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        GBMParams(**kwargs)  # type: ignore[arg-type]


def test_boolean_is_rejected_as_integer() -> None:
    with pytest.raises(ValueError):
        GBMParams(steps=True)  # type: ignore[arg-type]


def test_memory_guard_rejects_huge_simulation() -> None:
    with pytest.raises(ValueError, match="exceeds the limit"):
        GBMParams(steps=100_000, n_paths=100_000)


def test_round_trip_dict() -> None:
    params = GBMParams(s0=42.0, mu=0.05, sigma=0.3, steps=12, n_paths=500, seed=7)
    restored = GBMParams.from_dict(params.to_dict())
    assert restored == params


def test_from_dict_ignores_unknown_keys() -> None:
    params = GBMParams.from_dict({"s0": 50.0, "unknown": "ignored"})
    assert params.s0 == 50.0


def test_with_returns_validated_copy() -> None:
    params = GBMParams()
    changed = params.with_(sigma=0.4)
    assert changed.sigma == 0.4
    assert params.sigma == 0.2
    with pytest.raises(ValueError):
        params.with_(sigma=-1.0)


def test_describe_mentions_key_inputs() -> None:
    text = GBMParams(s0=123.0, sigma=0.33, seed=9).describe()
    assert "123" in text and "0.33" in text and "seed=9" in text
