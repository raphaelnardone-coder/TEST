"""Tests for the statistics and risk-measure helpers."""

from __future__ import annotations

import math

import pytest

from mcgbm import (
    conditional_value_at_risk,
    mean,
    percentile,
    quantile,
    stdev,
    summarize,
    value_at_risk,
)


def test_mean_uses_accurate_summation() -> None:
    assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert mean([0.1] * 10) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        mean([])


def test_stdev_sample_and_population() -> None:
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    assert stdev(values, sample=False) == pytest.approx(2.0)
    assert stdev(values, sample=True) == pytest.approx(math.sqrt(32.0 / 7.0))
    assert stdev([3.0]) == 0.0
    with pytest.raises(ValueError):
        stdev([])


def test_quantile_matches_numpy_linear_definition() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert quantile(values, 0.0) == 1.0
    assert quantile(values, 1.0) == 4.0
    assert quantile(values, 0.5) == pytest.approx(2.5)
    assert quantile(values, 0.25) == pytest.approx(1.75)
    assert quantile([7.0], 0.9) == 7.0


def test_quantile_is_order_independent() -> None:
    a = [5.0, 1.0, 9.0, 3.0]
    b = sorted(a)
    assert quantile(a, 0.3) == pytest.approx(quantile(b, 0.3))


@pytest.mark.parametrize("q", [-0.1, 1.1])
def test_quantile_rejects_out_of_range(q: float) -> None:
    with pytest.raises(ValueError):
        quantile([1.0, 2.0], q)


def test_percentile_is_quantile_over_100() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 50.0) == pytest.approx(quantile(values, 0.5))
    with pytest.raises(ValueError):
        percentile(values, 101.0)


def test_value_at_risk_measures_loss_against_reference() -> None:
    values = [80.0, 90.0, 100.0, 110.0, 120.0]
    # Linear 5 % quantile of this sample is 82 (0.8 * 80 + 0.2 * 90),
    # so the loss measured against 100 is 18.
    assert quantile(values, 0.05) == pytest.approx(82.0)
    assert value_at_risk(values, 0.95, reference=100.0) == pytest.approx(18.0)
    # Without a reference the loss is measured against zero and floored at 0.
    assert value_at_risk(values, 0.95) == 0.0


def test_value_at_risk_is_never_negative() -> None:
    assert value_at_risk([200.0, 210.0], 0.95, reference=100.0) == 0.0


def test_conditional_value_at_risk_exceeds_var() -> None:
    values = [50.0, 70.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0]
    var = value_at_risk(values, 0.9, reference=100.0)
    cvar = conditional_value_at_risk(values, 0.9, reference=100.0)
    assert cvar >= var > 0.0


@pytest.mark.parametrize("level", [0.0, 1.0, -0.5, 1.5])
def test_risk_measures_reject_bad_level(level: float) -> None:
    with pytest.raises(ValueError):
        value_at_risk([1.0, 2.0], level)
    with pytest.raises(ValueError):
        conditional_value_at_risk([1.0, 2.0], level)


def test_summarize_basic_fields() -> None:
    values = [90.0, 100.0, 110.0]
    summary = summarize(values, s0=100.0, expected_value=105.0)
    assert summary.n == 3
    assert summary.mean == pytest.approx(100.0)
    assert summary.minimum == 90.0
    assert summary.maximum == 110.0
    assert summary.median == pytest.approx(100.0)
    assert summary.expected_value == pytest.approx(105.0)
    assert summary.expected_return == pytest.approx(0.05)
    assert summary.prob_loss == pytest.approx(1 / 3)
    assert summary.stderr == pytest.approx(stdev(values) / math.sqrt(3))


def test_summarize_without_s0() -> None:
    summary = summarize([1.0, 2.0, 3.0])
    assert math.isnan(summary.expected_return)
    assert summary.expected_value == pytest.approx(2.0)
    assert summary.prob_loss == 0.0


def test_summary_ordering_of_quantiles() -> None:
    values = [float(i) for i in range(1, 101)]
    summary = summarize(values, s0=50.0)
    assert summary.minimum <= summary.p05 <= summary.p25 <= summary.median
    assert summary.median <= summary.p75 <= summary.p95 <= summary.maximum


def test_summary_to_dict_and_rows() -> None:
    summary = summarize([1.0, 2.0, 3.0], s0=2.0)
    payload = summary.to_dict()
    assert payload["n"] == 3
    rows = summary.as_rows()
    labels = [label for label, _ in rows]
    assert "median" in labels and "VaR 95%" in labels
    assert len(rows) == len(payload)


def test_summarize_rejects_empty() -> None:
    with pytest.raises(ValueError):
        summarize([])
