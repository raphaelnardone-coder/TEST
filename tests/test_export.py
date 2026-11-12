"""Tests for the CSV / JSON / SVG exporters."""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from mcgbm import (
    GBMParams,
    simulate_gbm,
    summarize,
    write_fan_chart_svg,
    write_paths_csv,
    write_report_json,
    write_summary_json,
)


@pytest.fixture()
def result():
    return simulate_gbm(GBMParams(steps=10, n_paths=20, seed=42))


def test_write_paths_csv_shape(tmp_path: Path, result) -> None:
    target = write_paths_csv(result, tmp_path / "paths.csv")
    assert target.exists()
    with target.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == result.n_steps + 2  # header + one row per date
    assert rows[0][0] == "time"
    assert len(rows[0]) == result.n_paths + 1
    assert float(rows[1][0]) == pytest.approx(0.0)
    assert float(rows[1][1]) == pytest.approx(result.params.s0)


def test_write_paths_csv_truncates_columns(tmp_path: Path, result) -> None:
    target = write_paths_csv(result, tmp_path / "paths.csv", max_paths=5)
    with target.open(encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert len(header) == 6


def test_write_paths_csv_creates_parent_dirs(tmp_path: Path, result) -> None:
    target = write_paths_csv(result, tmp_path / "nested" / "deep" / "paths.csv")
    assert target.exists()


def test_write_summary_json(tmp_path: Path, result) -> None:
    summary = summarize(result.terminal_values(), s0=result.params.s0)
    target = write_summary_json(summary, tmp_path / "summary.json", params=result.params.to_dict())
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["params"]["s0"] == result.params.s0
    assert payload["summary"]["n"] == result.n_paths
    assert "median" in payload["summary"]


def test_write_report_json_contains_fan_chart(tmp_path: Path, result) -> None:
    summary = summarize(result.terminal_values(), s0=result.params.s0)
    target = write_report_json(result, summary, tmp_path / "report.json", extra={"note": "x"})
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["note"] == "x"
    assert len(payload["times"]) == result.n_steps + 1
    assert set(payload["fan_chart"]) == {"mean", "p05", "p25", "p50", "p75", "p95"}


def test_fan_chart_svg_is_valid_xml(tmp_path: Path, result) -> None:
    target = write_fan_chart_svg(result, tmp_path / "fan.svg")
    text = target.read_text(encoding="utf-8")
    assert text.startswith("<svg")
    root = ET.fromstring(text)  # raises if malformed
    assert root.tag.endswith("svg")
    assert len(list(root.iter())) > 10


def test_fan_chart_svg_has_title_and_legend(tmp_path: Path, result) -> None:
    target = write_fan_chart_svg(result, tmp_path / "fan.svg", title="My chart")
    text = target.read_text(encoding="utf-8")
    assert "My chart" in text
    assert "median path" in text


def test_fan_chart_escapes_special_characters(tmp_path: Path, result) -> None:
    target = write_fan_chart_svg(result, tmp_path / "fan.svg", title="A & B <test>")
    text = target.read_text(encoding="utf-8")
    assert "&amp;" in text and "&lt;test&gt;" in text
    ET.fromstring(text)  # still valid XML


def test_fan_chart_handles_zero_volatility(tmp_path: Path) -> None:
    """A flat simulation must not divide by zero when scaling the y axis."""
    flat = simulate_gbm(GBMParams(sigma=0.0, steps=5, n_paths=4, seed=1))
    target = write_fan_chart_svg(flat, tmp_path / "flat.svg")
    ET.fromstring(target.read_text(encoding="utf-8"))
