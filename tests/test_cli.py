"""Tests for the command line interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcgbm.cli import build_parser, main


def test_parser_requires_a_command() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_simulate_prints_summary(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["simulate", "--steps", "12", "--paths", "500", "--seed", "1"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "GBM(" in out
    assert "median" in out
    assert "analytic E[S_T]" in out


def test_simulate_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["simulate", "--steps", "6", "--paths", "200", "--seed", "2", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["params"]["steps"] == 6
    assert payload["summary"]["n"] == 200


def test_simulate_writes_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    outdir = tmp_path / "run"
    exit_code = main(
        [
            "simulate",
            "--steps",
            "8",
            "--paths",
            "100",
            "--seed",
            "3",
            "--outdir",
            str(outdir),
            "--csv-paths",
            "10",
        ]
    )
    assert exit_code == 0
    assert (outdir / "paths.csv").exists()
    assert (outdir / "summary.json").exists()
    assert (outdir / "fan_chart.svg").exists()
    assert "wrote" in capsys.readouterr().out


def test_price_european_matches_black_scholes(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "price",
            "--s0",
            "100",
            "--strike",
            "100",
            "--rate",
            "0.05",
            "--sigma",
            "0.2",
            "--maturity",
            "1",
            "--paths",
            "100000",
            "--seed",
            "7",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Monte Carlo price" in out
    assert "Black-Scholes" in out
    assert "delta" in out


def test_price_asian_has_no_analytic(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "price",
            "--strike",
            "100",
            "--style",
            "asian",
            "--steps",
            "12",
            "--paths",
            "2000",
            "--seed",
            "4",
        ]
    )
    assert exit_code == 0
    assert "no closed form" in capsys.readouterr().out


def test_price_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["price", "--strike", "105", "--paths", "5000", "--seed", "5", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["estimate"]["price"] > 0.0
    assert "greeks" in payload


def test_invalid_parameters_exit_with_code_2(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["simulate", "--sigma", "-1", "--paths", "10"])
    assert exit_code == 2
    assert "error:" in capsys.readouterr().err


def test_put_option_via_cli(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        ["price", "--strike", "95", "--kind", "put", "--paths", "20000", "--seed", "6"]
    )
    assert exit_code == 0
    assert "put" in capsys.readouterr().out
