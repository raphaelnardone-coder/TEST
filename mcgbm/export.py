"""Export simulated paths and statistics to plain files (CSV, JSON, SVG).

Only the standard library is used, so the outputs can be produced anywhere:
no plotting dependency is required. The SVG fan chart is generated as a plain
string, which keeps the package dependency-free while still giving a visual
summary of the simulation.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from .simulation import SimulationResult
from .stats import Summary

__all__ = [
    "write_paths_csv",
    "write_summary_json",
    "write_fan_chart_svg",
    "write_report_json",
]

#: Palette used by the SVG fan chart (dark to light).
_BAND_COLORS = ("#fde3d3", "#f9c3a4", "#f2a179", "#e07a4a")


def _ensure_parent(path: Path) -> None:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def write_paths_csv(
    result: SimulationResult,
    path: str | Path,
    *,
    max_paths: int | None = None,
    float_format: str = "%.6f",
) -> Path:
    """Write simulated paths to a CSV file.

    The file has one row per observation date and one column per path, plus a
    leading ``time`` column in years. ``max_paths`` truncates the number of
    path columns, which is useful for large simulations.

    Returns the path that was written.
    """
    target = Path(path)
    _ensure_parent(target)
    n_paths = result.n_paths if max_paths is None else min(max_paths, result.n_paths)
    header = ["time", *(f"path_{i}" for i in range(n_paths))]
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for i, t in enumerate(result.times):
            row = [float_format % t]
            row.extend(float_format % result.paths[j][i] for j in range(n_paths))
            writer.writerow(row)
    return target


def write_summary_json(
    summary: Summary,
    path: str | Path,
    *,
    params: Mapping[str, object] | None = None,
    extra: Mapping[str, object] | None = None,
) -> Path:
    """Write a :class:`~mcgbm.stats.Summary` to a JSON file."""
    target = Path(path)
    _ensure_parent(target)
    payload: dict[str, object] = {}
    if params is not None:
        payload["params"] = dict(params)
    payload["summary"] = summary.to_dict()
    if extra:
        payload.update(extra)
    target.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return target


def write_report_json(
    result: SimulationResult,
    summary: Summary,
    path: str | Path,
    *,
    extra: Mapping[str, object] | None = None,
) -> Path:
    """Write parameters, summary statistics and fan-chart data to one JSON file."""
    target = Path(path)
    _ensure_parent(target)
    payload: dict[str, object] = {
        "params": result.params.to_dict(),
        "summary": summary.to_dict(),
        "times": list(result.times),
        "fan_chart": result.fan_chart_data(),
    }
    if extra:
        payload.update(extra)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def _nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    """A few round tick values between ``low`` and ``high``."""
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return [low]
    raw_step = (high - low) / max(1, count)
    magnitude = 10.0 ** math.floor(math.log10(raw_step))
    for factor in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = factor * magnitude
        if step >= raw_step:
            break
    start = math.ceil(low / step) * step
    ticks: list[float] = []
    value = start
    while value <= high + 1e-9 and len(ticks) < 50:
        ticks.append(round(value, 10))
        value += step
    return ticks or [low]


def write_fan_chart_svg(
    result: SimulationResult,
    path: str | Path,
    *,
    width: int = 900,
    height: int = 480,
    quantiles: Sequence[float] = (5, 25, 75, 95),
    title: str | None = None,
) -> Path:
    """Render a fan chart of the simulated paths as a standalone SVG file.

    The chart shows the median path, shaded bands between symmetric quantiles
    (5-95 % and 25-75 % by default) and the analytic expectation
    ``S_0 * exp((mu - q) * T)`` as a dashed line.
    """
    target = Path(path)
    _ensure_parent(target)

    times = result.times
    median = result.percentile_path(50.0)
    bands = {q: result.percentile_path(q) for q in quantiles}
    expectation = [
        result.params.s0 * math.exp((result.params.mu - result.params.dividend_yield) * t)
        for t in times
    ]

    all_values = [*median, *expectation]
    for series in bands.values():
        all_values.extend(series)
    y_min = min(all_values)
    y_max = max(all_values)
    pad = (y_max - y_min) * 0.08 or max(1.0, abs(y_max) * 0.05)
    y_min -= pad
    y_max += pad

    left, right, top, bottom = 70, 24, 56, 52
    plot_w = max(1, width - left - right)
    plot_h = max(1, height - top - bottom)
    t_max = times[-1] or 1.0

    def sx(t: float) -> float:
        return left + plot_w * (t / t_max)

    def sy(v: float) -> float:
        return top + plot_h * (1.0 - (v - y_min) / (y_max - y_min))

    def polyline(series: Sequence[float]) -> str:
        return " ".join(f"{sx(t):.2f},{sy(v):.2f}" for t, v in zip(times, series, strict=True))

    def band(low_q: float, high_q: float, color: str) -> str:
        upper = bands[high_q]
        lower = bands[low_q]
        points = [f"{sx(t):.2f},{sy(v):.2f}" for t, v in zip(times, upper, strict=True)]
        points += [
            f"{sx(t):.2f},{sy(v):.2f}"
            for t, v in zip(reversed(times), reversed(lower), strict=True)
        ]
        return f'<polygon points="{" ".join(points)}" fill="{color}" stroke="none" />'

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Monte Carlo GBM fan chart">'
    )
    parts.append(f'<rect width="{width}" height="{height}" fill="#ffffff" />')
    chart_title = title or (
        f"Monte Carlo GBM fan chart - {result.n_paths:,} paths, "
        f"sigma={result.params.sigma:g}, T={result.params.horizon:g}y"
    )
    parts.append(
        f'<text x="{left}" y="30" font-family="Georgia, serif" font-size="18" '
        f'fill="#1f2933">{_escape(chart_title)}</text>'
    )

    # Horizontal grid and y axis labels
    for tick in _nice_ticks(y_min, y_max, 5):
        y = sy(tick)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" '
            f'stroke="#e4e7eb" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="11" '
            f'fill="#52606d">{tick:,.2f}</text>'
        )

    # Quantile bands, widest first
    ordered = sorted(quantiles)
    for i in range(len(ordered) // 2):
        low_q, high_q = ordered[i], ordered[-1 - i]
        color = _BAND_COLORS[min(i, len(_BAND_COLORS) - 1)]
        parts.append(band(low_q, high_q, color))

    parts.append(
        f'<polyline points="{polyline(expectation)}" fill="none" stroke="#1f2933" '
        f'stroke-width="1.5" stroke-dasharray="6 4" />'
    )
    parts.append(
        f'<polyline points="{polyline(median)}" fill="none" stroke="#b23c17" '
        f'stroke-width="2.2" />'
    )

    # Axes
    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
        f'stroke="#9aa5b1" stroke-width="1" />'
    )
    parts.append(
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" '
        f'stroke="#9aa5b1" stroke-width="1" />'
    )
    for tick in _nice_ticks(0.0, t_max, 5):
        x = sx(tick)
        parts.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 20}" text-anchor="middle" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="11" '
            f'fill="#52606d">{tick:g}y</text>'
        )

    # Legend
    legend = [
        ("#b23c17", "median path"),
        ("#1f2933", "analytic expectation"),
        (_BAND_COLORS[0], f"p{ordered[0]:g}-p{ordered[-1]:g} band"),
    ]
    if len(ordered) >= 4:
        legend.append((_BAND_COLORS[1], f"p{ordered[1]:g}-p{ordered[-2]:g} band"))
    lx = left
    ly = height - 18
    for color, label in legend:
        parts.append(f'<rect x="{lx}" y="{ly - 9}" width="12" height="12" fill="{color}" />')
        parts.append(
            f'<text x="{lx + 18}" y="{ly + 1}" font-family="Helvetica, Arial, sans-serif" '
            f'font-size="11" fill="#3e4c59">{_escape(label)}</text>'
        )
        lx += 26 + 7 * len(label)

    parts.append("</svg>")
    target.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return target


def _escape(text: str) -> str:
    """Minimal XML escaping for text nodes."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
