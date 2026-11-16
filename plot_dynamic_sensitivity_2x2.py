#!/usr/bin/env python3
"""Combine four formal dynamic sensitivity sweeps into a single-column figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_dynamic_external_eight import save_figure, style_axis
from plot_dynamic_sensitivity_arrivals import DATASETS, STYLES, summarize as summarize_arrivals
from plot_dynamic_sensitivity_remaining import summarize as summarize_parameter


PLOT_FIELDS = (
    "panel",
    "dataset",
    "parameter",
    "value",
    "seed_count",
    "event_count",
    "paired_speedup",
    "source_batch",
    "code_fingerprint",
)


def scenario_values(manifest: dict, names: set[str], field: str) -> dict[str, float]:
    values = {
        item["name"]: float(item[field])
        for item in manifest["scenarios"]
        if item["name"] in names
    }
    if set(values) != names:
        raise ValueError(f"Missing sensitivity scenarios for {field}: {names - set(values)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("paper_results_dynamic_section_f_figures"))
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else workdir / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    sensitivity_source = workdir / "paper_results_dynamic_sensitivity_certificate_final_v2"
    k_source = workdir / "paper_results_dynamic_k_sensitivity_final"
    manifest = json.loads((workdir / "dynamic_scenarios" / "manifest.json").read_text(encoding="utf-8"))
    episodes = scenario_values(manifest, {"traffic_low", "baseline", "traffic_high"}, "edge_updates")
    factors = {
        item["name"]: float(item["traffic_factors"][0])
        for item in manifest["scenarios"]
        if item["name"] in {"severity_mild", "baseline", "severity_severe"}
    }
    if set(episodes.values()) != {4.0, 8.0, 12.0} or set(factors.values()) != {1.2, 1.5, 2.0}:
        raise ValueError("Unexpected traffic sensitivity settings")

    arrival_rows = summarize_arrivals(workdir, sensitivity_source)
    k_rows = summarize_parameter(
        k_source, (1, 3, 5, 10), "k",
        lambda row: int(row["k"]) if row["scenario"] == "baseline" else None,
    )
    episode_rows = summarize_parameter(
        sensitivity_source, (4.0, 8.0, 12.0), "traffic_episodes",
        lambda row: episodes.get(row["scenario"]),
    )
    factor_rows = summarize_parameter(
        sensitivity_source, (1.2, 1.5, 2.0), "traffic_factor",
        lambda row: factors.get(row["scenario"]),
    )
    panels = (
        ("(a) Request arrivals", "arrival_count", arrival_rows, (6, 12, 18)),
        ("(b) Top-k size", "value", k_rows, (1, 3, 5, 10)),
        ("(c) Traffic episodes", "value", episode_rows, (4, 8, 12)),
        ("(d) Traffic factor", "value", factor_rows, (1.2, 1.5, 2.0)),
    )

    audit_rows = []
    for panel_title, x_field, rows, _ in panels:
        for row in rows:
            audit_rows.append({
                "panel": panel_title,
                "dataset": row["dataset"],
                "parameter": "arrival_count" if x_field == "arrival_count" else row["parameter"],
                "value": row[x_field],
                "seed_count": row["seed_count"],
                "event_count": row["event_count"],
                "paired_speedup": row["paired_speedup"],
                "source_batch": row["source_batch"],
                "code_fingerprint": row["code_fingerprint"],
            })
    audit_path = output / "dynamic_sensitivity_2x2_plot_data.csv"
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLOT_FIELDS)
        writer.writeheader()
        writer.writerows(audit_rows)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 6.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.0), sharey=True)
    for ax, (panel_title, x_field, rows, x_values) in zip(axes.flat, panels):
        for dataset in DATASETS:
            points = sorted(
                (row for row in rows if row["dataset"] == dataset),
                key=lambda row: row[x_field],
            )
            if tuple(row[x_field] for row in points) != x_values:
                raise ValueError(f"Missing values for {panel_title} {dataset}")
            ax.plot(
                x_values,
                [row["paired_speedup"] for row in points],
                label=dataset,
                linewidth=0.9,
                markersize=2.8,
                markeredgecolor="#303030",
                markeredgewidth=0.3,
                **STYLES[dataset],
            )
        style_axis(ax)
        span = x_values[-1] - x_values[0]
        ax.set_xlim(x_values[0] - 0.08 * span, x_values[-1] + 0.08 * span)
        ax.set_ylim(0.8, 1.8)
        ax.set_xticks(x_values)
        ax.set_yticks((1.0, 1.4, 1.8))
        ax.axhline(1, color="#555555", linestyle="--", linewidth=0.55, zorder=0)
        ax.set_xlabel(panel_title, fontsize=6.4, labelpad=2)
        ax.tick_params(pad=1.5)
        ax.legend(
            loc="upper center", ncol=4, frameon=False, fontsize=5.1,
            handlelength=0.9, handletextpad=0.15, columnspacing=0.25,
            borderaxespad=0.15,
        )

    for ax in axes[:, 0]:
        ax.set_ylabel("Speedup ($\\times$)", fontsize=6.1, labelpad=1)
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.08, top=0.985, wspace=0.25, hspace=0.35)
    figure_path = output / "dynamic_sensitivity_2x2.pdf"
    save_figure(fig, figure_path, tight=True)
    print(f"Saved {figure_path}, PNG companion, and {audit_path}")


if __name__ == "__main__":
    main()
