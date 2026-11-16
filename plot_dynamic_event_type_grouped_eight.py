#!/usr/bin/env python3
"""Plot event-specific paired speedup as one single-column grouped chart."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_dynamic_event_type_speedup_eight import OUTPUT_FIELDS, summarize
from plot_dynamic_external_eight import DATASETS, save_figure, style_axis


SERIES = (
    ("RequestArrival", "Arrival", "#E58A32", "//"),
    ("RequestCancel", "Cancellation", "#187A32", "xx"),
    ("EdgeUpdate", "Edge update", "#174EA6", ""),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("paper_results_dynamic_figures_8"))
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else workdir / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    rows = summarize(output / "dynamic_paired_event_results_8.csv")
    values = {(row["dataset"], row["event_type"]): row["paired_speedup"] for row in rows}
    audit_path = output / "dynamic_event_type_8_grouped_plot_data.csv"
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.1,
        "axes.linewidth": 0.7,
        "hatch.linewidth": 0.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    centers = np.arange(len(DATASETS), dtype=float)
    bar_width = 0.235
    for index, (event_type, label, color, hatch) in enumerate(SERIES):
        offsets = centers + (index - 1) * bar_width
        ax.bar(
            offsets,
            [values[(dataset, event_type)] for dataset in DATASETS],
            width=bar_width * 0.92,
            label=label,
            color=color,
            edgecolor="#303030",
            linewidth=0.35,
            hatch=hatch,
            zorder=2,
        )
    style_axis(ax)
    ax.set_xlim(-0.55, len(DATASETS) - 0.45)
    ax.set_ylim(0, 3.1)
    ax.set_xticks(centers, DATASETS)
    ax.set_yticks((0, 1, 2, 3))
    ax.set_ylabel("Speedup ($\\times$)", fontsize=6.7, labelpad=2)
    ax.axhline(1, color="#555555", linestyle="--", linewidth=0.65, zorder=1)
    ax.axvline(3.5, color="#d0d0d0", linewidth=0.55, zorder=1)
    fig.legend(
        loc="upper center", bbox_to_anchor=(0.52, 0.995), ncol=3,
        frameon=False, handlelength=1.25, handletextpad=0.3, columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.20, right=0.99, bottom=0.16, top=0.83)
    figure_path = output / "dynamic_event_type_8_grouped_speedup.pdf"
    save_figure(fig, figure_path, tight=True)
    print(f"Saved {figure_path}, PNG companion, and {audit_path}")


if __name__ == "__main__":
    main()
