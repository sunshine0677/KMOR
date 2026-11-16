#!/usr/bin/env python3
"""Plot single-column paired maintenance speedups for eight networks."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_dynamic_external_eight import DATASETS, save_figure, style_axis


def paired_speedups(path: Path) -> dict[str, float]:
    totals = {name: [0.0, 0.0, 0] for name in DATASETS}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["experiment_variant"] != "FullIncremental":
                raise ValueError("The input must contain EI-DPKMOR paired events only")
            name = row["dataset"]
            if name not in totals:
                raise ValueError(f"Unexpected dataset: {name}")
            if row["status"] != "COMPLETE" or row["concrete_topk_match"] != "True":
                raise ValueError(f"Incomplete or mismatched event for {name}")
            ei = float(row["incremental_time"])
            full = float(row["full_recompute_time"])
            if not all(math.isfinite(value) and value > 0 for value in (ei, full)):
                raise ValueError(f"Invalid paired event time for {name}")
            totals[name][0] += ei
            totals[name][1] += full
            totals[name][2] += 1
    if any(count == 0 for _, _, count in totals.values()):
        raise ValueError("Missing events for at least one network")
    return {name: full / ei for name, (ei, full, _) in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("paper_results_dynamic_figures_8"))
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else workdir / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    values = paired_speedups(output / "dynamic_paired_event_results_8.csv")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.45, 2.25))
    bars = ax.bar(
        range(len(DATASETS)),
        [values[name] for name in DATASETS],
        width=0.65,
        color="#174EA6",
        edgecolor="#303030",
        linewidth=0.45,
        zorder=2,
    )
    style_axis(ax)
    ax.set_xlim(-0.55, len(DATASETS) - 0.45)
    ax.set_ylim(0, 3.0)
    ax.set_yticks((0, 1, 2, 3))
    ax.set_xticks(range(len(DATASETS)), DATASETS)
    ax.set_ylabel("Speedup ($\\times$)", fontsize=6.7, labelpad=2)
    ax.axhline(1, color="#555555", linestyle="--", linewidth=0.65, zorder=1)
    ax.axvline(3.5, color="#d0d0d0", linewidth=0.6, zorder=1)
    for bar, name in zip(bars, DATASETS):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.055,
            f"{values[name]:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
        )
    fig.subplots_adjust(left=0.18, right=0.99, bottom=0.16, top=0.91)
    path = output / "dynamic_overall_8_paired_speedup.pdf"
    save_figure(fig, path, tight=True)
    print(f"Saved {path} and PNG companion")
    print(", ".join(f"{name}={values[name]:.3f}x" for name in DATASETS))


if __name__ == "__main__":
    main()
