#!/usr/bin/env python3
"""Plot paired dynamic speedup by event type across eight networks."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_dynamic_external_eight import DATASETS, save_figure, style_axis


EVENT_TYPES = (
    ("RequestArrival", "dynamic_arrival_8_paired_speedup.pdf"),
    ("RequestCancel", "dynamic_cancellation_8_paired_speedup.pdf"),
    ("EdgeUpdate", "dynamic_edge_update_8_paired_speedup.pdf"),
)
OUTPUT_FIELDS = (
    "event_type",
    "dataset",
    "source_batch",
    "code_fingerprint",
    "event_count",
    "mean_ei_time_s",
    "mean_full_time_s",
    "paired_speedup",
)


def summarize(path: Path) -> list[dict[str, object]]:
    totals: dict[tuple[str, str], dict[str, object]] = {}
    valid_types = {name for name, _ in EVENT_TYPES}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            event_type = row["event_type"]
            dataset = row["dataset"]
            if event_type not in valid_types or dataset not in DATASETS:
                raise ValueError(f"Unexpected event record: {dataset} {event_type}")
            if row["experiment_variant"] != "FullIncremental":
                raise ValueError("Expected EI-DPKMOR paired event records only")
            if row["status"] != "COMPLETE" or row["concrete_topk_match"] != "True":
                raise ValueError(f"Incomplete or mismatched event: {dataset} {event_type}")
            ei_time = float(row["incremental_time"])
            full_time = float(row["full_recompute_time"])
            if not all(math.isfinite(value) and value > 0 for value in (ei_time, full_time)):
                raise ValueError(f"Invalid paired time: {dataset} {event_type}")
            key = (event_type, dataset)
            item = totals.setdefault(
                key,
                {
                    "source_batch": row["source_batch"],
                    "code_fingerprint": row["code_fingerprint"],
                    "event_count": 0,
                    "ei_total": 0.0,
                    "full_total": 0.0,
                },
            )
            if (
                item["source_batch"] != row["source_batch"]
                or item["code_fingerprint"] != row["code_fingerprint"]
            ):
                raise ValueError(f"Mixed provenance for {dataset} {event_type}")
            item["event_count"] += 1
            item["ei_total"] += ei_time
            item["full_total"] += full_time

    if len(totals) != len(EVENT_TYPES) * len(DATASETS):
        raise ValueError("Expected all three event types on all eight networks")
    rows: list[dict[str, object]] = []
    for event_type, _ in EVENT_TYPES:
        for dataset in DATASETS:
            item = totals[(event_type, dataset)]
            count = item["event_count"]
            rows.append(
                {
                    "event_type": event_type,
                    "dataset": dataset,
                    "source_batch": item["source_batch"],
                    "code_fingerprint": item["code_fingerprint"],
                    "event_count": count,
                    "mean_ei_time_s": item["ei_total"] / count,
                    "mean_full_time_s": item["full_total"] / count,
                    "paired_speedup": item["full_total"] / item["ei_total"],
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("paper_results_dynamic_figures_8"))
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else workdir / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rows = summarize(output / "dynamic_paired_event_results_8.csv")
    csv_path = output / "dynamic_event_type_8_paired_speedup.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    values = {(row["event_type"], row["dataset"]): row["paired_speedup"] for row in rows}
    for event_type, filename in EVENT_TYPES:
        fig, ax = plt.subplots(figsize=(3.45, 2.25))
        bars = ax.bar(
            range(len(DATASETS)),
            [values[(event_type, dataset)] for dataset in DATASETS],
            width=0.65,
            color="#174EA6",
            edgecolor="#303030",
            linewidth=0.45,
            zorder=2,
        )
        style_axis(ax)
        ax.set_xlim(-0.55, len(DATASETS) - 0.45)
        ax.set_ylim(0, 3.1)
        ax.set_xticks(range(len(DATASETS)), DATASETS)
        ax.set_yticks((0, 1, 2, 3))
        ax.set_ylabel("Speedup ($\\times$)", fontsize=6.7, labelpad=2)
        ax.axhline(1, color="#555555", linestyle="--", linewidth=0.65, zorder=1)
        ax.axvline(3.5, color="#d0d0d0", linewidth=0.6, zorder=1)
        for bar, dataset in zip(bars, DATASETS):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.055,
                f"{values[(event_type, dataset)]:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.2,
            )
        fig.subplots_adjust(left=0.18, right=0.99, bottom=0.16, top=0.91)
        path = output / filename
        save_figure(fig, path, tight=True)
        print(f"Saved {path} and PNG companion")
    print(f"Audit data: {csv_path}")


if __name__ == "__main__":
    main()
