#!/usr/bin/env python3
"""Plot Top-k size and traffic-factor sensitivity from formal paired events."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_dynamic_external_eight import save_figure, style_axis
from plot_dynamic_sensitivity_arrivals import DATASETS, STYLES


OUTPUT_FIELDS = (
    "dataset",
    "parameter",
    "value",
    "seed_count",
    "event_count",
    "mean_ei_time_s",
    "mean_full_time_s",
    "paired_speedup",
    "concrete_topk_matches",
    "source_batch",
    "code_fingerprint",
)


def summarize(
    source_dir: Path,
    values: tuple[int | float, ...],
    parameter: str,
    select_value,
) -> list[dict[str, object]]:
    metadata = json.loads((source_dir / "run_metadata.json").read_text(encoding="utf-8"))
    totals: dict[tuple[str, int | float], dict[str, object]] = {}
    seen: set[tuple[str, ...]] = set()
    with (source_dir / "dynamic_paired_event_results.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        for row in csv.DictReader(handle):
            value = select_value(row)
            if value is None:
                continue
            dataset = row["dataset"]
            if dataset not in DATASETS or value not in values:
                raise ValueError(f"Unexpected setting: {dataset} {value}")
            if row["experiment_variant"] != "FullIncremental":
                raise ValueError(f"Unexpected method for {dataset} {value}")
            if row["status"] != "COMPLETE" or row["concrete_topk_match"] != "True":
                raise ValueError(f"Incomplete or mismatched event: {dataset} {value}")
            key = (
                dataset,
                str(value),
                row["seed"],
                row["event_index"],
                row["event_type"],
            )
            if key in seen:
                raise ValueError(f"Duplicate event: {key}")
            seen.add(key)
            ei_time = float(row["incremental_time"])
            full_time = float(row["full_recompute_time"])
            if not all(math.isfinite(time) and time > 0 for time in (ei_time, full_time)):
                raise ValueError(f"Invalid paired time: {dataset} {value}")
            item = totals.setdefault(
                (dataset, value),
                {"ei_total": 0.0, "full_total": 0.0, "event_count": 0, "seeds": set()},
            )
            item["ei_total"] += ei_time
            item["full_total"] += full_time
            item["event_count"] += 1
            item["seeds"].add(row["seed"])

    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for value in values:
            item = totals[(dataset, value)]
            if len(item["seeds"]) != 10:
                raise ValueError(f"Expected ten seeds for {dataset} {value}")
            count = item["event_count"]
            rows.append(
                {
                    "dataset": dataset,
                    "parameter": parameter,
                    "value": value,
                    "seed_count": len(item["seeds"]),
                    "event_count": count,
                    "mean_ei_time_s": item["ei_total"] / count,
                    "mean_full_time_s": item["full_total"] / count,
                    "paired_speedup": item["full_total"] / item["ei_total"],
                    "concrete_topk_matches": count,
                    "source_batch": source_dir.name,
                    "code_fingerprint": metadata["code_fingerprint"],
                }
            )
    return rows


def write_plot(rows: list[dict[str, object]], output: Path, stem: str) -> None:
    csv_path = output / f"{stem}_plot_data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(2.76, 1.8))
    x_values = sorted({row["value"] for row in rows})
    for dataset in DATASETS:
        points = sorted(
            (row for row in rows if row["dataset"] == dataset),
            key=lambda row: row["value"],
        )
        ax.plot(
            [row["value"] for row in points],
            [row["paired_speedup"] for row in points],
            label=dataset,
            linewidth=1.05,
            markersize=3.4,
            markeredgecolor="#303030",
            markeredgewidth=0.35,
            **STYLES[dataset],
        )
    style_axis(ax)
    span = x_values[-1] - x_values[0]
    ax.set_xlim(x_values[0] - 0.08 * span, x_values[-1] + 0.08 * span)
    ax.set_ylim(0.8, 1.8)
    ax.set_xticks(x_values)
    ax.set_yticks((0.8, 1.0, 1.2, 1.4, 1.6, 1.8))
    ax.set_ylabel("Speedup ($\\times$)", fontsize=6.4, labelpad=2)
    ax.axhline(1, color="#555555", linestyle="--", linewidth=0.65, zorder=0)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.55, 1.01),
        ncol=4,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.23, right=0.98, bottom=0.20, top=0.80)
    path = output / f"{stem}_paired_speedup.pdf"
    save_figure(fig, path, tight=True)
    print(f"Saved {path}, PNG companion, and {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("paper_results_dynamic_section_f_figures"))
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else workdir / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    k_rows = summarize(
        workdir / "paper_results_dynamic_k_sensitivity_final",
        (1, 3, 5, 10),
        "k",
        lambda row: int(row["k"]) if row["scenario"] == "baseline" else None,
    )
    write_plot(k_rows, output, "dynamic_topk_size_4")

    manifest = json.loads((workdir / "dynamic_scenarios/manifest.json").read_text(encoding="utf-8"))
    factors = {
        item["name"]: float(item["traffic_factors"][0])
        for item in manifest["scenarios"]
        if item["name"] in {"severity_mild", "baseline", "severity_severe"}
    }
    if set(factors.values()) != {1.2, 1.5, 2.0}:
        raise ValueError("Expected traffic factors 1.2, 1.5, and 2.0")
    factor_rows = summarize(
        workdir / "paper_results_dynamic_sensitivity_certificate_final_v2",
        (1.2, 1.5, 2.0),
        "traffic_factor",
        lambda row: factors.get(row["scenario"]),
    )
    write_plot(factor_rows, output, "dynamic_traffic_factor_4")


if __name__ == "__main__":
    main()
