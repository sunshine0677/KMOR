#!/usr/bin/env python3
"""Plot paired speedup sensitivity to request arrivals on four networks."""

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


DATASETS = ("OL", "TG", "CD", "NY")
SCENARIOS = ("request_sparse", "baseline", "request_dense")
STYLES = {
    "OL": {"color": "#174EA6", "linestyle": "-", "marker": "^"},
    "TG": {"color": "#187A32", "linestyle": "--", "marker": "v"},
    "CD": {"color": "#E58A32", "linestyle": "-.", "marker": "o"},
    "NY": {"color": "#777777", "linestyle": ":", "marker": "*"},
}
OUTPUT_FIELDS = (
    "dataset",
    "arrival_count",
    "scenario",
    "seed_count",
    "event_count",
    "mean_ei_time_s",
    "mean_full_time_s",
    "paired_speedup",
    "concrete_topk_matches",
    "source_batch",
    "code_fingerprint",
)


def summarize(workdir: Path, source_dir: Path) -> list[dict[str, object]]:
    metadata = json.loads((source_dir / "run_metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((workdir / "dynamic_scenarios/manifest.json").read_text(encoding="utf-8"))
    arrivals = {
        item["name"]: int(item["arrival_count"])
        for item in manifest["scenarios"]
        if item["name"] in SCENARIOS
    }
    if set(arrivals.values()) != {6, 12, 18}:
        raise ValueError("Expected request-arrival settings 6, 12, and 18")
    totals: dict[tuple[str, str], dict[str, object]] = {}
    seen: set[tuple[str, ...]] = set()
    with (source_dir / "dynamic_paired_event_results.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        for row in csv.DictReader(handle):
            scenario = row["scenario"]
            if scenario not in SCENARIOS:
                continue
            dataset = row["dataset"]
            if dataset not in DATASETS or row["experiment_variant"] != "FullIncremental":
                raise ValueError(f"Unexpected sensitivity record: {dataset} {scenario}")
            if row["status"] != "COMPLETE" or row["concrete_topk_match"] != "True":
                raise ValueError(f"Incomplete or mismatched event: {dataset} {scenario}")
            key = (scenario, dataset, row["seed"], row["event_index"], row["event_type"])
            if key in seen:
                raise ValueError(f"Duplicate event: {key}")
            seen.add(key)
            ei_time = float(row["incremental_time"])
            full_time = float(row["full_recompute_time"])
            if not all(math.isfinite(value) and value > 0 for value in (ei_time, full_time)):
                raise ValueError(f"Invalid paired time: {dataset} {scenario}")
            item = totals.setdefault(
                (dataset, scenario),
                {"ei_total": 0.0, "full_total": 0.0, "event_count": 0, "seeds": set()},
            )
            item["ei_total"] += ei_time
            item["full_total"] += full_time
            item["event_count"] += 1
            item["seeds"].add(row["seed"])

    results: list[dict[str, object]] = []
    for dataset in DATASETS:
        for scenario in SCENARIOS:
            item = totals[(dataset, scenario)]
            if len(item["seeds"]) != 10:
                raise ValueError(f"Expected ten seeds for {dataset} {scenario}")
            count = item["event_count"]
            results.append(
                {
                    "dataset": dataset,
                    "arrival_count": arrivals[scenario],
                    "scenario": scenario,
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
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("paper_results_dynamic_sensitivity_certificate_final_v2"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("paper_results_dynamic_section_f_figures"))
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    source = args.source_dir if args.source_dir.is_absolute() else workdir / args.source_dir
    output = args.output_dir if args.output_dir.is_absolute() else workdir / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rows = summarize(workdir, source)
    csv_path = output / "dynamic_arrival_count_4_plot_data.csv"
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
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(2.76, 1.8))
    for dataset in DATASETS:
        points = sorted(
            (row for row in rows if row["dataset"] == dataset),
            key=lambda row: row["arrival_count"],
        )
        ax.plot(
            [row["arrival_count"] for row in points],
            [row["paired_speedup"] for row in points],
            label=dataset,
            linewidth=1.05,
            markersize=3.4,
            markeredgecolor="#303030",
            markeredgewidth=0.35,
            **STYLES[dataset],
        )
    style_axis(ax)
    ax.set_xlim(5, 19)
    ax.set_ylim(0.8, 1.8)
    ax.set_xticks((6, 12, 18))
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
    path = output / "dynamic_arrival_count_4_paired_speedup.pdf"
    save_figure(fig, path, tight=True)
    print(f"Saved {path}, PNG companion, and {csv_path}")


if __name__ == "__main__":
    main()
