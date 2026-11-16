#!/usr/bin/env python3
"""Plot eight-network dynamic baselines from their paired event records."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np


DATASETS = ("OL", "TG", "CD", "NY", "BAY", "COL", "FLA", "NW")
METHODS = (
    ("EI-DPKMOR", "FullIncremental", "#174EA6", "///"),
    ("KineticTree-TopK", "KineticTreeTopK", "#187A32", "\\\\"),
    ("TD-TopK-Insertion", "TDTopKInsertion", "#E58A32", "xx"),
    ("RHLS-TopK", "RHLSTopK", "#777777", "..."),
)
AVAILABILITY_SERIES = (
    ("TD-TopK-Insertion", "TD-TopK", "#ffffff", "", "#555555"),
    ("KineticTree-TopK", "KineticTree", "#eef8ef", "xxxx", "#55a86c"),
    ("RHLS-TopK", "RHLS", "#fff0eb", "xx", "#df947b"),
    ("EI-DPKMOR", "EI-DPKMOR", "#666666", "", "#555555"),
)
OLD_BATCHES = {
    "FullIncremental": "paper_results_dynamic_certificate_final",
    "KineticTreeTopK": "paper_results_dynamic_external_baselines_final",
    "TDTopKInsertion": "paper_results_dynamic_td_topk_final",
    "RHLSTopK": "paper_results_dynamic_rhls_final",
}
NEW_BATCH = "paper_results_dynamic_dimacs_external_final"
OUTPUT_FIELDS = (
    "dataset",
    "method",
    "batch",
    "code_fingerprint",
    "events",
    "complete_events",
    "method_mean_time_s",
    "paired_full_mean_time_s",
    "paired_speedup",
    "concrete_topk_matches",
    "concrete_topk_match_rate_percent",
    "mean_formal_routes",
    "mean_available_routes",
    "top5_available_events",
    "top5_plan_availability_percent",
    "shortfall_when_full_has_k",
)


def read_batch(workdir: Path, name: str) -> tuple[list[dict[str, str]], str]:
    directory = workdir / name
    with (directory / "run_metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if int(metadata["arguments"]["k"]) != 5:
        raise ValueError(f"Expected k=5 in {name} metadata")
    fingerprint = metadata["code_fingerprint"]
    with (directory / "dynamic_paired_event_results.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        return list(csv.DictReader(handle)), fingerprint


def event_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row["seed"],
        row["event_index"],
        row["event_type"],
        row["event_time"],
    )


def summarize(workdir: Path) -> list[dict[str, object]]:
    batches = {
        name: read_batch(workdir, name)
        for name in set(OLD_BATCHES.values()) | {NEW_BATCH}
    }
    results: list[dict[str, object]] = []
    for dataset in DATASETS:
        reference_keys = None
        for method, variant, _, _ in METHODS:
            batch = OLD_BATCHES[variant] if dataset in DATASETS[:4] else NEW_BATCH
            all_rows, fingerprint = batches[batch]
            rows = [
                row
                for row in all_rows
                if row["dataset"] == dataset
                and row["experiment_variant"] == variant
                and row["scenario"] == "baseline"
            ]
            if not rows:
                raise ValueError(f"Missing {dataset} {method} event records in {batch}")
            keys = [event_key(row) for row in rows]
            if len(keys) != len(set(keys)):
                raise ValueError(f"Duplicate events for {dataset} {method}")
            if reference_keys is None:
                reference_keys = set(keys)
            elif set(keys) != reference_keys:
                raise ValueError(f"Unpaired event keys for {dataset} {method}")

            complete = [row for row in rows if row["status"] == "COMPLETE"]
            if not complete:
                raise ValueError(f"No COMPLETE events for {dataset} {method}")
            if any("k" in row and int(row["k"]) != 5 for row in complete):
                raise ValueError(f"Expected k=5 for {dataset} {method}")
            if any(
                (row["incremental_available_topk_complete"] == "True")
                != (int(row["incremental_available_route_count"]) >= 5)
                for row in complete
            ):
                raise ValueError(f"Inconsistent availability fields for {dataset} {method}")
            method_total = sum(float(row["incremental_time"]) for row in complete)
            full_total = sum(float(row["full_recompute_time"]) for row in complete)
            if method_total <= 0 or not math.isfinite(full_total):
                raise ValueError(f"Invalid paired timing for {dataset} {method}")
            matches = sum(row["concrete_topk_match"] == "True" for row in complete)
            shortfall = sum(
                int(row["full_route_count"]) >= 5
                and int(row["incremental_route_count"]) < 5
                for row in complete
            )
            results.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "batch": batch,
                    "code_fingerprint": fingerprint,
                    "events": len(rows),
                    "complete_events": len(complete),
                    "method_mean_time_s": method_total / len(complete),
                    "paired_full_mean_time_s": full_total / len(complete),
                    "paired_speedup": full_total / method_total,
                    "concrete_topk_matches": matches,
                    "concrete_topk_match_rate_percent": 100 * matches / len(complete),
                    "mean_formal_routes": sum(
                        int(row["incremental_route_count"]) for row in complete
                    )
                    / len(complete),
                    "mean_available_routes": sum(
                        int(row["incremental_available_route_count"])
                        for row in complete
                    )
                    / len(complete),
                    "top5_available_events": sum(
                        row["incremental_available_topk_complete"] == "True"
                        for row in complete
                    ),
                    "top5_plan_availability_percent": 100
                    * sum(
                        row["incremental_available_topk_complete"] == "True"
                        for row in complete
                    )
                    / len(complete),
                    "shortfall_when_full_has_k": shortfall,
                }
            )
    return results


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.3,
            "axes.linewidth": 0.7,
            "hatch.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", direction="out", length=2.5, width=0.7)


def grouped_bars(
    ax: plt.Axes,
    values: dict[tuple[str, str], float],
    datasets: tuple[str, ...] = DATASETS,
    series: tuple[tuple[str, str, str, str, str], ...] | None = None,
) -> list[object]:
    if series is None:
        series = tuple(
            (name, name, color, hatch, "#303030")
            for name, _, color, hatch in METHODS
        )
    x = np.arange(len(datasets), dtype=float)
    width = 0.19
    handles = []
    for index, (name, label, color, hatch, edgecolor) in enumerate(series):
        heights = [values[(dataset, name)] for dataset in datasets]
        bars = ax.bar(
            x + (index - (len(series) - 1) / 2) * width,
            heights,
            width=width * 0.94,
            color=color,
            edgecolor=edgecolor,
            linewidth=0.45,
            hatch=hatch,
            zorder=2,
            label=label,
        )
        handles.append(bars[0])
    ax.set_xlim(-0.58, len(datasets) - 0.42)
    ax.set_xticks(x, datasets)
    if len(datasets) == len(DATASETS):
        ax.axvline(3.5, color="#c8c8c8", linewidth=0.65, zorder=1)
    return handles


def plot_speedup(values: dict[tuple[str, str], float], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.16, 2.9))
    x = np.arange(len(DATASETS), dtype=float)
    markers = ("^", "s", "o", "D")
    handles = []
    for index, (name, _, color, _) in enumerate(METHODS):
        handles.append(
            ax.scatter(
                x + (index - 1.5) * 0.16,
                [values[(dataset, name)] for dataset in DATASETS],
                s=34,
                marker=markers[index],
                color=color,
                edgecolors="#303030",
                linewidths=0.5,
                zorder=3,
                label=name,
            )
        )
    style_axis(ax)
    ax.set_xlim(-0.6, len(DATASETS) - 0.4)
    ax.set_xticks(x, DATASETS)
    ax.set_yscale("log")
    ax.set_ylim(0.4, 25)
    ax.set_yticks((0.5, 1, 2, 5, 10, 20))
    ax.set_yticklabels(("0.5", "1", "2", "5", "10", "20"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.axhline(1, color="#555555", linestyle="--", linewidth=0.65, zorder=1)
    ax.axvline(3.5, color="#c8c8c8", linewidth=0.65, zorder=1)
    ax.set_ylabel("Speedup over Paired Full-DPKMOR (log scale)")
    fig.legend(
        handles,
        [method[0] for method in METHODS],
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.125, right=0.992, bottom=0.16, top=0.83)
    save_figure(fig, output)


def plot_match(values: dict[tuple[str, str], float], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.16, 2.65))
    grouped_bars(ax, values)
    style_axis(ax)
    ax.set_ylim(0, 106)
    ax.set_yticks((0, 25, 50, 75, 100))
    ax.axhline(100, color="#555555", linestyle="--", linewidth=0.65, zorder=1)
    ax.set_ylabel("Concrete Top-k Match Rate (%)")
    handles = [ax.containers[index][0] for index in range(len(METHODS))]
    fig.legend(
        handles,
        [method[0] for method in METHODS],
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        handlelength=1.55,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.112, right=0.992, bottom=0.16, top=0.82)
    save_figure(fig, output)


def plot_top5_availability(values: dict[tuple[str, str], float], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.16, 2.65))
    handles = grouped_bars(ax, values, series=AVAILABILITY_SERIES)
    style_axis(ax)
    ax.set_ylim(0, 100)
    ax.set_yticks((0, 25, 50, 75, 100))
    ax.set_ylabel("Top-5 Plan Availability (%)")
    fig.legend(
        handles,
        [item[1] for item in AVAILABILITY_SERIES],
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        handlelength=1.55,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.112, right=0.992, bottom=0.16, top=0.82)
    save_figure(fig, output)


def save_figure(fig: plt.Figure, path: Path, *, tight: bool = False) -> None:
    crop = {"bbox_inches": "tight", "pad_inches": 0.02} if tight else {}
    fig.savefig(path, facecolor="white", **crop)
    fig.savefig(path.with_suffix(".png"), dpi=300, facecolor="white", **crop)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("paper_results_dynamic_figures_8"))
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    output = args.output_dir
    if not output.is_absolute():
        output = workdir / output
    output.mkdir(parents=True, exist_ok=True)
    rows = summarize(workdir)
    with (output / "dynamic_external_8_plot_data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    configure_style()
    plot_speedup(
        {(row["dataset"], row["method"]): row["paired_speedup"] for row in rows},
        output / "dynamic_external_8_speedup.pdf",
    )
    plot_match(
        {
            (row["dataset"], row["method"]): row["concrete_topk_match_rate_percent"]
            for row in rows
        },
        output / "dynamic_external_8_concrete_topk_match.pdf",
    )
    plot_top5_availability(
        {
            (row["dataset"], row["method"]): row["top5_plan_availability_percent"]
            for row in rows
        },
        output / "dynamic_external_8_top5_plan_availability.pdf",
    )
    print(f"Saved three PDF/PNG figures and audit CSV to {output}")


if __name__ == "__main__":
    main()
