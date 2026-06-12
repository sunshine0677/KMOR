#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run paper-ready KMOR experiments and generate figures.

Experiments:
1. Practical baseline comparison:
   KMOR vs GreedyInsertion vs NearestInsertion vs BestInsertionSolomon
2. Ablation study:
   Full vs NoPruning vs NoDP

Figure style: pure white background, fixed size for ICDE two-column layout.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

from Dpkmor import parse_requests_file, read_edge_list
from kmor_effectiveness import run_single_comparison_with_timeout
from kmor_experiments import run_ablation_triplet


@dataclass
class DatasetConfig:
    name: str
    edge_file: str
    requests_file: str
    source: str
    target: str
    time_budget: float
    ignore_first_col: bool
    directed: bool
    num_requests: int
    shortest_path_cost: float


METHOD_COLORS = {
    "KMOR":{"color": "#333333", "hatch": ""},                  # 最黑
    "GreedyInsertion":{"color": "#999999", "hatch": "///"},      # 中灰
    "NearestInsertion":{"color": "#cccccc", "hatch": "\\\\"},  # 浅灰
    "BestInsertionSolomon":{"color": "#ffffff", "hatch": "xxx"},       # 深灰
}

VARIANT_COLORS = {
    "Full":       {"color": "#333333", "hatch": ""},
    "NoPruning":  {"color": "#999999", "hatch": "///"},
    "NoDP":       {"color": "#cccccc", "hatch": "\\\\"},
}


# ---- 与参数敏感性实验保持一致的视觉风格 ----
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.title_fontsize": 8,
        "figure.titlesize": 10,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _sanitize_metric_filename(metric_label: str) -> str:
    safe = metric_label.lower().replace(" ", "_")
    safe = safe.replace("(", "").replace(")", "").replace("/", "_per_")
    return safe.replace("-", "_")


def _to_plot_value(value: object) -> float:
    if value == float("inf"):
        return np.nan
    return float(value)


def _grouped_bar_figure(
    rows, datasets, series_names, series_key, metric_key,
    metric_label, style_dict, out_base,
    log_scale=False, figsize=(3.5, 2.5),
    ylim=None, yticks=None,
):
    plt.style.use("seaborn-v0_8-white")
    fig = plt.figure(figsize=figsize, facecolor="white")
    is_wide = figsize[0] > 5.0
    left, width_ax = (0.08, 0.88) if is_wide else (0.18, 0.78)
    ax = fig.add_axes([left, 0.18, width_ax, 0.72])
    ax.set_facecolor("white")

    x = np.arange(len(datasets))
    bar_width = 0.8 / max(1, len(series_names))

    for idx, name in enumerate(series_names):
        vals = [_to_plot_value(
            next(item for item in rows if item["dataset"] == ds and item[series_key] == name)[metric_key]
        ) for ds in datasets]
        pos = x + (idx - (len(series_names) - 1) / 2.0) * bar_width

        # 兼容两种格式：纯颜色字符串，或带 hatch 的字典
        style = style_dict[name]
        if isinstance(style, dict):
            color = style["color"]
            hatch = style.get("hatch", "")
        else:
            color = style
            hatch = ""

        ax.bar(pos, vals, width=bar_width,
               color=color,
               edgecolor="black", linewidth=0.7,
               hatch=hatch,
               label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontweight="semibold")
    ax.set_ylabel("")
    ax.set_title(metric_label, fontweight="bold", pad=8)
    ax.tick_params(axis="x", labelsize=8, pad=4)
    ax.tick_params(axis="y", labelsize=8, pad=4)

    if log_scale:
        ax.set_yscale("log")
        if ylim: ax.set_ylim(*ylim)
        if yticks:
            ax.set_yticks(yticks)
            ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.margins(y=0.05)
    else:
        ax.margins(y=0.12)

    leg = ax.legend(frameon=True, fancybox=False, edgecolor="black",
                    borderpad=0.4, handlelength=1.4, loc="best", fontsize=7.5)
    leg.get_frame().set_linewidth(0.7)
    fig.savefig(out_base.with_suffix(".pdf"), pad_inches=0.05)
    plt.close(fig)


def load_configs(path: Path) -> List[DatasetConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [DatasetConfig(**item) for item in payload["datasets"]]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str] = None) -> None:
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_effectiveness(
    workdir: Path,
    configs: Sequence[DatasetConfig],
    capacity: int,
    k: int,
    detour_ratio: float,
    max_states: int,
    timeout: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for cfg in configs:
        input_path = workdir / cfg.edge_file
        requests_path = workdir / cfg.requests_file
        results = run_single_comparison_with_timeout(
            str(input_path),
            str(requests_path),
            cfg.source,
            cfg.target,
            "space",
            cfg.directed,
            cfg.ignore_first_col,
            {
                "capacity": capacity,
                "k": k,
                "time_budget": cfg.time_budget,
                "detour_ratio": detour_ratio,
                "max_states": max_states,
            },
            timeout,
        )
        for item in results:
            cost_per_served = item.total_cost / item.served_count if item.served_count > 0 else float("inf")
            rows.append(
                {
                    "dataset": cfg.name,
                    "method": item.method,
                    "status": "OK" if item.feasible else "INF",
                    "served": item.served_count,
                    "cost": item.total_cost,
                    "avg_wait": item.avg_wait,
                    "max_wait": item.max_wait,
                    "preprocess_time": item.preprocess_time,
                    "solve_time": item.solve_time,
                    "total_time": item.runtime,
                    "cost_per_served": cost_per_served,
                }
            )
    return rows


def run_ablation(
    workdir: Path,
    configs: Sequence[DatasetConfig],
    capacity: int,
    k: int,
    detour_ratio: float,
    max_states: int,
    timeout: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for cfg in configs:
        input_path = workdir / cfg.edge_file
        requests_path = workdir / cfg.requests_file
        graph = read_edge_list(str(input_path), directed=cfg.directed, ignore_first_col=cfg.ignore_first_col)
        requests = parse_requests_file(str(requests_path))
        common_params = {
            "capacity": capacity,
            "k": k,
            "time_budget": cfg.time_budget,
            "detour_ratio": detour_ratio,
            "max_states": max_states,
            "aggressive_served_pruning": True,
        }

        aligned_results = run_ablation_triplet(
            graph,
            cfg.source,
            cfg.target,
            requests,
            timeout,
            input_path=str(input_path),
            delimiter="space",
            ignore_first_col=cfg.ignore_first_col,
            directed=cfg.directed,
            align_served_to_full=True,
            **common_params,
        )
        for result in aligned_results:
            variant_name = result.method
            rows.append(
                {
                    "dataset": cfg.name,
                    "variant": variant_name,
                    "served": result.served_count,
                    "search_time": result.search_time,
                    "examined_partial_routes": result.expanded_states,
                }
            )
    return rows


def plot_effectiveness(rows: Sequence[Dict[str, object]], out_dir: Path) -> None:
    datasets = sorted({str(row["dataset"]) for row in rows})
    methods = ["KMOR", "GreedyInsertion", "NearestInsertion", "BestInsertionSolomon"]
    metric_specs = [
        ("cost_per_served", "Cost Per Served", True),
    ]

    for metric_key, metric_label, log_scale in metric_specs:
        _grouped_bar_figure(
            rows,
            datasets,
            methods,
            "method",
            metric_key,
            metric_label,
            METHOD_COLORS,
            out_dir / f"effectiveness_{_sanitize_metric_filename(metric_label)}",
            log_scale=log_scale,
            figsize=(7, 4.0),
            ylim=(1000, 60000),
            yticks=[2000, 5000, 10000, 20000, 50000],
        )


def plot_ablation(rows: Sequence[Dict[str, object]], out_dir: Path) -> None:
    datasets = sorted({str(row["dataset"]) for row in rows})
    variants = ["Full", "NoPruning", "NoDP"]
    metric_specs = [
        ("search_time", "Search Time (s)", True, None, None),
        ("examined_partial_routes", "Examined Partial Routes", True, (1000, 170000), [2000, 5000, 10000, 20000, 50000, 100000, 150000]),
    ]

    for metric_key, metric_label, log_scale, ylim, yticks in metric_specs:
        _grouped_bar_figure(
            rows,
            datasets,
            variants,
            "variant",
            metric_key,
            metric_label,
            VARIANT_COLORS,
            out_dir / f"ablation_{_sanitize_metric_filename(metric_label)}",
            log_scale=log_scale,
            figsize=(3.5, 2.5),
            ylim=ylim,
            yticks=yticks,
        )

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run paper-ready KMOR experiments")
    ap.add_argument("--workdir", default=".", help="Workspace directory")
    ap.add_argument("--config", default="paper_datasets.json", help="Dataset config JSON")
    ap.add_argument("--output-dir", default="paper_results", help="Directory for CSV and figures")
    ap.add_argument("--capacity", type=int, default=4)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--detour-ratio", type=float, default=1.4)
    ap.add_argument("--max-states", type=int, default=30000)
    ap.add_argument("--ablation-max-states", type=int, default=100000)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--only", choices=["effectiveness", "ablation", "all", "plot"], default="all")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    workdir = Path(args.workdir).resolve()
    out_dir = workdir / args.output_dir
    ensure_dir(out_dir)

    if args.only == "plot":
        eff_path = out_dir / "effectiveness_results.csv"
        abl_path = out_dir / "ablation_results.csv"
        if eff_path.exists():
            plot_effectiveness(read_csv(eff_path), out_dir)
            print(f"Saved effectiveness figures to {out_dir}")
        if abl_path.exists():
            plot_ablation(read_csv(abl_path), out_dir)
            print(f"Saved ablation figures to {out_dir}")
        return 0

    configs = load_configs(workdir / args.config)

    if args.only in {"effectiveness", "all"}:
        eff_rows = run_effectiveness(
            workdir,
            configs,
            args.capacity,
            args.k,
            args.detour_ratio,
            args.max_states,
            args.timeout,
        )
        write_csv(out_dir / "effectiveness_results.csv", eff_rows)
        plot_effectiveness(eff_rows, out_dir)
        print(f"Saved effectiveness CSV and figures to {out_dir}")

    if args.only in {"ablation", "all"}:
        abl_rows = run_ablation(
            workdir,
            configs,
            args.capacity,
            args.k,
            args.detour_ratio,
            args.ablation_max_states,
            args.timeout,
        )
        write_csv(out_dir / "ablation_results.csv", abl_rows)
        plot_ablation(abl_rows, out_dir)
        print(f"Saved ablation CSV and figures to {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())