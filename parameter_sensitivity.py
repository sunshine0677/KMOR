#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter sensitivity experiments for the final DPKMOR algorithm.
(plotting configuration adjusted for ICDE two‑column layout)

Generates compact PDF figures that fit cleanly into a single column
without further scaling, using appropriate font sizes, line widths,
and marker styles.
"""

from __future__ import annotations


import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator,MultipleLocator

from Dpkmor import Graph, Request, read_edge_list
from kmor_experiments import ExperimentResult, run_kmor_variant
from prepare_paper_datasets import (
    make_competitive_request,
    make_core_request,
    make_reverse_request,
    path_prefix_costs,
)

# ---------------------------------------------------------------------------
# Default parameter values (unchanged)
# ---------------------------------------------------------------------------
DEFAULT_POOL_VALUES = [1000, 2000, 3000, 4000, 5000]
DEFAULT_K_VALUES = [10, 20, 30, 40, 50]
DEFAULT_DEADLINE_VALUES = [5, 10, 15, 20, 25]
DEFAULT_CAPACITY_VALUES = [3, 4, 6, 10, 20]

DEFAULT_POOL_SIZE = 3000
DEFAULT_K = 30
DEFAULT_DEADLINE = 10
DEFAULT_CAPACITY = 4

DATASET_ORDER = ["OL", "TG", "NY", "CD"]

# ---------------------------------------------------------------------------
# Visual style – tuned for ICDE column width (≈ 3.5 in)
# ---------------------------------------------------------------------------
LINE_STYLES = {
    "OL": {"color": "#7f7f7f", "marker": "s", "linestyle": "-"},
    "TG": {"color": "#7f7f7f", "marker": "D", "linestyle": "--"},
    "NY": {"color": "#222222", "marker": "o", "linestyle": "-"},
    "CD": {"color": "#222222", "marker": "^", "linestyle": "--"},
}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 10,
        "axes.linewidth": 0.8,
        "lines.markersize": 5,
        "lines.linewidth": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# ---------------------------------------------------------------------------
# Data structures (unchanged)
# ---------------------------------------------------------------------------
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


@dataclass
class DatasetRuntime:
    config: DatasetConfig
    graph: Graph
    reference_path: List[str]
    shortest_cost: float
    shortest_path_cache: Dict[Tuple[str, str], Tuple[List[str], float]]


# ---------------------------------------------------------------------------
# Helper functions (unchanged)
# ---------------------------------------------------------------------------
def parse_int_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def load_configs(path: Path) -> List[DatasetConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [DatasetConfig(**item) for item in payload["datasets"]]


def dataset_sort_key(name: str) -> Tuple[int, str]:
    if name in DATASET_ORDER:
        return DATASET_ORDER.index(name), name
    return len(DATASET_ORDER), name


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def request_tuple_to_request(item: Tuple[str, str, str, int, float, float]) -> Request:
    name, pickup, dropoff, demand, request_time, wait_limit = item
    return Request(
        name=str(name),
        pickup=str(pickup),
        dropoff=str(dropoff),
        demand=int(demand),
        request_time=float(request_time),
        wait_limit=float(wait_limit),
    )


def generate_request_pool(
    graph: Graph,
    path: Sequence[str],
    shortest_cost: float,
    pool_size: int,
    deadline_minutes: int,
    rng: random.Random,
) -> List[Tuple[str, str, str, int, float, float]]:
    prefix_costs = path_prefix_costs(graph, path)
    wait_scale = max(0.25, deadline_minutes / DEFAULT_DEADLINE)
    requests: List[Tuple[str, str, str, int, float, float]] = []

    for idx in range(1, pool_size + 1):
        x = rng.random()
        if x < 0.42:
            item = make_core_request(idx, path, prefix_costs, shortest_cost, rng, wait_scale=wait_scale)
        elif x < 0.86:
            item = make_competitive_request(idx, path, prefix_costs, shortest_cost, rng, wait_scale=wait_scale)
        else:
            item = make_reverse_request(idx, path, prefix_costs, shortest_cost, rng, wait_scale=wait_scale)
        requests.append(item)

    return requests


def active_count_for_pool(pool_size: int, active_limit: int) -> int:
    return min(active_limit, max(8, 8 + pool_size // 500))


def select_active_requests(
    pool: Sequence[Tuple[str, str, str, int, float, float]],
    path: Sequence[str],
    pool_size: int,
    active_limit: int,
    rng: random.Random,
) -> List[Request]:
    path_index = {node: idx for idx, node in enumerate(path)}
    n = max(1, len(path))
    target_count = min(len(pool), active_count_for_pool(pool_size, active_limit))

    scored: List[Tuple[Tuple[float, float, float, float, float], Tuple[str, str, str, int, float, float]]] = []
    for item in pool:
        _, pickup, dropoff, demand, request_time, wait_limit = item
        pickup_idx = path_index.get(pickup)
        dropoff_idx = path_index.get(dropoff)
        if pickup_idx is None or dropoff_idx is None:
            continue
        span = (dropoff_idx - pickup_idx) / n
        forward_penalty = 0.0 if span > 0 else 2.0
        useful_span = abs(span - 0.30)
        demand_penalty = 0.12 * max(0, int(demand) - 1)
        urgency = request_time / max(1.0, wait_limit)
        jitter = rng.random() * 0.05
        score = (forward_penalty, useful_span, demand_penalty, urgency, jitter)
        scored.append((score, item))

    if not scored:
        return []

    scored.sort(key=lambda pair: pair[0])
    selected = [request_tuple_to_request(item) for _, item in scored[:target_count]]
    for idx, req in enumerate(selected, start=1):
        selected[idx - 1] = Request(
            name=f"s{idx}_{req.name}",
            pickup=req.pickup,
            dropoff=req.dropoff,
            demand=req.demand,
            request_time=req.request_time,
            wait_limit=req.wait_limit,
        )
    return selected


def load_dataset_runtime(workdir: Path, cfg: DatasetConfig) -> DatasetRuntime:
    graph = read_edge_list(
        str(workdir / cfg.edge_file),
        directed=cfg.directed,
        ignore_first_col=cfg.ignore_first_col,
    )
    reference_path, shortest_cost = graph.shortest_path(cfg.source, cfg.target)
    if not reference_path or math.isinf(shortest_cost):
        raise RuntimeError(f"{cfg.name}: source/target pair is unreachable")
    return DatasetRuntime(
        config=cfg,
        graph=graph,
        reference_path=reference_path,
        shortest_cost=float(shortest_cost),
        shortest_path_cache={},
    )


def run_one_case(
    runtime: DatasetRuntime,
    param_name: str,
    param_value: int,
    trial: int,
    pool_size: int,
    k: int,
    deadline: int,
    capacity: int,
    active_limit: int,
    detour_ratio: float,
    max_states: int,
    seed: int,
) -> Dict[str, object]:
    cfg = runtime.config
    rng = random.Random(seed)
    pool = generate_request_pool(
        runtime.graph,
        runtime.reference_path,
        runtime.shortest_cost,
        pool_size,
        deadline,
        rng,
    )
    active_requests = select_active_requests(pool, runtime.reference_path, pool_size, active_limit, rng)

    if not active_requests:
        result = ExperimentResult(
            method="DPKMOR",
            served_count=0,
            total_cost=float("inf"),
            total_time=float("inf"),
            avg_wait=0.0,
            max_wait=0.0,
            runtime=0.0,
        )
    else:
        try:
            result = run_kmor_variant(
                runtime.graph,
                cfg.source,
                cfg.target,
                active_requests,
                "DPKMOR",
                capacity=capacity,
                k=k,
                time_budget=cfg.time_budget,
                detour_ratio=detour_ratio,
                max_states=max_states,
                max_candidates=max(k * 8, 40),
                shortest_path_cache=runtime.shortest_path_cache,
            )
        except MemoryError:
            result = ExperimentResult(
                method="DPKMOR",
                served_count=-1,
                total_cost=float("inf"),
                total_time=float("inf"),
                avg_wait=float("inf"),
                max_wait=float("inf"),
                runtime=float("inf"),
                preprocess_time=float("inf"),
                search_time=float("inf"),
                timed_out=True,
            )

    status = "INF" if result.timed_out or math.isinf(result.total_cost) else "OK"
    return {
        "parameter": param_name,
        "value": param_value,
        "dataset": cfg.name,
        "trial": trial,
        "status": status,
        "pool_size": pool_size,
        "active_requests": len(active_requests),
        "k": k,
        "deadline": deadline,
        "capacity": capacity,
        "served": result.served_count,
        "cost": result.total_cost,
        "query_time": result.runtime,
        "preprocess_time": result.preprocess_time,
        "search_time": result.search_time,
        "examined_partial_routes": result.expanded_states,
        "generated": result.generated_states,
        "dp_pruned": result.dp_pruned_states,
        "feas_pruned": result.feasibility_pruned_states,
        "heur_pruned": result.heuristic_pruned_states,
        "hit_max_states": result.hit_max_states,
    }


def average(values: Iterable[float]) -> float:
    items = [float(item) for item in values if not math.isinf(float(item))]
    if not items:
        return float("inf")
    return sum(items) / len(items)


def summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, int, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (str(row["parameter"]), int(row["value"]), str(row["dataset"]))
        groups.setdefault(key, []).append(row)

    summary: List[Dict[str, object]] = []
    for (parameter, value, dataset), items in sorted(
        groups.items(), key=lambda x: (x[0][0], x[0][1], dataset_sort_key(x[0][2]))
    ):
        ok_items = [item for item in items if item["status"] == "OK"]
        source_items = ok_items if ok_items else items
        summary.append(
            {
                "parameter": parameter,
                "value": value,
                "dataset": dataset,
                "trials": len(items),
                "ok_trials": len(ok_items),
                "served": average(float(item["served"]) for item in source_items),
                "cost": average(float(item["cost"]) for item in source_items),
                "query_time": average(float(item["query_time"]) for item in source_items),
                "preprocess_time": average(float(item["preprocess_time"]) for item in source_items),
                "search_time": average(float(item["search_time"]) for item in source_items),
                "examined_partial_routes": average(float(item["examined_partial_routes"]) for item in source_items),
                "generated": average(float(item["generated"]) for item in source_items),
                "dp_pruned": average(float(item["dp_pruned"]) for item in source_items),
                "feas_pruned": average(float(item["feas_pruned"]) for item in source_items),
                "heur_pruned": average(float(item["heur_pruned"]) for item in source_items),
            }
        )
    return summary


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_plot_value(value: object) -> float:
    number = float(value)
    return np.nan if math.isinf(number) else number


def set_balanced_ylim(ax, series_values: Sequence[float], log_scale: bool) -> None:
    finite = [float(value) for value in series_values if not np.isnan(value) and value > 0]
    if not finite:
        return
    ymin = min(finite)
    ymax = max(finite)
    if ymin == ymax:
        ymin *= 0.75
        ymax *= 1.35
    if log_scale:
        ax.set_ylim(max(ymin / 2.2, 1e-9), ymax * 2.2)
    else:
        padding = (ymax - ymin) * 0.45
        ax.set_ylim(max(0.0, ymin - padding), ymax + padding)


# ===================================================================
#  Plotting function – fixed axes layout for consistent PDF sizes
# ===================================================================
def plot_metric(
    summary_rows: Sequence[Dict[str, object]],
    parameter: str,
    metric_key: str,
    metric_label: str,
    out_dir: Path,
    log_scale: bool,
    group_name: str,
    group_datasets: Sequence[str],
) -> None:
    allowed = set(group_datasets)
    rows = [row for row in summary_rows if row["parameter"] == parameter and row["dataset"] in allowed]
    if not rows:
        return

    datasets = [name for name in group_datasets if any(str(row["dataset"]) == name for row in rows)]
    values = sorted({int(row["value"]) for row in rows})

    # 纯白背景，固定 figsize
    plt.style.use("seaborn-v0_8-white")
    fig = plt.figure(figsize=(3.5, 2.5), facecolor="white")
    
    # 关键：固定 axes 的位置 [left, bottom, width, height]（相对 figure 的比例）
    # left=0.22 给 y 轴标签预留固定空间，不论标签是 "0.0" 还是 "100000"
    fig = plt.figure(figsize=(3.6, 2.5), facecolor="white")
    ax = fig.add_axes([0.18, 0.18, 0.78, 0.72])
    all_y_values: List[float] = []
    for dataset in datasets:
        style = LINE_STYLES.get(dataset, {"color": "#333333", "marker": "o", "linestyle": "-"})
        y_values = []
        for value in values:
            row = next(
                (item for item in rows if str(item["dataset"]) == dataset and int(item["value"]) == value),
                None,
            )
            y_val = np.nan if row is None else to_plot_value(row[metric_key])
            y_values.append(y_val)
        all_y_values.extend(y_values)
        ax.plot(
            values,
            y_values,
            label=dataset,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=9.5,
            linewidth=1.2,
            markeredgecolor="white",
            markeredgewidth=0.6,
        )

    ax.set_title(metric_label, fontweight="bold", pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks(values)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="x", labelsize=8, pad=4)
    ax.tick_params(axis="y", labelsize=8, pad=4)

    if log_scale:
        ax.set_yscale("log")
    set_balanced_ylim(ax, all_y_values, log_scale)

    legend = ax.legend(
        frameon=True,
        fancybox=False,
        edgecolor="black",
        ncol=2,
        loc="best",
         markerscale=0.8,            # 原来是 1.2，调回默认大小
        fontsize=7.5,               # 新增：图例字体比刻度略小
    )
    legend.get_frame().set_linewidth(0.7)

    out_base = out_dir / f"sensitivity_{group_name}_{parameter}_{metric_key}"
    fig.savefig(out_base.with_suffix(".pdf"), pad_inches=0.05)
    plt.close(fig)


def _sanitize_metric_filename(metric_label: str) -> str:
    safe = metric_label.lower().replace(" ", "_")
    safe = safe.replace("(", "").replace(")", "").replace("/", "_per_")
    return safe.replace("-", "_")
# ===================================================================
#  Plot orchestrator
# ===================================================================
def plot_all(summary_rows: Sequence[Dict[str, object]], out_dir: Path) -> None:
    """每个参数生成两张图：一张 search_time，一张 examined_partial_routes。
    每张图里 OL、TG、NY、CD 四条折线画在同一个坐标系中。
    y 轴刻度采用纯 10 的幂次形式，范围紧贴数据最大值。
    search_time 只保留 10⁻¹ 和 10⁰，上限为最大值的 1.1 倍。
    examined_partial_routes 保持 10³, 10⁴, 10⁵，范围动态调整。
    """
    datasets_all = ["OL", "TG", "NY", "CD"]

    for parameter in ["R", "k", "deadline", "capacity"]:
        for metric_key, metric_label in [
            ("search_time", "Search Time (s)"),
            ("examined_partial_routes", "Examined Partial Routes"),
        ]:
            fig, ax = plt.subplots(figsize=(3.5, 2.5), facecolor="white")
            ax.set_facecolor("white")

            all_y_vals = []  # 收集所有数据点，用于动态设定y轴范围

            for dataset in datasets_all:
                rows = [row for row in summary_rows
                        if row["parameter"] == parameter and row["dataset"] == dataset]
                if not rows:
                    continue

                values = sorted({int(row["value"]) for row in rows})
                style = LINE_STYLES.get(dataset, {"color": "#333333", "marker": "o", "linestyle": "-"})
                y_vals = []
                for v in values:
                    row = next((r for r in rows if int(r["value"]) == v), None)
                    y_val = np.nan if row is None else to_plot_value(row[metric_key])
                    y_vals.append(y_val)
                all_y_vals.extend(y_vals)

                ax.plot(values, y_vals,
                        label=dataset,
                        color=style["color"], linestyle=style["linestyle"],
                        marker=style["marker"], markersize=6, linewidth=1.2,
                        markeredgecolor="white", markeredgewidth=0.5)

            # ---------- 根据指标设置不同的 y 轴刻度和范围 ----------
            if metric_key == "search_time":
                # 刻度：只保留 10⁻¹ 和 10⁰
                ticks = [0.1, 1.0]
                labels = [r"$10^{-1}$", r"$10^{0}$"]
                ax.set_yticks(ticks)
                ax.set_yticklabels(labels)

                # 范围：下限 0.03（数据最小约 0.05），上限为最大值的 1.1 倍
                clean = [v for v in all_y_vals if not np.isnan(v) and v > 0]
                if clean:
                    ymax = max(clean)
                    ax.set_ylim(0.03, ymax * 1.3)

            else:  # examined_partial_routes
                # 刻度：保留 10³, 10⁴, 10⁵
                ticks = [1000, 10000, 100000]
                labels = [r"$10^{3}$", r"$10^{4}$", r"$10^{5}$"]
                ax.set_yticks(ticks)
                ax.set_yticklabels(labels)

                # 范围：下限为最小值的 0.9 倍，上限为最大值的 1.1 倍
                clean = [v for v in all_y_vals if not np.isnan(v) and v > 0]
                if clean:
                    ymin, ymax = min(clean), max(clean)
                    ax.set_ylim(ymin * 0.9, ymax * 1.3)

            ax.set_xticks(values)
            ax.tick_params(axis="x", labelsize=8, pad=3)
            ax.tick_params(axis="y", labelsize=8, pad=3)

            ax.set_title(metric_label, fontweight="bold", fontsize=10, pad=6)

            legend = ax.legend(
                frameon=True, fancybox=False, edgecolor="black",
                ncol=4, loc="upper left", fontsize=7.5,
                markerscale=0.8,
            )
            legend.get_frame().set_linewidth(0.6)

            plt.tight_layout()
            out_base = out_dir / f"sensitivity_{parameter}_{_sanitize_metric_filename(metric_label)}"
            fig.savefig(out_base.with_suffix(".pdf"), pad_inches=0.05, bbox_inches="tight")
            plt.close(fig)
# ===================================================================
#  CLI (unchanged)
# ===================================================================
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run DPKMOR parameter sensitivity experiments")
    ap.add_argument("--workdir", default=".", help="Workspace directory")
    ap.add_argument("--config", default="paper_datasets.json", help="Dataset config JSON")
    ap.add_argument("--output-dir", default="paper_results", help="Directory for CSV and figures")
    ap.add_argument("--datasets", default="OL,TG,NY,CD", help="Comma-separated dataset names")
    ap.add_argument("--params", default="R,k,deadline,capacity", help="Comma-separated parameters to run")
    ap.add_argument("--only", choices=["run", "plot", "all"], default="all")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--active-limit", type=int, default=18)
    ap.add_argument("--max-states", type=int, default=100000)
    ap.add_argument("--detour-ratio", type=float, default=1.4)
    ap.add_argument("--pool-values", default=",".join(map(str, DEFAULT_POOL_VALUES)))
    ap.add_argument("--k-values", default=",".join(map(str, DEFAULT_K_VALUES)))
    ap.add_argument("--deadline-values", default=",".join(map(str, DEFAULT_DEADLINE_VALUES)))
    ap.add_argument("--capacity-values", default=",".join(map(str, DEFAULT_CAPACITY_VALUES)))
    ap.add_argument("--default-pool-size", type=int, default=DEFAULT_POOL_SIZE)
    ap.add_argument("--default-k", type=int, default=DEFAULT_K)
    ap.add_argument("--default-deadline", type=int, default=DEFAULT_DEADLINE)
    ap.add_argument("--default-capacity", type=int, default=DEFAULT_CAPACITY)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    workdir = Path(args.workdir).resolve()
    out_dir = workdir / args.output_dir
    ensure_dir(out_dir)

    result_path = out_dir / "parameter_sensitivity_results.csv"
    summary_path = out_dir / "parameter_sensitivity_summary.csv"

    if args.only in {"run", "all"}:
        wanted_datasets = {item.strip() for item in args.datasets.split(",") if item.strip()}
        wanted_params = [item.strip() for item in args.params.split(",") if item.strip()]
        configs = [cfg for cfg in load_configs(workdir / args.config) if cfg.name in wanted_datasets]
        configs.sort(key=lambda cfg: dataset_sort_key(cfg.name))

        param_values = {
            "R": parse_int_list(args.pool_values),
            "k": parse_int_list(args.k_values),
            "deadline": parse_int_list(args.deadline_values),
            "capacity": parse_int_list(args.capacity_values),
        }

        runtimes = [load_dataset_runtime(workdir, cfg) for cfg in configs]
        rows: List[Dict[str, object]] = []
        for runtime in runtimes:
            for parameter in wanted_params:
                if parameter not in param_values:
                    raise ValueError(f"unsupported parameter: {parameter}")
                for value in param_values[parameter]:
                    for trial in range(1, args.trials + 1):
                        pool_size = value if parameter == "R" else args.default_pool_size
                        k = value if parameter == "k" else args.default_k
                        deadline = value if parameter == "deadline" else args.default_deadline
                        capacity = value if parameter == "capacity" else args.default_capacity
                        param_offset = {"R": 11, "k": 23, "deadline": 37, "capacity": 41}[parameter]
                        seed = args.seed + trial * 1009 + dataset_sort_key(runtime.config.name)[0] * 100003 + param_offset
                        row = run_one_case(
                            runtime,
                            parameter,
                            value,
                            trial,
                            pool_size,
                            k,
                            deadline,
                            capacity,
                            args.active_limit,
                            args.detour_ratio,
                            args.max_states,
                            seed,
                        )
                        rows.append(row)
                        print(
                            f"{runtime.config.name} {parameter}={value} trial={trial}: "
                            f"served={row['served']} search={float(row['search_time']):.4f}s "
                            f"examined={row['examined_partial_routes']}"
                        )

        summary_rows = summarize(rows)
        write_csv(result_path, rows)
        write_csv(summary_path, summary_rows)
        print(f"Saved raw results to {result_path}")
        print(f"Saved summary to {summary_path}")

    if args.only in {"plot", "all"}:
        summary_rows = read_csv(summary_path)
        plot_all(summary_rows, out_dir)
        print(f"Saved sensitivity figures to {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())