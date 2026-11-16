#!/usr/bin/env python3
"""Run Basic, KPrun, and bundled DPKMOR on one archived R=25 query batch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

from Dpkmor import RideSharePlanner, read_edge_list
from static_correctness_experiment import (
    COMPLETE,
    OOM,
    load_configs,
    parse_name_list,
    query_fingerprint,
    read_query_bank,
)
from static_kprun_efficiency import measure_search, planner_row, result_fingerprint


METHODS = ("Basic Search", "KPrun-Search", "DPKMOR")
VERSION = "static-threeway-r25-v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_id(query_fingerprint_value: str, args: argparse.Namespace) -> str:
    payload = {
        "version": VERSION,
        "query": query_fingerprint_value,
        "capacity": args.capacity,
        "k": args.k,
        "detour_ratio": args.detour_ratio,
        "max_states": args.max_states,
        "timeout": args.timeout,
        "cost_tolerance": args.cost_tolerance,
        "dp_mode": "bundled",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def archived_queries(args: argparse.Namespace):
    names = set(parse_name_list(args.datasets))
    banks = [Path(item.strip()).resolve() for item in args.query_bank.split(",")]
    archives = [Path(item.strip()).resolve() for item in args.reference_results.split(",")]
    if any(not path.is_file() for path in banks + archives):
        raise FileNotFoundError("All archived query banks and reference results are required")
    all_queries = [query for bank in banks for query in read_query_bank(bank)]
    queries = [
        query for query in all_queries
        if query.dataset.upper() in names and query.request_size == 25
    ]
    expected = args.queries_per_dataset * len(names)
    if len(queries) != expected or len({query.query_id for query in queries}) != expected:
        raise ValueError(f"Expected {expected} distinct R=25 queries, found {len(queries)}")
    for name in names:
        if sum(query.dataset.upper() == name for query in queries) != args.queries_per_dataset:
            raise ValueError(f"Expected {args.queries_per_dataset} queries for {name}")

    reference = {}
    for archive in archives:
        for row in read_csv(archive):
            if row["method"] in ("KPrun-Search", "DPKMOR"):
                key = (row["query_id"], row["method"])
                if key in reference and reference[key]["query_fingerprint"] != row["query_fingerprint"]:
                    raise ValueError(f"Conflicting reference rows for {key}")
                reference[key] = row
    for query in queries:
        fingerprint = query_fingerprint(query)
        expected_run = result_fingerprint(
            query, args.capacity, args.k, args.detour_ratio,
            args.max_states, False, args.timeout, args.cost_tolerance, "bundled",
        )
        for method in ("KPrun-Search", "DPKMOR"):
            row = reference.get((query.query_id, method))
            if (
                row is None
                or row["query_fingerprint"] != fingerprint
                or (not args.reference_query_only and row["run_fingerprint"] != expected_run)
            ):
                raise ValueError(
                    f"Archive mismatch for {query.query_id}/{method}; "
                    "check the query bank and solver parameters before running"
                )
    return queries


def failed_row(query, method: str, fingerprint: str) -> dict[str, object]:
    return {
        "query_id": query.query_id,
        "query_fingerprint": query_fingerprint(query),
        "run_fingerprint": fingerprint,
        "dataset": query.dataset,
        "request_size": query.request_size,
        "seed": query.seed,
        "method": method,
        "status": OOM,
        "served": 0,
        "cost": math.nan,
        "preprocess_time": math.nan,
        "search_time": math.nan,
        "expanded_states": 0,
        "generated_states": 0,
        "frontier_objectives": "",
        "concrete_topk_fingerprint": "",
    }


def concrete_topk_fingerprint(routes, tolerance: float) -> str:
    scale = max(tolerance, 1e-12)
    payload = [
        {
            "served": route.served_count,
            "cost": round(route.total_cost / scale),
            "served_requests": sorted(route.served_requests),
            "route_events": route.route_events,
            "key_path": route.key_path,
            "full_path": route.full_path,
            "hops": route.hops,
        }
        for route in routes
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def execute(args: argparse.Namespace, queries, result_path: Path) -> None:
    workdir = Path(args.workdir).resolve()
    configs = {config.name: config for config in load_configs(workdir / args.config)}
    if any(query.dataset not in configs for query in queries):
        raise ValueError("The dataset config does not contain all archived queries")

    existing = read_csv(result_path)
    by_key = {
        (row["query_id"], row["method"], row["run_fingerprint"]): row
        for row in existing
    }
    for dataset in parse_name_list(args.datasets):
        config = configs[dataset]
        dataset_queries = [query for query in queries if query.dataset == dataset]
        print(f"Loading {dataset} graph ({len(dataset_queries)} queries)...", flush=True)
        graph = read_edge_list(
            str(workdir / config.edge_file),
            directed=config.directed,
            ignore_first_col=config.ignore_first_col,
        )
        cache = {}
        for index, query in enumerate(dataset_queries, 1):
            fingerprint = run_id(query_fingerprint(query), args)
            order = METHODS[query.seed % 3:] + METHODS[:query.seed % 3]
            for method in order:
                key = (query.query_id, method, fingerprint)
                if key in by_key:
                    continue
                print(f"  {query.query_id}: starting {method}", flush=True)
                try:
                    started = time.perf_counter()
                    planner = RideSharePlanner(
                        graph,
                        query.source,
                        query.target,
                        list(query.requests),
                        capacity=args.capacity,
                        k=args.k,
                        time_budget=query.time_budget,
                        detour_ratio=args.detour_ratio,
                        max_states=args.max_states,
                        search_timeout_seconds=args.timeout,
                        pareto_only=True,
                        enable_compatibility_pruning=False,
                        enable_state_bounds=True,
                        aggressive_served_pruning=False,
                        numeric_tolerance=args.cost_tolerance,
                        shortest_path_cache=cache,
                        dp_mode="bundled",
                        enable_dp=method == "DPKMOR",
                        enable_pruning=method != "Basic Search",
                    )
                    preprocess_time = time.perf_counter() - started
                    routes, search_time = measure_search(planner)
                    row = planner_row(
                        query, method, planner, routes, preprocess_time,
                        search_time, math.nan, fingerprint, args.cost_tolerance,
                    )
                    row["concrete_topk_fingerprint"] = concrete_topk_fingerprint(
                        routes, args.cost_tolerance,
                    )
                except MemoryError:
                    row = failed_row(query, method, fingerprint)
                by_key[key] = row
                write_csv(result_path, list(by_key.values()))
                print(
                    f"  {query.query_id}: {method} {row['status']} "
                    f"search={row['search_time']}s expanded={row['expanded_states']}",
                    flush=True,
                )
            print(f"[{dataset} {index}/{len(dataset_queries)}] {query.query_id}", flush=True)


def summarize(queries, args: argparse.Namespace, result_path: Path, out_dir: Path) -> bool:
    rows = {
        (row["query_id"], row["method"], row["run_fingerprint"]): row
        for row in read_csv(result_path)
    }
    pairs = []
    for query in queries:
        fingerprint = run_id(query_fingerprint(query), args)
        methods = {method: rows.get((query.query_id, method, fingerprint)) for method in METHODS}
        complete = all(row is not None and row["status"] == COMPLETE for row in methods.values())
        match = complete and len({row["frontier_objectives"] for row in methods.values()}) == 1
        concrete_match = complete and len({
            row.get("concrete_topk_fingerprint", "") for row in methods.values()
        }) == 1
        pairs.append({
            "query_id": query.query_id,
            "dataset": query.dataset,
            "request_size": query.request_size,
            "all_complete": complete,
            "objective_frontier_match": match if complete else "",
            "concrete_topk_match": concrete_match if complete else "",
            **{f"{method}_status": row["status"] if row else "MISSING" for method, row in methods.items()},
        })
    write_csv(out_dir / "threeway_efficiency_pairs.csv", pairs)

    summary = []
    for dataset in parse_name_list(args.datasets):
        dataset_ids = {
            pair["query_id"] for pair in pairs
            if pair["dataset"] == dataset and pair["all_complete"]
            and pair["objective_frontier_match"] and pair["concrete_topk_match"]
        }
        for method in METHODS:
            selected = [
                rows[(query.query_id, method, run_id(query_fingerprint(query), args))]
                for query in queries if query.query_id in dataset_ids
            ]
            summary.append({
                "dataset": dataset,
                "method": method,
                "total_queries": sum(query.dataset == dataset for query in queries),
                "paired_complete_queries": len(selected),
                "mean_search_time": statistics.fmean(float(row["search_time"]) for row in selected) if selected else math.nan,
                "mean_expanded_states": statistics.fmean(int(row["expanded_states"]) for row in selected) if selected else math.nan,
                "mean_generated_states": statistics.fmean(int(row["generated_states"]) for row in selected) if selected else math.nan,
            })
    write_csv(out_dir / "threeway_efficiency_summary.csv", summary)
    ready = len(pairs) == len(queries) and all(
        pair["all_complete"] and pair["objective_frontier_match"]
        and pair["concrete_topk_match"] for pair in pairs
    )
    if ready:
        plot(summary, parse_name_list(args.datasets), out_dir / "threeway_efficiency.pdf")
    else:
        print("Figure not generated: every query must complete with matching concrete Top-k results.")
    return ready


def plot(summary: list[dict[str, object]], datasets: list[str], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "pdf.fonttype": 42,
    })
    lookup = {(row["dataset"], row["method"]): row for row in summary}
    colors = ("#cccccc", "#999999", "#333333")
    hatches = ("\\\\", "///", "")
    positions = np.arange(len(datasets))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.85))
    handles = []
    for ax, column, ylabel, scale in (
        (axes[0], "mean_search_time", "Search Time (s)", 1.0),
        (axes[1], "mean_expanded_states", "Expanded States (10$^3$)", 1000.0),
    ):
        for index, method in enumerate(METHODS):
            heights = [float(lookup[(dataset, method)][column]) / scale for dataset in datasets]
            bars = ax.bar(
                positions + (index - 1) * width, heights, width,
                color=colors[index], edgecolor="#292929", linewidth=0.55,
                hatch=hatches[index], label=method,
            )
            if ax is axes[0]:
                handles.append(bars[0])
        ax.set_xticks(positions, datasets)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis="both", labelsize=8, width=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.legend(handles, METHODS, loc="upper center", ncol=3, frameon=False,
               fontsize=8, handlelength=1.6, columnspacing=1.2,
               bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.17, top=0.82, wspace=0.34)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(f"Saved {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--config", default="paper_datasets.json")
    parser.add_argument("--query-bank", default="paper_results_kprun_bundled_final/efficiency_queries.jsonl")
    parser.add_argument("--reference-results", default="paper_results_kprun_bundled_dag_r25_final/kprun_efficiency_results.csv")
    parser.add_argument("--reference-query-only", action="store_true",
                        help="Check archived query identity but allow a new common solver configuration")
    parser.add_argument("--datasets", default="OL,TG,CD,NY")
    parser.add_argument("--queries-per-dataset", type=int, default=30)
    parser.add_argument("--output-dir", default="paper_results_threeway_r25_final")
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--detour-ratio", type=float, default=1.4)
    parser.add_argument("--max-states", type=int, default=1_000_000)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--cost-tolerance", type=float, default=1e-6)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    args.query_bank = ",".join(
        str((workdir / item.strip()).resolve()) for item in args.query_bank.split(",")
    )
    args.reference_results = ",".join(
        str((workdir / item.strip()).resolve()) for item in args.reference_results.split(",")
    )
    queries = archived_queries(args)
    out_dir = (workdir / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "threeway_run_config.json").write_text(
        json.dumps({"version": VERSION, **vars(args), "verified_query_count": len(queries)},
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result_path = out_dir / "threeway_efficiency_results.csv"
    if not args.summarize_only:
        execute(args, queries, result_path)
    summarize(queries, args, result_path, out_dir)
    print(f"Saved {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
