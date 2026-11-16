#!/usr/bin/env python3
"""Fair efficiency comparison between KPrun-Search and state-DP DPKMOR."""

from __future__ import annotations

import argparse
import bisect
import csv
import gc
import hashlib
import json
import math
import random
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from Dpkmor import (
    BUNDLED_IMPLEMENTATION_VERSION,
    Candidate,
    Graph,
    RideSharePlanner,
    read_edge_list,
)
from static_correctness_experiment import (
    COMPLETE,
    OOM,
    STATE_LIMIT,
    TIMEOUT,
    DatasetConfig,
    QueryInstance,
    generate_query_bank,
    load_configs,
    parse_int_list,
    parse_name_list,
    query_fingerprint,
    read_query_bank,
    tuple_request_to_request,
    write_query_bank,
)
from prepare_paper_datasets import (
    generate_requests_for_path,
    make_request_with_window,
    path_prefix_costs,
    sample_competitive_demand,
    scaled_wait,
)


ALGORITHM_VERSION = "static-kprun-efficiency-exact-fast-v6.3"

RESULT_FIELDS = [
    "query_id",
    "query_fingerprint",
    "run_fingerprint",
    "dataset",
    "request_size",
    "seed",
    "method",
    "status",
    "served",
    "cost",
    "frontier_size",
    "returned_routes",
    "preprocess_time",
    "search_time",
    "expanded_states",
    "generated_states",
    "feasibility_checks",
    "feasibility_pruned",
    "heuristic_pruned",
    "dp_pruned",
    "peak_open_size",
    "peak_active_dp_labels",
    "dag_nodes_created",
    "dag_edges_created",
    "peak_memory_mb",
    "hit_max_states",
    "timed_out",
    "frontier_objectives",
]

SUMMARY_FIELDS = [
    "dataset",
    "request_size",
    "method",
    "total_queries",
    "completed_queries",
    "completion_rate",
    "timeout_rate",
    "oom_rate",
    "state_limit_rate",
    "mean_search_time",
    "median_search_time",
    "p95_search_time",
    "mean_expanded_states",
    "mean_generated_states",
    "mean_feasibility_checks",
    "mean_peak_open_size",
    "mean_peak_active_dp_labels",
    "mean_dag_nodes_created",
    "mean_dag_edges_created",
    "mean_peak_memory_mb",
]

PAIR_FIELDS = [
    "query_id",
    "dataset",
    "request_size",
    "kprun_status",
    "dpkmor_status",
    "paired_complete",
    "answer_match",
    "speedup",
    "expanded_reduction",
    "generated_reduction",
    "memory_reduction",
]

PAIR_SUMMARY_FIELDS = [
    "dataset",
    "request_size",
    "total_pairs",
    "completed_pairs",
    "answer_match_rate",
    "mean_speedup",
    "median_speedup",
    "p95_speedup",
    "mean_expanded_reduction",
    "mean_generated_reduction",
    "mean_memory_reduction",
]


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def percentile(values: Sequence[float], fraction: float) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan
    ordered = sorted(finite)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def safe_mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else math.nan


def result_fingerprint(
    query: QueryInstance,
    capacity: int,
    k: int,
    detour_ratio: float,
    max_states: int,
    measure_memory: bool,
    timeout_seconds: float,
    cost_tolerance: float,
    dp_mode: str = "exact_fast",
) -> str:
    payload = {
        "version": ALGORITHM_VERSION,
        "query": query_fingerprint(query),
        "capacity": capacity,
        "k": k,
        "detour_ratio": detour_ratio,
        "max_states": max_states,
        "measure_memory": measure_memory,
        "timeout_seconds": timeout_seconds,
        "cost_tolerance": cost_tolerance,
    }
    if dp_mode != "exact_fast":
        payload["dp_mode"] = dp_mode
        payload["dp_implementation"] = BUNDLED_IMPLEMENTATION_VERSION
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def quantized_frontier(
    candidates: Sequence[Candidate],
    tolerance: float,
) -> str:
    scale = max(tolerance, 1e-12)
    objectives = sorted(
        {
            (candidate.served_count, round(candidate.total_cost / scale))
            for candidate in candidates
        },
        key=lambda item: (-item[0], item[1]),
    )
    return json.dumps(objectives, separators=(",", ":"))


def measure_search(planner: RideSharePlanner) -> Tuple[List[Candidate], float]:
    gc.collect()
    started = time.perf_counter()
    routes = planner.solve()
    search_time = time.perf_counter() - started
    return routes, search_time


def measure_peak_memory(planner: RideSharePlanner) -> float:
    gc.collect()
    tracemalloc.start()
    planner.solve()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak_bytes / (1024.0 * 1024.0)


def status_for(planner: RideSharePlanner) -> str:
    if planner.stats.timed_out:
        return TIMEOUT
    if planner.stats.hit_max_states:
        return STATE_LIMIT
    return COMPLETE


def planner_row(
    query: QueryInstance,
    method: str,
    planner: RideSharePlanner,
    routes: Sequence[Candidate],
    preprocess_time: float,
    search_time: float,
    peak_memory_mb: float,
    run_id: str,
    tolerance: float,
) -> Dict[str, object]:
    best = routes[0] if routes else None
    return {
        "query_id": query.query_id,
        "query_fingerprint": query_fingerprint(query),
        "run_fingerprint": run_id,
        "dataset": query.dataset,
        "request_size": query.request_size,
        "seed": query.seed,
        "method": method,
        "status": status_for(planner),
        "served": best.served_count if best else 0,
        "cost": best.total_cost if best else math.inf,
        "frontier_size": len(planner.last_complete_frontier),
        "returned_routes": len(routes),
        "preprocess_time": preprocess_time,
        "search_time": search_time,
        "expanded_states": planner.stats.expanded_states,
        "generated_states": planner.stats.generated_states,
        "feasibility_checks": planner.stats.feasibility_checks,
        "feasibility_pruned": planner.stats.feasibility_pruned_states,
        "heuristic_pruned": planner.stats.heuristic_pruned_states,
        "dp_pruned": planner.stats.dp_pruned_states,
        "peak_open_size": planner.stats.peak_open_size,
        "peak_active_dp_labels": planner.stats.peak_active_dp_labels,
        "dag_nodes_created": planner.stats.dag_nodes_created,
        "dag_edges_created": planner.stats.dag_edges_created,
        "peak_memory_mb": peak_memory_mb,
        "hit_max_states": planner.stats.hit_max_states,
        "timed_out": planner.stats.timed_out,
        "frontier_objectives": quantized_frontier(
            planner.last_complete_frontier,
            tolerance,
        ),
    }


def failed_method_rows(
    query: QueryInstance,
    run_id: str,
    status: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for method in ["KPrun-Search", "DPKMOR"]:
        rows.append(
            {
                "query_id": query.query_id,
                "query_fingerprint": query_fingerprint(query),
                "run_fingerprint": run_id,
                "dataset": query.dataset,
                "request_size": query.request_size,
                "seed": query.seed,
                "method": method,
                "status": status,
                "served": 0,
                "cost": math.inf,
                "frontier_size": 0,
                "returned_routes": 0,
                "preprocess_time": math.inf,
                "search_time": math.inf,
                "expanded_states": 0,
                "generated_states": 0,
                "feasibility_checks": 0,
                "feasibility_pruned": 0,
                "heuristic_pruned": 0,
                "dp_pruned": 0,
                "peak_open_size": 0,
                "peak_active_dp_labels": 0,
                "dag_nodes_created": 0,
                "dag_edges_created": 0,
                "peak_memory_mb": math.nan,
                "hit_max_states": False,
                "timed_out": status == TIMEOUT,
                "frontier_objectives": "[]",
            }
        )
    return rows


def run_pair(
    graph: Graph,
    query: QueryInstance,
    shortest_path_cache: Dict[Tuple[str, str], Tuple[List[str], float]],
    capacity: int,
    k: int,
    detour_ratio: float,
    max_states: int,
    timeout_seconds: float,
    tolerance: float,
    measure_memory: bool,
    dp_mode: str = "exact_fast",
) -> List[Dict[str, object]]:
    common = {
        "capacity": capacity,
        "k": k,
        "time_budget": query.time_budget,
        "detour_ratio": detour_ratio,
        "max_states": max_states,
        "search_timeout_seconds": timeout_seconds,
        "pareto_only": True,
        "enable_compatibility_pruning": False,
        "enable_state_bounds": True,
        "aggressive_served_pruning": False,
        "numeric_tolerance": tolerance,
        "shortest_path_cache": shortest_path_cache,
        "dp_mode": dp_mode,
    }
    run_id = result_fingerprint(
        query,
        capacity,
        k,
        detour_ratio,
        max_states,
        measure_memory,
        timeout_seconds,
        tolerance,
        dp_mode,
    )

    started = time.perf_counter()
    kprun = RideSharePlanner(
        graph,
        query.source,
        query.target,
        list(query.requests),
        enable_pruning=True,
        enable_dp=False,
        **common,
    )
    shared_preprocess_time = time.perf_counter() - started

    started = time.perf_counter()
    dpkmor = RideSharePlanner(
        graph,
        query.source,
        query.target,
        list(query.requests),
        enable_pruning=True,
        enable_dp=True,
        **common,
    )
    secondary_setup_time = time.perf_counter() - started

    planners = {
        "KPrun-Search": (kprun, shared_preprocess_time),
        "DPKMOR": (dpkmor, shared_preprocess_time),
    }
    order = (
        ["KPrun-Search", "DPKMOR"]
        if query.seed % 2 == 0
        else ["DPKMOR", "KPrun-Search"]
    )
    measured = {}
    for method in order:
        planner, preprocess_time = planners[method]
        print(
            f"    {query.query_id}: starting {method} "
            f"(per-method timeout={timeout_seconds:g}s)",
            flush=True,
        )
        routes, search_time = measure_search(planner)
        measured[method] = (routes, search_time, math.nan, preprocess_time)
        print(
            f"    {query.query_id}: finished {method} "
            f"status={status_for(planner)} time={search_time:.3f}s "
            f"expanded={planner.stats.expanded_states}",
            flush=True,
        )

    if measure_memory:
        for method in ["KPrun-Search", "DPKMOR"]:
            memory_planner = RideSharePlanner(
                graph,
                query.source,
                query.target,
                list(query.requests),
                enable_pruning=True,
                enable_dp=method == "DPKMOR",
                **common,
            )
            routes, search_time, _, preprocess_time = measured[method]
            measured[method] = (
                routes,
                search_time,
                measure_peak_memory(memory_planner),
                preprocess_time,
            )

    rows = []
    for method in ["KPrun-Search", "DPKMOR"]:
        planner, _ = planners[method]
        routes, search_time, peak_memory_mb, preprocess_time = measured[method]
        row = planner_row(
            query,
            method,
            planner,
            routes,
            preprocess_time,
            search_time,
            peak_memory_mb,
            run_id,
            tolerance,
        )
        row["secondary_setup_time"] = secondary_setup_time
        rows.append(row)
    return rows


def reduction(baseline: float, optimized: float) -> float:
    if baseline <= 0:
        return math.nan
    return (baseline - optimized) / baseline


def make_pair_row(
    kprun: Dict[str, object],
    dpkmor: Dict[str, object],
) -> Dict[str, object]:
    paired = kprun["status"] == COMPLETE and dpkmor["status"] == COMPLETE
    if paired:
        k_time = float(kprun["search_time"])
        d_time = float(dpkmor["search_time"])
        speedup = k_time / d_time if d_time > 0 else math.inf
        answer_match = kprun["frontier_objectives"] == dpkmor["frontier_objectives"]
        expanded_reduction = reduction(
            float(kprun["expanded_states"]),
            float(dpkmor["expanded_states"]),
        )
        generated_reduction = reduction(
            float(kprun["generated_states"]),
            float(dpkmor["generated_states"]),
        )
        memory_reduction = reduction(
            float(kprun["peak_memory_mb"]),
            float(dpkmor["peak_memory_mb"]),
        )
    else:
        speedup = math.nan
        answer_match = ""
        expanded_reduction = math.nan
        generated_reduction = math.nan
        memory_reduction = math.nan
    return {
        "query_id": kprun["query_id"],
        "dataset": kprun["dataset"],
        "request_size": kprun["request_size"],
        "kprun_status": kprun["status"],
        "dpkmor_status": dpkmor["status"],
        "paired_complete": paired,
        "answer_match": answer_match,
        "speedup": speedup,
        "expanded_reduction": expanded_reduction,
        "generated_reduction": generated_reduction,
        "memory_reduction": memory_reduction,
    }


def summarize_method_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, int, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (str(row["dataset"]), int(row["request_size"]), str(row["method"]))
        groups.setdefault(key, []).append(row)

    summary = []
    for (dataset, request_size, method), items in sorted(groups.items()):
        completed = [item for item in items if item["status"] == COMPLETE]
        times = [float(item["search_time"]) for item in completed]
        summary.append(
            {
                "dataset": dataset,
                "request_size": request_size,
                "method": method,
                "total_queries": len(items),
                "completed_queries": len(completed),
                "completion_rate": len(completed) / len(items),
                "timeout_rate": sum(item["status"] == TIMEOUT for item in items) / len(items),
                "oom_rate": sum(item["status"] == OOM for item in items) / len(items),
                "state_limit_rate": sum(item["status"] == STATE_LIMIT for item in items) / len(items),
                "mean_search_time": safe_mean(times),
                "median_search_time": statistics.median(times) if times else math.nan,
                "p95_search_time": percentile(times, 0.95),
                "mean_expanded_states": safe_mean(
                    [float(item["expanded_states"]) for item in completed]
                ),
                "mean_generated_states": safe_mean(
                    [float(item["generated_states"]) for item in completed]
                ),
                "mean_feasibility_checks": safe_mean(
                    [float(item["feasibility_checks"]) for item in completed]
                ),
                "mean_peak_open_size": safe_mean(
                    [float(item["peak_open_size"]) for item in completed]
                ),
                "mean_peak_active_dp_labels": safe_mean(
                    [float(item["peak_active_dp_labels"]) for item in completed]
                ),
                "mean_dag_nodes_created": safe_mean(
                    [float(item["dag_nodes_created"]) for item in completed]
                ),
                "mean_dag_edges_created": safe_mean(
                    [float(item["dag_edges_created"]) for item in completed]
                ),
                "mean_peak_memory_mb": safe_mean(
                    [float(item["peak_memory_mb"]) for item in completed]
                ),
            }
        )
    return summary


def summarize_pair_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["dataset"]), int(row["request_size"])), []).append(row)
    summary = []
    for (dataset, request_size), items in sorted(groups.items()):
        completed = [item for item in items if str(item["paired_complete"]).lower() == "true"]
        speeds = [float(item["speedup"]) for item in completed]
        summary.append(
            {
                "dataset": dataset,
                "request_size": request_size,
                "total_pairs": len(items),
                "completed_pairs": len(completed),
                "answer_match_rate": (
                    sum(str(item["answer_match"]).lower() == "true" for item in completed)
                    / len(completed)
                    if completed
                    else math.nan
                ),
                "mean_speedup": safe_mean(speeds),
                "median_speedup": statistics.median(speeds) if speeds else math.nan,
                "p95_speedup": percentile(speeds, 0.95),
                "mean_expanded_reduction": safe_mean(
                    [float(item["expanded_reduction"]) for item in completed]
                ),
                "mean_generated_reduction": safe_mean(
                    [float(item["generated_reduction"]) for item in completed]
                ),
                "mean_memory_reduction": safe_mean(
                    [float(item["memory_reduction"]) for item in completed]
                ),
            }
        )
    return summary


def build_pair_rows(
    rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["query_id"]), {})[str(row["method"])] = row
    pairs = []
    for methods in grouped.values():
        if "KPrun-Search" in methods and "DPKMOR" in methods:
            pairs.append(make_pair_row(methods["KPrun-Search"], methods["DPKMOR"]))
    return pairs


def write_aggregate_outputs(
    rows: Sequence[Dict[str, object]],
    summary_path: Path,
    pair_path: Path,
    pair_summary_path: Path,
) -> None:
    pairs = build_pair_rows(rows)
    write_csv(summary_path, summarize_method_rows(rows), SUMMARY_FIELDS)
    write_csv(pair_path, pairs, PAIR_FIELDS)
    write_csv(pair_summary_path, summarize_pair_rows(pairs), PAIR_SUMMARY_FIELDS)


def generate_independent_nested_query_bank(
    workdir: Path,
    configs: Sequence[DatasetConfig],
    request_sizes: Sequence[int],
    queries_per_size: int,
    seed: int,
) -> List[QueryInstance]:
    """Generate one independent nested request family per trial.

    Each trial receives its own maximum-size request pool. Smaller request
    sizes use prefixes of a deterministically shuffled pool, so scalability
    comparisons are nested within a trial without correlating all trials
    through one dataset-wide master pool.
    """
    queries: List[QueryInstance] = []
    ordered_sizes = sorted(set(int(size) for size in request_sizes))
    max_size = max(ordered_sizes)

    for dataset_index, config in enumerate(configs):
        graph = read_edge_list(
            str(workdir / config.edge_file),
            directed=config.directed,
            ignore_first_col=config.ignore_first_col,
        )
        reference_path, shortest_cost = graph.shortest_path(
            config.source,
            config.target,
        )
        if not reference_path or math.isinf(shortest_cost):
            raise RuntimeError(f"{config.name}: unreachable configured endpoints")

        dataset_seed = seed + dataset_index * 1_000_003
        for query_number in range(1, queries_per_size + 1):
            trial_seed = dataset_seed + query_number * 100_003
            pool = generate_requests_for_path(
                graph,
                reference_path,
                float(shortest_cost),
                max_size,
                random.Random(trial_seed),
                profile="effectiveness",
            )
            if len(pool) < max_size:
                raise RuntimeError(
                    f"{config.name} trial {query_number}: generated only "
                    f"{len(pool)} of {max_size} requests"
                )
            random.Random(trial_seed + 53).shuffle(pool)

            for request_size in ordered_sizes:
                query_seed = trial_seed + request_size * 101
                requests = tuple(
                    tuple_request_to_request(
                        item,
                        f"q{query_number}_r{request_index}",
                    )
                    for request_index, item in enumerate(
                        pool[:request_size],
                        start=1,
                    )
                )
                queries.append(
                    QueryInstance(
                        query_id=(
                            f"{config.name}_R{request_size}_Q{query_number:04d}"
                        ),
                        dataset=config.name,
                        request_size=request_size,
                        seed=query_seed,
                        source=config.source,
                        target=config.target,
                        time_budget=config.time_budget,
                        requests=requests,
                    )
                )
    return queries


def _cost_fraction_index(
    prefix_costs: Sequence[float],
    low_fraction: float,
    high_fraction: float,
    rng: random.Random,
) -> int:
    """Sample a path position by cumulative travel cost, not node index."""
    total = float(prefix_costs[-1])
    target = total * rng.uniform(low_fraction, high_fraction)
    index = bisect.bisect_left(prefix_costs, target)
    return max(1, min(len(prefix_costs) - 2, index))


def generate_cost_normalized_requests_for_path(
    graph: Graph,
    path: Sequence[str],
    shortest_cost: float,
    num_requests: int,
    rng: random.Random,
) -> List[Tuple[str, str, str, int, float, float]]:
    """Generate the effectiveness workload in travel-cost coordinates.

    The original generator sampled fractions of path *nodes*. That makes the
    workload depend on OSM segmentation density. Sampling cumulative cost
    gives all road networks the same normalized pickup/dropoff geometry.
    """
    prefix_costs = path_prefix_costs(graph, path)
    requests: List[Tuple[str, str, str, int, float, float]] = []
    num_core = round(num_requests * 0.35)
    num_competitive = round(num_requests * 0.45)
    num_reverse = num_requests - num_core - num_competitive
    request_index = 1

    def append_request(
        pickup_range: Tuple[float, float],
        dropoff_range: Tuple[float, float],
        demand: int,
        wait_fraction: float,
        reverse: bool = False,
    ) -> None:
        nonlocal request_index
        pickup_idx = _cost_fraction_index(
            prefix_costs,
            pickup_range[0],
            pickup_range[1],
            rng,
        )
        dropoff_idx = _cost_fraction_index(
            prefix_costs,
            dropoff_range[0],
            dropoff_range[1],
            rng,
        )
        if reverse and dropoff_idx >= pickup_idx:
            pickup_fraction = prefix_costs[pickup_idx] / shortest_cost
            upper = max(0.22, min(0.55, pickup_fraction - 0.06))
            dropoff_idx = _cost_fraction_index(
                prefix_costs,
                0.22,
                upper,
                rng,
            )
        elif not reverse and dropoff_idx <= pickup_idx:
            pickup_fraction = prefix_costs[pickup_idx] / shortest_cost
            lower = min(0.92, pickup_fraction + 0.06)
            dropoff_idx = _cost_fraction_index(
                prefix_costs,
                lower,
                max(lower, dropoff_range[1]),
                rng,
            )
        wait_limit = scaled_wait(shortest_cost, rng, wait_fraction)
        requests.append(
            make_request_with_window(
                request_index,
                pickup_idx,
                dropoff_idx,
                path,
                prefix_costs,
                demand,
                wait_limit,
                rng,
            )
        )
        request_index += 1

    for _ in range(num_core):
        append_request(
            (0.16, 0.34),
            (0.46, 0.68),
            1 if rng.random() < 0.65 else 2,
            0.035,
        )
    for _ in range(num_competitive):
        append_request(
            (0.24, 0.45),
            (0.50, 0.76),
            sample_competitive_demand(rng),
            0.024,
        )
    for _ in range(num_reverse):
        append_request(
            (0.46, 0.72),
            (0.22, 0.55),
            1 if rng.random() < 0.70 else 2,
            0.045,
            reverse=True,
        )

    rng.shuffle(requests)
    return requests


def generate_cost_normalized_nested_query_bank(
    workdir: Path,
    configs: Sequence[DatasetConfig],
    request_sizes: Sequence[int],
    queries_per_size: int,
    seed: int,
) -> List[QueryInstance]:
    """Generate independent nested trials with cost-normalized locations."""
    queries: List[QueryInstance] = []
    ordered_sizes = sorted(set(int(size) for size in request_sizes))
    max_size = max(ordered_sizes)

    for dataset_index, config in enumerate(configs):
        graph = read_edge_list(
            str(workdir / config.edge_file),
            directed=config.directed,
            ignore_first_col=config.ignore_first_col,
        )
        reference_path, shortest_cost = graph.shortest_path(
            config.source,
            config.target,
        )
        if not reference_path or math.isinf(shortest_cost):
            raise RuntimeError(f"{config.name}: unreachable configured endpoints")

        dataset_seed = seed + dataset_index * 1_000_003
        for query_number in range(1, queries_per_size + 1):
            trial_seed = dataset_seed + query_number * 100_003
            pool = generate_cost_normalized_requests_for_path(
                graph,
                reference_path,
                float(shortest_cost),
                max_size,
                random.Random(trial_seed),
            )
            random.Random(trial_seed + 53).shuffle(pool)
            for request_size in ordered_sizes:
                query_seed = trial_seed + request_size * 101
                requests = tuple(
                    tuple_request_to_request(
                        item,
                        f"q{query_number}_r{request_index}",
                    )
                    for request_index, item in enumerate(
                        pool[:request_size],
                        start=1,
                    )
                )
                queries.append(
                    QueryInstance(
                        query_id=(
                            f"{config.name}_R{request_size}_Q{query_number:04d}"
                        ),
                        dataset=config.name,
                        request_size=request_size,
                        seed=query_seed,
                        source=config.source,
                        target=config.target,
                        time_budget=config.time_budget,
                        requests=requests,
                    )
                )
    return queries


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare KPrun-Search with DPKMOR")
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--config", default="paper_datasets.json")
    parser.add_argument("--output-dir", default="paper_results_kprun_efficiency")
    parser.add_argument("--query-bank", default="")
    parser.add_argument(
        "--query-ids",
        default="",
        help="Optional comma-separated query IDs for targeted verification",
    )
    parser.add_argument("--datasets", default="OL,TG,CD,NY")
    parser.add_argument("--request-sizes", default="8,10,12,15,20")
    parser.add_argument("--queries-per-size", type=int, default=30)
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--detour-ratio", type=float, default=1.4)
    parser.add_argument("--max-states", type=int, default=2_000_000)
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="Search-time limit for each method; one query pair may use twice this value",
    )
    parser.add_argument("--cost-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--dp-mode",
        choices=("exact_fast", "bundled"),
        default="exact_fast",
        help="DPKMOR mode; the default preserves prior experiment behavior",
    )
    parser.add_argument("--measure-memory", action="store_true")
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--query-bank-mode",
        choices=[
            "independent-cost-nested",
            "independent-nested",
            "legacy-shared-pool",
        ],
        default="independent-cost-nested",
        help=(
            "Query generation design. independent-cost-nested additionally "
            "normalizes pickup/dropoff positions by cumulative travel cost; "
            "independent-nested keeps the former node-index geometry; "
            "legacy-shared-pool reproduces the dataset-wide shared pool."
        ),
    )
    parser.add_argument("--reuse-query-bank", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild aggregate CSV files from all existing raw result rows",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = make_parser().parse_args(argv)
    workdir = Path(args.workdir).resolve()
    output_dir = (workdir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "kprun_efficiency_results.csv"
    summary_path = output_dir / "kprun_efficiency_summary.csv"
    pair_path = output_dir / "kprun_efficiency_pairs.csv"
    pair_summary_path = output_dir / "kprun_efficiency_speedup_summary.csv"
    query_bank_path = (
        Path(args.query_bank).resolve()
        if args.query_bank
        else output_dir / "efficiency_queries.jsonl"
    )

    if args.summarize_only:
        existing_rows = read_csv(result_path)
        write_aggregate_outputs(
            existing_rows,
            summary_path,
            pair_path,
            pair_summary_path,
        )
        print(f"Summarized {len(existing_rows)} rows from {result_path}")
        print(f"Saved {summary_path}")
        print(f"Saved {pair_summary_path}")
        return 0

    request_sizes = parse_int_list(args.request_sizes)
    if max(request_sizes) > 25:
        print(
            "WARNING: request_size is the active combinatorial search size; "
            "values above 25 are likely to time out."
        )
    selected_names = set(parse_name_list(args.datasets))
    configs = [
        config
        for config in load_configs(workdir / args.config)
        if config.name.upper() in selected_names
    ]

    if (args.reuse_query_bank or args.query_bank) and query_bank_path.exists():
        queries = [
            query
            for query in read_query_bank(query_bank_path)
            if query.dataset.upper() in selected_names
            and query.request_size in request_sizes
        ]
    else:
        if args.query_bank_mode == "independent-cost-nested":
            queries = generate_cost_normalized_nested_query_bank(
                workdir,
                configs,
                request_sizes,
                args.queries_per_size,
                args.seed,
            )
        elif args.query_bank_mode == "independent-nested":
            queries = generate_independent_nested_query_bank(
                workdir,
                configs,
                request_sizes,
                args.queries_per_size,
                args.seed,
            )
        else:
            queries = generate_query_bank(
                workdir,
                configs,
                request_sizes,
                args.queries_per_size,
                args.seed,
            )
        write_query_bank(query_bank_path, queries)
    selected_query_ids = set(parse_name_list(args.query_ids))
    if selected_query_ids:
        queries = [query for query in queries if query.query_id in selected_query_ids]
        missing_query_ids = selected_query_ids - {query.query_id for query in queries}
        if missing_query_ids:
            raise ValueError(
                f"query IDs not found after dataset/size filtering: "
                f"{sorted(missing_query_ids)}"
            )
    print(
        f"Prepared {len(queries)} queries in {query_bank_path} "
        f"(mode={args.query_bank_mode})"
    )
    if args.generate_only:
        return 0

    existing = [] if args.no_resume else read_csv(result_path)
    rows_by_key: Dict[Tuple[str, str, str], Dict[str, object]] = {
        (row["query_id"], row["method"], row["run_fingerprint"]): row
        for row in existing
    }
    configs_by_name = {config.name: config for config in configs}
    grouped: Dict[str, List[QueryInstance]] = {}
    for query in queries:
        grouped.setdefault(query.dataset, []).append(query)

    for dataset, dataset_queries in grouped.items():
        config = configs_by_name[dataset]
        print(f"Loading {dataset} graph...")
        graph = read_edge_list(
            str(workdir / config.edge_file),
            directed=config.directed,
            ignore_first_col=config.ignore_first_col,
        )
        cache: Dict[Tuple[str, str], Tuple[List[str], float]] = {}
        for index, query in enumerate(dataset_queries, start=1):
            run_id = result_fingerprint(
                query,
                args.capacity,
                args.k,
                args.detour_ratio,
                args.max_states,
                args.measure_memory,
                args.timeout,
                args.cost_tolerance,
                args.dp_mode,
            )
            keys = [
                (query.query_id, method, run_id)
                for method in ["KPrun-Search", "DPKMOR"]
            ]
            if all(
                key in rows_by_key
                and rows_by_key[key].get("status") == COMPLETE
                for key in keys
            ):
                label = "resumed"
            else:
                try:
                    pair_rows = run_pair(
                        graph,
                        query,
                        cache,
                        args.capacity,
                        args.k,
                        args.detour_ratio,
                        args.max_states,
                        args.timeout,
                        args.cost_tolerance,
                        args.measure_memory,
                        args.dp_mode,
                    )
                except MemoryError:
                    pair_rows = failed_method_rows(query, run_id, OOM)
                for row in pair_rows:
                    stale_keys = [
                        key
                        for key in rows_by_key
                        if key[0] == query.query_id
                        and key[1] == row["method"]
                        and key[2] != run_id
                    ]
                    for stale_key in stale_keys:
                        del rows_by_key[stale_key]
                    rows_by_key[(query.query_id, row["method"], run_id)] = row
                write_csv(
                    result_path,
                    sorted(
                        rows_by_key.values(),
                        key=lambda row: (
                            str(row["dataset"]),
                            int(row["request_size"]),
                            str(row["query_id"]),
                            str(row["method"]),
                        ),
                    ),
                    RESULT_FIELDS,
                )
                label = "completed"
            print(
                f"[{dataset} {index}/{len(dataset_queries)}] "
                f"{query.query_id} {label}"
            )

    current_rows = []
    for query in queries:
        run_id = result_fingerprint(
            query,
            args.capacity,
            args.k,
            args.detour_ratio,
            args.max_states,
            args.measure_memory,
            args.timeout,
            args.cost_tolerance,
            args.dp_mode,
        )
        for method in ["KPrun-Search", "DPKMOR"]:
            row = rows_by_key.get((query.query_id, method, run_id))
            if row is not None:
                current_rows.append(row)

    write_aggregate_outputs(
        current_rows,
        summary_path,
        pair_path,
        pair_summary_path,
    )
    print(f"Saved {result_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {pair_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
