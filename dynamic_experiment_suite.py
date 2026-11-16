#!/usr/bin/env python3
"""Run dynamic KMOR workloads and export paper-ready CSV tables."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import multiprocessing as mp
import queue as queue_module
import statistics
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from Dpkmor import read_edge_list, parse_requests_file
from dynamic_kmor import (
    DynamicKMORManager,
    RequestStatus,
    TimeDependentGraph,
    UpdateStrategy,
    parse_event_file,
)


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def percentile(values: Sequence[float], percentage: float) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentage
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def validate_manager(manager: DynamicKMORManager) -> Tuple[bool, str]:
    capacity = int(manager.planner_params.get("capacity", 4))
    time_budget = float(manager.planner_params.get("time_budget", math.inf))
    detour_ratio = float(manager.planner_params.get("detour_ratio", 1.8))
    enforce_detour = bool(
        manager.planner_params.get("enforce_passenger_detour", True)
    )
    active_statuses = {RequestStatus.WAITING, RequestStatus.ONBOARD}
    errors: List[str] = []

    for name in manager.active_requests:
        if manager.request_status.get(name) not in active_statuses:
            errors.append(f"active request {name} has invalid status")

    onboard_names = {
        name
        for name, status in manager.request_status.items()
        if status == RequestStatus.ONBOARD
    }
    onboard_load = sum(
        manager.active_requests[name].demand
        for name in onboard_names
        if name in manager.active_requests
    )
    if onboard_load > capacity:
        errors.append(f"vehicle load {onboard_load} exceeds capacity {capacity}")

    for route_index, candidate in enumerate(manager.routes):
        if route_index >= len(manager.route_records):
            errors.append(f"route {route_index} has no dynamic route record")
            continue
        record = manager.route_records[route_index]
        if not candidate.key_path or candidate.key_path[0] != manager.vehicle.node:
            errors.append(f"route {route_index} starts from the wrong vehicle node")
        if not candidate.full_path or candidate.full_path[0] != manager.vehicle.node:
            errors.append(f"route {route_index} full path starts from the wrong node")

        reconstructed_path = [manager.vehicle.node]
        previous_node = manager.vehicle.node
        previous_arrival = float(record.segments[0].depart_time) if record.segments else candidate.total_cost
        for segment in record.segments:
            if segment.u != previous_node:
                errors.append(f"route {route_index} contains a disconnected physical path")
                break
            expected_weight = manager.graph.edge_weight_at(
                segment.u,
                segment.v,
                segment.depart_time,
            )
            if not math.isfinite(expected_weight):
                errors.append(f"route {route_index} uses an unreachable edge")
                break
            if abs(expected_weight - segment.travel_time) > 1e-7:
                errors.append(f"route {route_index} has an inconsistent edge cost")
            if segment.depart_time + 1e-7 < previous_arrival:
                errors.append(f"route {route_index} has non-monotone segment times")
            if abs(
                segment.arrive_time
                - segment.depart_time
                - segment.travel_time
            ) > 1e-7:
                errors.append(f"route {route_index} has an inconsistent arrival time")
            reconstructed_path.append(segment.v)
            previous_node = segment.v
            previous_arrival = segment.arrive_time

        if reconstructed_path != candidate.full_path:
            errors.append(f"route {route_index} full path differs from its execution record")
        if candidate.hops != max(0, len(candidate.full_path) - 1):
            errors.append(f"route {route_index} has an inconsistent hop count")

        expected_stops = list(candidate.key_path[1:])
        actual_stops = [stop.node for stop in record.stops]
        if actual_stops != expected_stops:
            errors.append(f"route {route_index} stop nodes are inconsistent")
        if record.stops:
            if abs(record.stops[-1].event_time - candidate.total_cost) > 1e-7:
                errors.append(f"route {route_index} total cost is inconsistent")

        onboard = set(onboard_names)
        load = onboard_load
        actual_pickup_times = dict(manager.vehicle.pickup_times)
        pickup_info = {name: (request_time, pickup_time, wait_time)
                       for name, request_time, pickup_time, wait_time
                       in candidate.pickup_info}
        dropped = []
        for event_index, (action, name) in enumerate(candidate.route_events):
            if action == "target":
                continue
            request = manager.active_requests.get(name)
            if request is None:
                errors.append(f"route {route_index} references inactive request {name}")
                continue
            if action == "pickup":
                if name in onboard:
                    errors.append(f"route {route_index} picks {name} twice")
                    continue
                load += request.demand
                onboard.add(name)
                if load > capacity:
                    errors.append(f"route {route_index} exceeds capacity")
                info = pickup_info.get(name)
                if info is None or info[2] > request.wait_limit + 1e-7:
                    errors.append(f"route {route_index} violates wait limit for {name}")
                matching_stops = [
                    stop
                    for stop in record.stops
                    if stop.action == "pickup" and stop.request_name == name
                ]
                if (
                    not matching_stops
                    or matching_stops[0].event_time - request.request_time
                    > request.wait_limit + 1e-7
                ):
                    errors.append(
                        f"route {route_index} physical schedule violates wait limit for {name}"
                    )
                if matching_stops:
                    actual_pickup_times[name] = matching_stops[0].event_time
            elif action == "dropoff":
                if name not in onboard:
                    errors.append(f"route {route_index} drops {name} before pickup")
                    continue
                if enforce_detour:
                    pickup_time = actual_pickup_times.get(name)
                    dropoff_time = (
                        record.stops[event_index].event_time
                        if event_index < len(record.stops)
                        else math.inf
                    )
                    _, direct_cost = manager.cached_lower_bound_shortest_path(
                        request.pickup,
                        request.dropoff,
                    )
                    if (
                        pickup_time is None
                        or not math.isfinite(direct_cost)
                        or dropoff_time - pickup_time
                        > detour_ratio * direct_cost + 1e-7
                    ):
                        errors.append(
                            f"route {route_index} violates beta detour for {name}"
                        )
                load -= request.demand
                onboard.remove(name)
                dropped.append(name)

        if onboard:
            errors.append(f"route {route_index} leaves onboard requests undelivered")
        if set(dropped) != set(candidate.served_requests):
            errors.append(f"route {route_index} served-request list is inconsistent")
        if candidate.total_cost > time_budget + 1e-7:
            errors.append(f"route {route_index} exceeds time budget")

    return not errors, "; ".join(errors[:5])


def route_diversity_metrics(manager: DynamicKMORManager) -> Dict[str, object]:
    requested_k = int(manager.planner_params.get("k", 5))
    available = manager.available_routes or manager.routes
    return {
        "requested_k": requested_k,
        "unique_service_plans": len(
            {tuple(candidate.route_events) for candidate in manager.routes}
        ),
        "unique_physical_paths": len(
            {tuple(candidate.full_path) for candidate in manager.routes}
        ),
        "topk_complete": len(manager.routes) >= requested_k,
        "backup_route_count": len(manager.backup_routes),
        "available_route_count": len(available),
        "available_unique_service_plans": len(
            {tuple(candidate.route_events) for candidate in available}
        ),
        "available_unique_physical_paths": len(
            {tuple(candidate.full_path) for candidate in available}
        ),
        "available_topk_complete": len(available) >= requested_k,
        "deviation_attempts": manager.last_deviation_attempts,
        "deviation_new_routes": manager.last_deviation_new_routes,
        "backup_generated_routes": manager.last_backup_new_routes,
    }


def worker_run(task: Mapping[str, object], result_queue) -> None:
    started = time.perf_counter()
    workload = dict(task["workload"])  # type: ignore[arg-type]
    scenario_context = {
        "scenario": workload.get("scenario", "baseline"),
        "sweep": workload.get("sweep", "baseline"),
        "parameter": workload.get("parameter", "baseline"),
        "parameter_value": workload.get("parameter_value", 1.0),
    }
    try:
        workdir = Path(str(task["workdir"]))
        strategy = str(task["strategy"])
        edge_file = workdir / str(workload["edge_file"])
        request_file = workdir / str(workload["initial_requests_file"])
        event_file = workdir / str(workload["events_file"])

        load_started = time.perf_counter()
        static_graph = read_edge_list(
            str(edge_file),
            directed=bool(workload.get("directed", False)),
            ignore_first_col=bool(workload.get("ignore_first_col", True)),
        )
        dynamic_graph = TimeDependentGraph.from_static_graph(
            static_graph,
            bucket_size=float(task["bucket_size"]),
        )
        del static_graph
        gc.collect()
        requests = parse_requests_file(str(request_file))
        events = parse_event_file(event_file)
        max_events = int(task.get("max_events", 0))
        if max_events > 0:
            events = events[:max_events]
        preprocess_time = time.perf_counter() - load_started

        manager = DynamicKMORManager(
            dynamic_graph,
            str(workload["source"]),
            str(workload["target"]),
            requests,
            strategy=strategy,
            capacity=int(task["capacity"]),
            k=int(task["k"]),
            time_budget=float(workload["time_budget"]),
            detour_ratio=float(task["detour_ratio"]),
            max_states=int(task["max_states"]),
            archive_factor=int(task["archive_factor"]),
            replacement_limit_per_route=int(task["replacement_limit_per_route"]),
            arrival_companion_limit=int(task["arrival_companion_limit"]),
            arrival_batch_sequence_limit=int(task["arrival_batch_sequence_limit"]),
            enable_arrival_local_reoptimization=bool(
                task["enable_arrival_local_reoptimization"]
            ),
            arrival_local_reopt_request_limit=int(
                task["arrival_local_reopt_request_limit"]
            ),
            arrival_local_reopt_max_states=int(
                task["arrival_local_reopt_max_states"]
            ),
            refill_request_limit=int(task["refill_request_limit"]),
            cancel_refill_depth=int(task["cancel_refill_depth"]),
            enable_request_replacement=bool(task["enable_request_replacement"]),
            enable_cancel_refill=bool(task["enable_cancel_refill"]),
            adaptive_recompute_on_failed_arrival=bool(
                task["adaptive_recompute_on_failed_arrival"]
            ),
            adaptive_recompute_on_archive_depletion=bool(
                task["adaptive_recompute_on_archive_depletion"]
            ),
            archive_depletion_ratio=float(task["archive_depletion_ratio"]),
            enable_backup_routes=bool(task["enable_backup_routes"]),
            backup_coverage_gap=int(task["backup_coverage_gap"]),
            backup_cost_regret_limit=float(task["backup_cost_regret_limit"]),
            backup_deviation_attempts=int(task["backup_deviation_attempts"]),
            backup_event_order_attempts=int(task["backup_event_order_attempts"]),
            full_recompute_reuse_tdsp_cache=bool(
                task.get("full_recompute_reuse_tdsp_cache", False)
            ),
            periodic_recompute_interval=int(task["periodic_recompute_interval"]),
            strict_static_semantics=bool(task["strict_static_semantics"]),
            enable_certified_recovery=bool(
                task.get("enable_certified_recovery", True)
            ),
        )

        rows: List[Dict[str, object]] = []
        initialize_result = manager.initialize(0.0)
        valid, error = validate_manager(manager)
        def certified_status(result, constraint_valid: bool) -> str:
            if not constraint_valid:
                return "INVALID"
            if result.timed_out:
                return "TIMEOUT"
            if result.hit_max_states:
                return "STATE_LIMIT"
            if not result.objective_complete or not result.concrete_complete:
                return "INCOMPLETE"
            return "OK"

        first_row: Dict[str, object] = {
            **scenario_context,
            "dataset": workload["dataset"],
            "seed": workload["seed"],
            "event_index": 0,
            "status": certified_status(initialize_result, valid),
            **asdict(initialize_result),
            **route_diversity_metrics(manager),
            "constraint_valid": valid,
            "constraint_error": error,
        }
        rows.append(first_row)

        for event_index, event in enumerate(events, start=1):
            result = manager.handle_event(event)
            valid, error = validate_manager(manager)
            rows.append(
                {
                    **scenario_context,
                    "dataset": workload["dataset"],
                    "seed": workload["seed"],
                    "event_index": event_index,
                    "status": certified_status(result, valid),
                    **asdict(result),
                    **route_diversity_metrics(manager),
                    "constraint_valid": valid,
                    "constraint_error": error,
                }
            )

        result_queue.put(
            {
                "status": (
                    "OK"
                    if all(row["status"] == "OK" for row in rows)
                    else next(
                        str(row["status"])
                        for row in rows
                        if row["status"] != "OK"
                    )
                ),
                **scenario_context,
                "dataset": workload["dataset"],
                "seed": workload["seed"],
                "strategy": strategy,
                "preprocess_time": preprocess_time,
                "initialization_time": initialize_result.update_time,
                "wall_time": time.perf_counter() - started,
                "rows": rows,
            }
        )
    except MemoryError:
        result_queue.put(
            {
                "status": "OOM",
                **scenario_context,
                "dataset": workload.get("dataset", ""),
                "seed": workload.get("seed", ""),
                "strategy": task.get("strategy", ""),
                "error": "MemoryError",
                "rows": [],
            }
        )
    except BaseException as exc:
        result_queue.put(
            {
                "status": "ERROR",
                **scenario_context,
                "dataset": workload.get("dataset", ""),
                "seed": workload.get("seed", ""),
                "strategy": task.get("strategy", ""),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "rows": [],
            }
        )


def run_isolated(task: Mapping[str, object], timeout: float) -> Dict[str, object]:
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=worker_run, args=(task, result_queue))
    process.start()
    workload = dict(task["workload"])  # type: ignore[arg-type]
    deadline = time.monotonic() + timeout
    result: Optional[Dict[str, object]] = None

    # Consume the queue while the worker is alive. Joining first can deadlock
    # on Windows when the queue payload exceeds the OS pipe buffer.
    while result is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            process.terminate()
            process.join(10)
            result_queue.close()
            return {
                "status": "INF",
                "scenario": workload.get("scenario", "baseline"),
                "sweep": workload.get("sweep", "baseline"),
                "parameter": workload.get("parameter", "baseline"),
                "parameter_value": workload.get("parameter_value", 1.0),
                "dataset": workload["dataset"],
                "seed": workload["seed"],
                "strategy": task["strategy"],
                "error": f"timeout after {timeout:g}s",
                "rows": [],
            }
        try:
            result = result_queue.get(timeout=min(1.0, remaining))
        except queue_module.Empty:
            if process.is_alive():
                continue
            try:
                result = result_queue.get(timeout=1)
            except queue_module.Empty:
                result = {
                    "status": "ERROR",
                    "scenario": workload.get("scenario", "baseline"),
                    "sweep": workload.get("sweep", "baseline"),
                    "parameter": workload.get("parameter", "baseline"),
                    "parameter_value": workload.get("parameter_value", 1.0),
                    "dataset": workload["dataset"],
                    "seed": workload["seed"],
                    "strategy": task["strategy"],
                    "error": (
                        f"worker exited with code {process.exitcode} "
                        "without a result"
                    ),
                    "rows": [],
                }

    process.join(10)
    if process.is_alive():
        process.terminate()
        process.join(10)
    result_queue.close()
    return result


def summarize_payload(payload: Mapping[str, object]) -> List[Dict[str, object]]:
    status = str(payload["status"])
    base = {
        "scenario": payload.get("scenario", "baseline"),
        "sweep": payload.get("sweep", "baseline"),
        "parameter": payload.get("parameter", "baseline"),
        "parameter_value": payload.get("parameter_value", 1.0),
        "dataset": payload.get("dataset", ""),
        "seed": payload.get("seed", ""),
        "strategy": payload.get("strategy", ""),
        "status": status,
        "preprocess_time": payload.get("preprocess_time", math.inf),
        "initialization_time": payload.get("initialization_time", math.inf),
        "wall_time": payload.get("wall_time", math.inf),
        "error": payload.get("error", ""),
    }
    rows = list(payload.get("rows", []))  # type: ignore[arg-type]
    if status != "OK" or not rows:
        return [{**base, "event_type": "ALL", "event_count": 0}]

    event_rows = [row for row in rows if row["event_type"] != "Initialize"]
    event_types = ["ALL", "RequestArrival", "EdgeUpdate", "RequestCancel"]
    summaries = []
    final_row = rows[-1]
    for event_type in event_types:
        selected = (
            event_rows
            if event_type == "ALL"
            else [row for row in event_rows if row["event_type"] == event_type]
        )
        latencies = [float(row["update_time"]) for row in selected]
        recompute_times = [float(row["recompute_time"]) for row in selected]
        def diagnostic_total(field: str) -> int:
            return sum(int(row.get(field, 0) or 0) for row in selected)

        tdsp_queries = diagnostic_total("tdsp_queries")
        tdsp_cache_hits = diagnostic_total("tdsp_cache_hits")
        lower_bound_queries = sum(
            int(row.get("lower_bound_queries", 0) or 0) for row in selected
        )
        lower_bound_cache_hits = sum(
            int(row.get("lower_bound_cache_hits", 0) or 0) for row in selected
        )
        summaries.append(
            {
                **base,
                "event_type": event_type,
                "event_count": len(selected),
                "mean_event_time": statistics.fmean(latencies) if latencies else math.inf,
                "median_event_time": statistics.median(latencies) if latencies else math.inf,
                "p95_event_time": percentile(latencies, 0.95),
                "p99_event_time": percentile(latencies, 0.99),
                "max_event_time": max(latencies) if latencies else math.inf,
                "mean_recompute_time": (
                    statistics.fmean(recompute_times) if recompute_times else math.inf
                ),
                "throughput_events_per_second": (
                    len(selected) / sum(latencies)
                    if latencies and sum(latencies) > 0
                    else 0.0
                ),
                "final_served": final_row["best_served"],
                "final_cost": final_row["best_cost"],
                "final_cost_per_served": (
                    float(final_row["best_cost"]) / int(final_row["best_served"])
                    if int(final_row["best_served"]) > 0
                    else math.inf
                ),
                "mean_route_count": (
                    statistics.fmean(float(row["route_count"]) for row in selected)
                    if selected
                    else 0.0
                ),
                "mean_available_route_count": (
                    statistics.fmean(
                        float(row["available_route_count"]) for row in selected
                    )
                    if selected
                    else 0.0
                ),
                "mean_backup_route_count": (
                    statistics.fmean(
                        float(row["backup_route_count"]) for row in selected
                    )
                    if selected
                    else 0.0
                ),
                "mean_unique_service_plans": (
                    statistics.fmean(
                        float(row["unique_service_plans"]) for row in selected
                    )
                    if selected
                    else 0.0
                ),
                "mean_unique_physical_paths": (
                    statistics.fmean(
                        float(row["unique_physical_paths"]) for row in selected
                    )
                    if selected
                    else 0.0
                ),
                "topk_complete_rate": (
                    sum(bool(row["topk_complete"]) for row in selected)
                    / len(selected)
                    if selected
                    else 0.0
                ),
                "available_topk_complete_rate": (
                    sum(bool(row["available_topk_complete"]) for row in selected)
                    / len(selected)
                    if selected
                    else 0.0
                ),
                "deviation_attempts": sum(
                    int(row["deviation_attempts"]) for row in selected
                ),
                "deviation_new_routes": sum(
                    int(row["deviation_new_routes"]) for row in selected
                ),
                "expanded_states": sum(int(row["expanded_states"]) for row in selected),
                "dp_pruned_states": sum(int(row["dp_pruned_states"]) for row in selected),
                "affected_routes": sum(int(row["affected_routes"]) for row in selected),
                "tdsp_queries": tdsp_queries,
                "tdsp_cache_hits": tdsp_cache_hits,
                "tdsp_cache_hit_rate": (
                    tdsp_cache_hits / tdsp_queries if tdsp_queries else 0.0
                ),
                "tdsp_lower_path_reuses": diagnostic_total(
                    "tdsp_lower_path_reuses"
                ),
                "tdsp_dijkstra_runs": diagnostic_total("tdsp_dijkstra_runs"),
                "tdsp_cache_invalidations": diagnostic_total(
                    "tdsp_cache_invalidations"
                ),
                "lower_bound_queries": lower_bound_queries,
                "lower_bound_cache_hits": lower_bound_cache_hits,
                "lower_bound_cache_hit_rate": (
                    lower_bound_cache_hits / lower_bound_queries
                    if lower_bound_queries
                    else 0.0
                ),
                "lower_bound_dijkstra_runs": diagnostic_total(
                    "lower_bound_dijkstra_runs"
                ),
                "deviation_queries": diagnostic_total("deviation_queries"),
                "deviation_cache_hits": diagnostic_total("deviation_cache_hits"),
                "deviation_dijkstra_runs": diagnostic_total(
                    "deviation_dijkstra_runs"
                ),
                "insertion_transitions_tested": diagnostic_total(
                    "insertion_transitions_tested"
                ),
                "insertion_labels_generated": diagnostic_total(
                    "insertion_labels_generated"
                ),
                "insertion_dominance_checks": diagnostic_total(
                    "insertion_dominance_checks"
                ),
                "insertion_resource_checks": diagnostic_total(
                    "insertion_resource_checks"
                ),
                "dependency_lookups": diagnostic_total("dependency_lookups"),
                "dependency_candidates": diagnostic_total(
                    "dependency_candidates"
                ),
                "plan_dag_builds": diagnostic_total("plan_dag_builds"),
                "plan_dag_reuses": diagnostic_total("plan_dag_reuses"),
                "plan_prefix_labels_evaluated": diagnostic_total(
                    "plan_prefix_labels_evaluated"
                ),
                "plan_prefix_transitions_saved": diagnostic_total(
                    "plan_prefix_transitions_saved"
                ),
                "plan_prefix_cache_hits": diagnostic_total(
                    "plan_prefix_cache_hits"
                ),
                "plan_prefix_labels_reopened": diagnostic_total(
                    "plan_prefix_labels_reopened"
                ),
                "plan_prefix_invalidations": diagnostic_total(
                    "plan_prefix_invalidations"
                ),
                "local_reoptimization_runs": diagnostic_total(
                    "local_reoptimization_runs"
                ),
                "local_reoptimization_requests": diagnostic_total(
                    "local_reoptimization_requests"
                ),
                "local_reoptimization_expanded": diagnostic_total(
                    "local_reoptimization_expanded"
                ),
                "fallback_count": sum(
                    bool(row["fallback_recompute"]) for row in selected
                ),
                "fallback_rate": (
                    sum(bool(row["fallback_recompute"]) for row in selected)
                    / len(selected)
                    if selected
                    else 0.0
                ),
                "objective_certificate_rate": (
                    sum(bool(row.get("objective_complete")) for row in selected)
                    / len(selected)
                    if selected
                    else 0.0
                ),
                "concrete_certificate_rate": (
                    sum(bool(row.get("concrete_complete")) for row in selected)
                    / len(selected)
                    if selected
                    else 0.0
                ),
                "direct_certificate_acceptance_rate": (
                    sum(
                        row.get("certificate_stage") == "incremental"
                        for row in selected
                    )
                    / len(selected)
                    if selected
                    else 0.0
                ),
                "inherited_certificate_rate": (
                    sum(
                        row.get("certificate_stage") == "inherited"
                        for row in selected
                    )
                    / len(selected)
                    if selected
                    else 0.0
                ),
                "local_certificate_acceptance_rate": (
                    sum(
                        row.get("certificate_stage") == "local"
                        for row in selected
                    )
                    / len(selected)
                    if selected
                    else 0.0
                ),
                "full_recovery_rate": (
                    sum(
                        row.get("certificate_stage") == "full"
                        for row in selected
                    )
                    / len(selected)
                    if selected
                    else 0.0
                ),
                "local_reoptimization_rate": (
                    sum(
                        int(row.get("local_reoptimization_runs", 0) or 0) > 0
                        for row in selected
                    )
                    / len(selected)
                    if selected
                    else 0.0
                ),
                "constraint_violations": sum(
                    not bool(row["constraint_valid"]) for row in selected
                ),
            }
        )
    return summaries


def build_quality_rows(
    payloads: Sequence[Mapping[str, object]],
    summaries: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    summary_index = {
        (row["scenario"], row["dataset"], row["seed"], row["strategy"]): row
        for row in summaries
        if row.get("event_type") == "ALL"
    }
    workload_keys = sorted(
        {(row["scenario"], row["dataset"], row["seed"]) for row in summaries}
    )
    quality_rows = []
    for scenario, dataset, seed in workload_keys:
        incremental = summary_index.get(
            (scenario, dataset, seed, UpdateStrategy.INCREMENTAL.value)
        )
        full = summary_index.get(
            (scenario, dataset, seed, UpdateStrategy.FULL_RECOMPUTE.value)
        )
        if incremental is None or full is None:
            continue
        status = "OK" if incremental["status"] == full["status"] == "OK" else "INF"
        inc_served = int(incremental.get("final_served", 0) or 0)
        full_served = int(full.get("final_served", 0) or 0)
        inc_cost = float(incremental.get("final_cost", math.inf))
        full_cost = float(full.get("final_cost", math.inf))
        same_served = inc_served == full_served
        quality_rows.append(
            {
                "scenario": scenario,
                "sweep": incremental.get("sweep", "baseline"),
                "parameter": incremental.get("parameter", "baseline"),
                "parameter_value": incremental.get("parameter_value", 1.0),
                "dataset": dataset,
                "seed": seed,
                "status": status,
                "incremental_final_served": inc_served,
                "full_final_served": full_served,
                "served_gap": inc_served - full_served,
                "incremental_final_cost": inc_cost,
                "full_final_cost": full_cost,
                "cost_regret": (
                    (inc_cost - full_cost) / full_cost
                    if same_served and finite(inc_cost) and finite(full_cost) and full_cost > 0
                    else ""
                ),
                "incremental_mean_event_time": incremental.get("mean_event_time", math.inf),
                "full_mean_event_time": full.get("mean_event_time", math.inf),
                "speedup": (
                    float(full["mean_event_time"]) / float(incremental["mean_event_time"])
                    if finite(full.get("mean_event_time"))
                    and finite(incremental.get("mean_event_time"))
                    and float(incremental["mean_event_time"]) > 0
                    else ""
                ),
                "incremental_fallback_rate": incremental.get("fallback_rate", math.inf),
                "incremental_constraint_violations": incremental.get(
                    "constraint_violations", math.inf
                ),
                "full_constraint_violations": full.get("constraint_violations", math.inf),
            }
        )
    return quality_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        if path.exists():
            path.unlink()
        return
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect_event_rows(
    payloads: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for payload in payloads:
        payload_rows = list(payload.get("rows", []))  # type: ignore[arg-type]
        if payload_rows:
            rows.extend(payload_rows)
            continue
        rows.append(
            {
                "scenario": payload.get("scenario", "baseline"),
                "sweep": payload.get("sweep", "baseline"),
                "parameter": payload.get("parameter", "baseline"),
                "parameter_value": payload.get("parameter_value", 1.0),
                "dataset": payload.get("dataset", ""),
                "seed": payload.get("seed", ""),
                "event_index": "",
                "status": payload.get("status", "ERROR"),
                "event_type": "ALL",
                "strategy": payload.get("strategy", ""),
                "update_time": math.inf,
                "constraint_valid": "",
                "constraint_error": payload.get("error", ""),
            }
        )
    return rows


def persist_outputs(
    output_dir: Path,
    manifest_path: Path,
    arguments: Mapping[str, object],
    payloads: Sequence[Mapping[str, object]],
) -> None:
    event_rows = collect_event_rows(payloads)
    summaries = [
        summary
        for payload in payloads
        for summary in summarize_payload(payload)
    ]
    quality_rows = build_quality_rows(payloads, summaries)
    write_csv(output_dir / "dynamic_event_results.csv", event_rows)
    write_csv(output_dir / "dynamic_summary_results.csv", summaries)
    write_csv(output_dir / "dynamic_quality_results.csv", quality_rows)
    metadata = {
        "manifest": str(manifest_path),
        "arguments": dict(arguments),
        "completed_runs": len(payloads),
        "runs": [
            {
                key: payload.get(key)
                for key in (
                    "scenario",
                    "sweep",
                    "parameter",
                    "parameter_value",
                    "dataset",
                    "seed",
                    "strategy",
                    "status",
                    "preprocess_time",
                    "initialization_time",
                    "wall_time",
                    "error",
                    "traceback",
                )
                if payload.get(key) is not None
            }
            for payload in payloads
        ],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def parse_values(text: str, cast) -> List:
    return [cast(value.strip()) for value in text.split(",") if value.strip()]


def code_fingerprint(workdir: Path) -> str:
    digest = hashlib.sha256()
    for filename in ("Dpkmor.py", "dynamic_kmor.py", "dynamic_experiment_suite.py"):
        path = workdir / filename
        digest.update(filename.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dynamic KMOR paper experiments")
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--manifest", default="dynamic_workloads/manifest.json")
    parser.add_argument("--output-dir", default="paper_results_dynamic")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--scenarios", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument(
        "--strategies",
        default="incremental,full_recompute",
        help="Comma-separated UpdateStrategy values",
    )
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--detour-ratio", type=float, default=1.4)
    parser.add_argument("--max-states", type=int, default=30000)
    parser.add_argument("--bucket-size", type=float, default=300.0)
    parser.add_argument("--archive-factor", type=int, default=3)
    parser.add_argument("--replacement-limit-per-route", type=int, default=2)
    parser.add_argument("--arrival-companion-limit", type=int, default=3)
    parser.add_argument("--arrival-batch-sequence-limit", type=int, default=512)
    parser.add_argument("--arrival-local-reopt-request-limit", type=int, default=12)
    parser.add_argument("--arrival-local-reopt-max-states", type=int, default=30000)
    parser.add_argument("--disable-arrival-local-reoptimization", action="store_true")
    parser.add_argument("--refill-request-limit", type=int, default=5)
    parser.add_argument("--cancel-refill-depth", type=int, default=2)
    parser.add_argument("--archive-depletion-ratio", type=float, default=0.5)
    parser.add_argument("--disable-backup-routes", action="store_true")
    parser.add_argument("--backup-coverage-gap", type=int, default=1)
    parser.add_argument("--backup-cost-regret-limit", type=float, default=0.10)
    parser.add_argument("--backup-deviation-attempts", type=int, default=60)
    parser.add_argument("--backup-event-order-attempts", type=int, default=20)
    parser.add_argument(
        "--full-recompute-warm-tdsp-cache",
        action="store_true",
        help="Retain event-dependent TDSP answers across full recomputations.",
    )
    parser.add_argument("--disable-request-replacement", action="store_true")
    parser.add_argument("--disable-cancel-refill", action="store_true")
    parser.add_argument("--disable-adaptive-recompute", action="store_true")
    parser.add_argument("--periodic-recompute-interval", type=int, default=5)
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Process only the first N events of each workload (0 means all)",
    )
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument(
        "--legacy-dynamic-semantics",
        action="store_true",
        help=(
            "Reproduce the earlier dynamic experiments that filled dominated "
            "Top-k routes and did not enforce passenger detour limits"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse matching per-run checkpoints from the output directory",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workdir = Path(args.workdir).resolve()
    manifest_path = (workdir / args.manifest).resolve()
    output_dir = (workdir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current_code_fingerprint = code_fingerprint(workdir)
    selected_datasets = {name.upper() for name in parse_values(args.datasets, str)}
    selected_scenarios = {name.lower() for name in parse_values(args.scenarios, str)}
    selected_seeds = set(parse_values(args.seeds, int))
    strategies = parse_values(args.strategies, str)
    for strategy in strategies:
        UpdateStrategy(strategy)

    workloads = []
    for workload in manifest["workloads"]:
        if (
            selected_scenarios
            and str(workload.get("scenario", "baseline")).lower()
            not in selected_scenarios
        ):
            continue
        if selected_datasets and str(workload["dataset"]).upper() not in selected_datasets:
            continue
        if selected_seeds and int(workload["seed"]) not in selected_seeds:
            continue
        workloads.append(workload)

    payloads: List[Dict[str, object]] = []
    for workload in workloads:
        for strategy in strategies:
            task = {
                "workdir": str(workdir),
                "workload": workload,
                "strategy": strategy,
                "capacity": args.capacity,
                "k": args.k,
                "detour_ratio": args.detour_ratio,
                "max_states": args.max_states,
                "bucket_size": args.bucket_size,
                "archive_factor": args.archive_factor,
                "replacement_limit_per_route": args.replacement_limit_per_route,
                "arrival_companion_limit": args.arrival_companion_limit,
                "arrival_batch_sequence_limit": args.arrival_batch_sequence_limit,
                "enable_arrival_local_reoptimization": (
                    not args.disable_arrival_local_reoptimization
                ),
                "arrival_local_reopt_request_limit": (
                    args.arrival_local_reopt_request_limit
                ),
                "arrival_local_reopt_max_states": args.arrival_local_reopt_max_states,
                "refill_request_limit": args.refill_request_limit,
                "cancel_refill_depth": args.cancel_refill_depth,
                "enable_request_replacement": not args.disable_request_replacement,
                "enable_cancel_refill": not args.disable_cancel_refill,
                "adaptive_recompute_on_failed_arrival": not args.disable_adaptive_recompute,
                "adaptive_recompute_on_archive_depletion": not args.disable_adaptive_recompute,
                "archive_depletion_ratio": args.archive_depletion_ratio,
                "enable_backup_routes": not args.disable_backup_routes,
                "backup_coverage_gap": args.backup_coverage_gap,
                "backup_cost_regret_limit": args.backup_cost_regret_limit,
                "backup_deviation_attempts": args.backup_deviation_attempts,
                "backup_event_order_attempts": args.backup_event_order_attempts,
                "full_recompute_reuse_tdsp_cache": (
                    args.full_recompute_warm_tdsp_cache
                ),
                "periodic_recompute_interval": args.periodic_recompute_interval,
                "max_events": args.max_events,
                "strict_static_semantics": not args.legacy_dynamic_semantics,
                "enable_certified_recovery": not args.legacy_dynamic_semantics,
                "code_fingerprint": current_code_fingerprint,
            }
            task_signature = json.dumps(task, sort_keys=True)
            digest = hashlib.sha256(task_signature.encode("utf-8")).hexdigest()[:16]
            checkpoint_path = checkpoint_dir / (
                f"{workload.get('scenario', 'baseline')}__{workload['dataset']}"
                f"__seed{workload['seed']}__{strategy}__{digest}.json"
            )
            if args.resume and checkpoint_path.exists():
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                payload = checkpoint["payload"]
                print(
                    f"[{workload.get('scenario', 'baseline')} {workload['dataset']} "
                    f"seed={workload['seed']} {strategy}] resumed "
                    f"status={payload['status']}",
                    flush=True,
                )
            else:
                print(
                    f"[{workload.get('scenario', 'baseline')} {workload['dataset']} "
                    f"seed={workload['seed']} {strategy}] running...",
                    flush=True,
                )
                payload = run_isolated(task, args.timeout)
                checkpoint_path.write_text(
                    json.dumps(
                        {
                            "task_signature": task_signature,
                            "payload": payload,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(
                    f"[{workload.get('scenario', 'baseline')} {workload['dataset']} "
                    f"seed={workload['seed']} {strategy}] "
                    f"status={payload['status']}",
                    flush=True,
                )
            payloads.append(payload)
            persist_outputs(
                output_dir,
                manifest_path,
                vars(args),
                payloads,
            )

    persist_outputs(output_dir, manifest_path, vars(args), payloads)
    print(f"Saved dynamic experiment results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
