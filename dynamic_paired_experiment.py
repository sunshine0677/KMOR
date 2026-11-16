#!/usr/bin/env python3
"""Paired same-state evaluation for online dynamic KMOR strategies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import queue as queue_module
import statistics
import time
import traceback
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from Dpkmor import candidate_identity, parse_requests_file, read_edge_list
from dynamic_experiment_suite import validate_manager
from dynamic_kmor import (
    DynamicKMORManager,
    EdgeUpdate,
    TimeDependentGraph,
    UpdateStrategy,
    parse_event_file,
    run_paired_event,
)
from dynamic_kinetic_tree import KineticTreeKMORManager
from dynamic_rhls import RHLSTopKManager
from dynamic_td_insertion import (
    TDInsertionKMORManager,
    TDTopKInsertionKMORManager,
)


EXPERIMENT_VARIANTS = {
    "full": "FullIncremental",
    "incremental_only": "IncrementalOnly",
    "no_local_reoptimization": "NoLocalReoptimization",
    "no_backup_maintenance": "NoBackupMaintenance",
}


def online_method_name(strategy: object, periodic_interval: object = 0) -> str:
    value = str(strategy)
    if value == UpdateStrategy.KINETIC_TREE.value:
        return "KineticTree-TopK"
    if value == UpdateStrategy.TD_INSERTION.value:
        return "TD-Insertion"
    if value == UpdateStrategy.TD_TOPK_INSERTION.value:
        return "TD-TopK-Insertion"
    if value == UpdateStrategy.RHLS_TOPK.value:
        return "RHLS-TopK"
    if value == UpdateStrategy.PERIODIC_RECOMPUTE.value:
        return f"Incremental+PeriodicRefresh-{periodic_interval}"
    return "Incremental"


def variant_runtime_settings(
    variant: str,
    *,
    local_reoptimization_enabled: bool,
    backup_routes_enabled: bool,
    adaptive_recompute_enabled: bool,
) -> Dict[str, bool]:
    """Resolve variant switches without changing the formal default method."""
    incremental_only = variant == "incremental_only"
    return {
        "enable_local_reoptimization": (
            local_reoptimization_enabled
            and variant not in {
                "incremental_only",
                "no_local_reoptimization",
            }
        ),
        "enable_backup_routes": (
            backup_routes_enabled and variant != "no_backup_maintenance"
        ),
        "enable_certified_recovery": not incremental_only,
        "require_incremental_certificate": not incremental_only,
        "enable_adaptive_recompute": (
            adaptive_recompute_enabled and not incremental_only
        ),
    }


def comparison_status(
    *,
    incremental_valid: bool,
    full_valid: bool,
    incremental_timed_out: bool,
    full_timed_out: bool,
    incremental_hit_max_states: bool,
    full_hit_max_states: bool,
    incremental_objective_complete: bool,
    incremental_concrete_complete: bool,
    full_objective_complete: bool,
    full_concrete_complete: bool,
    require_incremental_certificate: bool,
) -> str:
    """Classify a paired event while retaining non-exact diagnostics."""
    if not incremental_valid or not full_valid:
        return "INVALID"
    if incremental_timed_out or full_timed_out:
        return "TIMEOUT"
    if incremental_hit_max_states or full_hit_max_states:
        return "STATE_LIMIT"
    certificate_incomplete = not (
        full_objective_complete
        and full_concrete_complete
        and (
            not require_incremental_certificate
            or (
                incremental_objective_complete
                and incremental_concrete_complete
            )
        )
    )
    return "INCOMPLETE" if certificate_incomplete else "COMPLETE"


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def parse_values(text: str, cast) -> List:
    return [cast(value.strip()) for value in text.split(",") if value.strip()]


def objective_key(candidate) -> Tuple[int, float]:
    return candidate.served_count, round(float(candidate.total_cost), 7)


def service_key(candidate) -> Tuple[str, ...]:
    return tuple(sorted(candidate.served_requests))


def event_plan_key(candidate) -> Tuple[object, ...]:
    """Concrete service history without its expanded road trajectory."""
    return (
        tuple(candidate.route_events),
        tuple(candidate.key_path),
        tuple(sorted(candidate.served_requests)),
    )


def physical_path_key(candidate) -> Tuple[str, ...]:
    return tuple(candidate.full_path)


def physical_path_mismatch_details(
    incremental_candidates: Sequence,
    full_candidates: Sequence,
) -> str:
    """Return compact first-divergence diagnostics for matching event plans."""
    incremental_by_plan = {
        event_plan_key(candidate): candidate for candidate in incremental_candidates
    }
    full_by_plan = {
        event_plan_key(candidate): candidate for candidate in full_candidates
    }
    details = []
    for plan_key in sorted(set(incremental_by_plan) & set(full_by_plan), key=repr):
        incremental_path = tuple(incremental_by_plan[plan_key].full_path)
        full_path = tuple(full_by_plan[plan_key].full_path)
        if incremental_path == full_path:
            continue
        shared_length = min(len(incremental_path), len(full_path))
        divergence = next(
            (
                index
                for index in range(shared_length)
                if incremental_path[index] != full_path[index]
            ),
            shared_length,
        )
        left = max(0, divergence - 2)
        right = divergence + 3
        details.append(
            {
                "events": list(plan_key[0]),
                "incremental_length": len(incremental_path),
                "full_length": len(full_path),
                "first_divergence": divergence,
                "incremental_window": list(incremental_path[left:right]),
                "full_window": list(full_path[left:right]),
            }
        )
    return json.dumps(details, ensure_ascii=True, separators=(",", ":"))


def set_recall(reference: set, observed: set) -> float:
    if not reference:
        return 1.0 if not observed else 0.0
    return len(reference & observed) / len(reference)


def classify_event_subtype(
    manager: DynamicKMORManager,
    event,
    tolerance: float = 1e-9,
) -> str:
    """Classify an event from the common pre-event system state."""
    event_type = type(event).__name__
    if not isinstance(event, EdgeUpdate):
        return event_type
    old_weight = manager.graph.edge_weight_at(
        event.u,
        event.v,
        event.event_time,
    )
    scale = max(1.0, abs(float(old_weight)), abs(float(event.new_weight)))
    threshold = tolerance * scale
    if event.new_weight > old_weight + threshold:
        return "WeightIncrease"
    if event.new_weight < old_weight - threshold:
        return "WeightDecreaseOrRestore"
    return "WeightUnchanged"


def compare_event(
    workload: Mapping[str, object],
    event_index: int,
    event,
    incremental_manager: DynamicKMORManager,
    full_manager: DynamicKMORManager,
    incremental_result,
    full_result,
    experiment_variant: str = "FullIncremental",
    branch_clone_time: float = 0.0,
    execution_order: str = "incremental_first",
    event_subtype: str = "",
    requested_k: int = 5,
    require_incremental_certificate: bool = True,
) -> Dict[str, object]:
    incremental_valid, incremental_error = validate_manager(incremental_manager)
    full_valid, full_error = validate_manager(full_manager)
    status = comparison_status(
        incremental_valid=incremental_valid,
        full_valid=full_valid,
        incremental_timed_out=bool(incremental_result.timed_out),
        full_timed_out=bool(full_result.timed_out),
        incremental_hit_max_states=bool(incremental_result.hit_max_states),
        full_hit_max_states=bool(full_result.hit_max_states),
        incremental_objective_complete=bool(
            incremental_result.objective_complete
        ),
        incremental_concrete_complete=bool(
            incremental_result.concrete_complete
        ),
        full_objective_complete=bool(full_result.objective_complete),
        full_concrete_complete=bool(full_result.concrete_complete),
        require_incremental_certificate=require_incremental_certificate,
    )

    incremental_objectives = {
        objective_key(candidate) for candidate in incremental_manager.routes
    }
    full_objectives = {objective_key(candidate) for candidate in full_manager.routes}
    incremental_plans = {
        candidate_identity(candidate) for candidate in incremental_manager.routes
    }
    full_plans = {candidate_identity(candidate) for candidate in full_manager.routes}
    incremental_services = {
        service_key(candidate) for candidate in incremental_manager.routes
    }
    full_services = {service_key(candidate) for candidate in full_manager.routes}
    incremental_event_plans = {
        event_plan_key(candidate) for candidate in incremental_manager.routes
    }
    full_event_plans = {
        event_plan_key(candidate) for candidate in full_manager.routes
    }
    incremental_physical_paths = {
        physical_path_key(candidate) for candidate in incremental_manager.routes
    }
    full_physical_paths = {
        physical_path_key(candidate) for candidate in full_manager.routes
    }

    same_coverage = incremental_result.best_served == full_result.best_served
    cost_regret: object = ""
    if (
        same_coverage
        and finite(incremental_result.best_cost)
        and finite(full_result.best_cost)
        and float(full_result.best_cost) > 0
    ):
        cost_regret = (
            float(incremental_result.best_cost) - float(full_result.best_cost)
        ) / float(full_result.best_cost)

    is_periodic = (
        incremental_result.strategy == UpdateStrategy.PERIODIC_RECOMPUTE.value
    )
    periodic_interval = (
        incremental_manager.periodic_recompute_interval if is_periodic else 0
    )
    online_method = online_method_name(
        incremental_result.strategy,
        periodic_interval,
    )
    return {
        "scenario": workload.get("scenario", "baseline"),
        "sweep": workload.get("sweep", "baseline"),
        "parameter": workload.get("parameter", "baseline"),
        "parameter_value": workload.get("parameter_value", 1.0),
        "dataset": workload["dataset"],
        "seed": workload["seed"],
        "k": requested_k,
        "experiment_variant": experiment_variant,
        "diagnostic_non_exact": not require_incremental_certificate,
        "event_index": event_index,
        "event_type": type(event).__name__,
        "event_subtype": event_subtype or type(event).__name__,
        "event_time": event.event_time,
        "status": status,
        "online_strategy": incremental_result.strategy,
        "online_method": online_method,
        "periodic_recompute_interval": periodic_interval,
        "online_update_mode": incremental_result.update_mode,
        "incremental_certificate_stage": incremental_result.certificate_stage,
        "full_certificate_stage": full_result.certificate_stage,
        "execution_order": execution_order,
        "branch_clone_time": branch_clone_time,
        "periodic_recompute": (
            incremental_result.update_mode == "PeriodicFullRecompute"
        ),
        "incremental_time": incremental_result.update_time,
        "full_recompute_time": full_result.update_time,
        "speedup": (
            full_result.update_time / incremental_result.update_time
            if incremental_result.update_time > 0
            else math.inf
        ),
        "incremental_served": incremental_result.best_served,
        "full_served": full_result.best_served,
        "served_gap": incremental_result.best_served - full_result.best_served,
        "coverage_match": same_coverage,
        "incremental_cost": incremental_result.best_cost,
        "full_cost": full_result.best_cost,
        "cost_regret_same_coverage": cost_regret,
        "incremental_route_count": len(incremental_manager.routes),
        "full_route_count": len(full_manager.routes),
        "incremental_backup_route_count": len(incremental_manager.backup_routes),
        "full_backup_route_count": len(full_manager.backup_routes),
        "incremental_available_route_count": len(
            incremental_manager.available_routes
        ),
        "full_available_route_count": len(full_manager.available_routes),
        "incremental_available_topk_complete": (
            incremental_result.available_topk_complete
        ),
        "full_available_topk_complete": full_result.available_topk_complete,
        "availability_preserved": (
            not full_result.available_topk_complete
            or incremental_result.available_topk_complete
        ),
        "incremental_mean_backup_cost_regret": (
            incremental_result.mean_backup_cost_regret
        ),
        "full_mean_backup_cost_regret": full_result.mean_backup_cost_regret,
        "incremental_max_backup_cost_regret": (
            incremental_result.max_backup_cost_regret
        ),
        "full_max_backup_cost_regret": full_result.max_backup_cost_regret,
        "incremental_archive_size": len(incremental_manager.route_archive),
        "full_archive_size": len(full_manager.route_archive),
        "incremental_retained_routes_before_event": (
            incremental_result.retained_routes_before_event
        ),
        "full_retained_routes_before_event": (
            full_result.retained_routes_before_event
        ),
        "incremental_affected_routes": incremental_result.affected_routes,
        "full_affected_routes": full_result.affected_routes,
        "incremental_affected_route_ratio": (
            incremental_result.affected_route_ratio
        ),
        "full_affected_route_ratio": full_result.affected_route_ratio,
        "objective_frontier_match": incremental_objectives == full_objectives,
        "objective_recall": set_recall(full_objectives, incremental_objectives),
        "service_set_match": incremental_services == full_services,
        "service_set_recall": set_recall(full_services, incremental_services),
        "event_plan_match": incremental_event_plans == full_event_plans,
        "event_plan_recall": set_recall(
            full_event_plans,
            incremental_event_plans,
        ),
        "physical_path_match": (
            incremental_physical_paths == full_physical_paths
        ),
        "physical_path_recall": set_recall(
            full_physical_paths,
            incremental_physical_paths,
        ),
        "physical_path_mismatch_details": physical_path_mismatch_details(
            incremental_manager.routes,
            full_manager.routes,
        ),
        "concrete_topk_match": incremental_plans == full_plans,
        "concrete_plan_recall": set_recall(full_plans, incremental_plans),
        "incremental_expanded_states": incremental_result.expanded_states,
        "full_expanded_states": full_result.expanded_states,
        "incremental_peak_open_size": incremental_result.peak_open_size,
        "full_peak_open_size": full_result.peak_open_size,
        "incremental_peak_active_dp_labels": (
            incremental_result.peak_active_dp_labels
        ),
        "full_peak_active_dp_labels": full_result.peak_active_dp_labels,
        "incremental_dag_nodes_created": incremental_result.dag_nodes_created,
        "full_dag_nodes_created": full_result.dag_nodes_created,
        "incremental_dp_pruned": incremental_result.dp_pruned_states,
        "full_dp_pruned": full_result.dp_pruned_states,
        "incremental_tdsp_queries": incremental_result.tdsp_queries,
        "full_tdsp_queries": full_result.tdsp_queries,
        "incremental_tdsp_cache_hits": incremental_result.tdsp_cache_hits,
        "full_tdsp_cache_hits": full_result.tdsp_cache_hits,
        "incremental_tdsp_lower_path_reuses": (
            incremental_result.tdsp_lower_path_reuses
        ),
        "full_tdsp_lower_path_reuses": full_result.tdsp_lower_path_reuses,
        "incremental_tdsp_dijkstra_runs": incremental_result.tdsp_dijkstra_runs,
        "full_tdsp_dijkstra_runs": full_result.tdsp_dijkstra_runs,
        "incremental_tdsp_cache_invalidations": (
            incremental_result.tdsp_cache_invalidations
        ),
        "full_tdsp_cache_invalidations": full_result.tdsp_cache_invalidations,
        "incremental_lower_bound_queries": incremental_result.lower_bound_queries,
        "full_lower_bound_queries": full_result.lower_bound_queries,
        "incremental_lower_bound_cache_hits": (
            incremental_result.lower_bound_cache_hits
        ),
        "full_lower_bound_cache_hits": full_result.lower_bound_cache_hits,
        "incremental_lower_bound_dijkstra_runs": (
            incremental_result.lower_bound_dijkstra_runs
        ),
        "full_lower_bound_dijkstra_runs": full_result.lower_bound_dijkstra_runs,
        "incremental_deviation_queries": incremental_result.deviation_queries,
        "full_deviation_queries": full_result.deviation_queries,
        "incremental_deviation_cache_hits": (
            incremental_result.deviation_cache_hits
        ),
        "full_deviation_cache_hits": full_result.deviation_cache_hits,
        "incremental_deviation_dijkstra_runs": (
            incremental_result.deviation_dijkstra_runs
        ),
        "full_deviation_dijkstra_runs": full_result.deviation_dijkstra_runs,
        "incremental_insertion_transitions_tested": (
            incremental_result.insertion_transitions_tested
        ),
        "full_insertion_transitions_tested": (
            full_result.insertion_transitions_tested
        ),
        "incremental_insertion_labels_generated": (
            incremental_result.insertion_labels_generated
        ),
        "full_insertion_labels_generated": full_result.insertion_labels_generated,
        "incremental_insertion_dominance_checks": (
            incremental_result.insertion_dominance_checks
        ),
        "full_insertion_dominance_checks": full_result.insertion_dominance_checks,
        "incremental_insertion_resource_checks": (
            incremental_result.insertion_resource_checks
        ),
        "full_insertion_resource_checks": full_result.insertion_resource_checks,
        "incremental_dependency_lookups": incremental_result.dependency_lookups,
        "full_dependency_lookups": full_result.dependency_lookups,
        "incremental_dependency_candidates": (
            incremental_result.dependency_candidates
        ),
        "full_dependency_candidates": full_result.dependency_candidates,
        "incremental_plan_dag_builds": incremental_result.plan_dag_builds,
        "full_plan_dag_builds": full_result.plan_dag_builds,
        "incremental_plan_dag_reuses": incremental_result.plan_dag_reuses,
        "full_plan_dag_reuses": full_result.plan_dag_reuses,
        "incremental_plan_prefix_labels_evaluated": (
            incremental_result.plan_prefix_labels_evaluated
        ),
        "full_plan_prefix_labels_evaluated": (
            full_result.plan_prefix_labels_evaluated
        ),
        "incremental_plan_prefix_transitions_saved": (
            incremental_result.plan_prefix_transitions_saved
        ),
        "full_plan_prefix_transitions_saved": (
            full_result.plan_prefix_transitions_saved
        ),
        "incremental_plan_prefix_cache_hits": (
            incremental_result.plan_prefix_cache_hits
        ),
        "full_plan_prefix_cache_hits": full_result.plan_prefix_cache_hits,
        "incremental_plan_prefix_labels_reopened": (
            incremental_result.plan_prefix_labels_reopened
        ),
        "full_plan_prefix_labels_reopened": (
            full_result.plan_prefix_labels_reopened
        ),
        "incremental_plan_prefix_invalidations": (
            incremental_result.plan_prefix_invalidations
        ),
        "full_plan_prefix_invalidations": full_result.plan_prefix_invalidations,
        "incremental_local_reoptimization_runs": (
            incremental_result.local_reoptimization_runs
        ),
        "full_local_reoptimization_runs": full_result.local_reoptimization_runs,
        "incremental_local_reoptimization_requests": (
            incremental_result.local_reoptimization_requests
        ),
        "full_local_reoptimization_requests": (
            full_result.local_reoptimization_requests
        ),
        "incremental_local_reoptimization_expanded": (
            incremental_result.local_reoptimization_expanded
        ),
        "full_local_reoptimization_expanded": (
            full_result.local_reoptimization_expanded
        ),
        "incremental_generated_schedules": (
            incremental_result.generated_schedules
        ),
        "incremental_feasible_schedules": (
            incremental_result.feasible_schedules
        ),
        "incremental_evicted_schedules": incremental_result.evicted_schedules,
        "incremental_schedule_generation_capped": (
            incremental_result.schedule_generation_capped
        ),
        "incremental_local_search_iterations": (
            incremental_result.local_search_iterations
        ),
        "incremental_local_search_moves": incremental_result.local_search_moves,
        "incremental_candidate_pool_peak_size": (
            incremental_result.candidate_pool_peak_size
        ),
        "incremental_local_search_iteration_capped": (
            incremental_result.local_search_iteration_capped
        ),
        "incremental_fallback": incremental_result.fallback_recompute,
        "incremental_adaptive_recompute": incremental_result.adaptive_recompute,
        "incremental_hit_max_states": incremental_result.hit_max_states,
        "full_hit_max_states": full_result.hit_max_states,
        "incremental_timed_out": incremental_result.timed_out,
        "full_timed_out": full_result.timed_out,
        "incremental_search_exhausted": incremental_result.search_exhausted,
        "full_search_exhausted": full_result.search_exhausted,
        "incremental_scope_complete": incremental_result.scope_complete,
        "full_scope_complete": full_result.scope_complete,
        "incremental_unresolved_count": incremental_result.unresolved_count,
        "full_unresolved_count": full_result.unresolved_count,
        "incremental_objective_complete": incremental_result.objective_complete,
        "full_objective_complete": full_result.objective_complete,
        "incremental_concrete_complete": incremental_result.concrete_complete,
        "full_concrete_complete": full_result.concrete_complete,
        "incremental_completed": incremental_result.completed,
        "full_completed": full_result.completed,
        "incremental_constraint_valid": incremental_valid,
        "full_constraint_valid": full_valid,
        "incremental_constraint_error": incremental_error,
        "full_constraint_error": full_error,
    }


def paired_worker(task: Mapping[str, object], result_queue) -> None:
    workload = dict(task["workload"])  # type: ignore[arg-type]
    started = time.perf_counter()
    try:
        workdir = Path(str(task["workdir"]))
        load_started = time.perf_counter()
        static_graph = read_edge_list(
            str(workdir / str(workload["edge_file"])),
            directed=bool(workload.get("directed", False)),
            ignore_first_col=bool(workload.get("ignore_first_col", True)),
        )
        dynamic_graph = TimeDependentGraph.from_static_graph(
            static_graph,
            bucket_size=float(task["bucket_size"]),
        )
        requests = parse_requests_file(
            str(workdir / str(workload["initial_requests_file"]))
        )
        events = parse_event_file(workdir / str(workload["events_file"]))
        max_events = max(0, int(task.get("max_events", 0)))
        if max_events:
            events = events[:max_events]
        preprocess_time = time.perf_counter() - load_started

        manager_classes = {
            UpdateStrategy.KINETIC_TREE.value: KineticTreeKMORManager,
            UpdateStrategy.TD_INSERTION.value: TDInsertionKMORManager,
            UpdateStrategy.TD_TOPK_INSERTION.value: TDTopKInsertionKMORManager,
            UpdateStrategy.RHLS_TOPK.value: RHLSTopKManager,
        }
        manager_class = manager_classes.get(
            str(task["online_strategy"]),
            DynamicKMORManager,
        )
        manager = manager_class(
            dynamic_graph,
            str(workload["source"]),
            str(workload["target"]),
            requests,
            strategy=str(task["online_strategy"]),
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
                task["full_recompute_reuse_tdsp_cache"]
            ),
            periodic_recompute_interval=int(task["periodic_recompute_interval"]),
            strict_static_semantics=True,
            enable_certified_recovery=bool(task["enable_certified_recovery"]),
            kinetic_tree_max_schedules=int(task["kinetic_tree_max_schedules"]),
            kinetic_tree_max_generated=int(task["kinetic_tree_max_generated"]),
            rhls_pool_limit=int(task["rhls_pool_limit"]),
            rhls_max_rounds=int(task["rhls_max_rounds"]),
            rhls_max_generated=int(task["rhls_max_generated"]),
        )
        initialization = manager.initialize(0.0)
        initial_valid, initial_error = validate_manager(manager)
        rows: List[Dict[str, object]] = []
        initialization_complete = (
            initial_valid
            and not initialization.hit_max_states
            and not initialization.timed_out
            and (
                not bool(task["require_incremental_certificate"])
                or (
                    initialization.objective_complete
                    and initialization.concrete_complete
                )
            )
        )
        if initialization_complete:
            for event_index, event in enumerate(events, start=1):
                event_subtype = classify_event_subtype(manager, event)
                paired_order = str(task.get("paired_order", "alternating"))
                if paired_order == "alternating":
                    # Counterbalance warm-cache effects across both methods.
                    # Including the seed reverses parity across replications.
                    full_first = (
                        event_index + int(workload["seed"])
                    ) % 2 == 0
                    execution_order = (
                        "full_first" if full_first else "incremental_first"
                    )
                else:
                    execution_order = paired_order
                outcome = run_paired_event(
                    manager,
                    event,
                    commit_incremental=True,
                    online_strategy=str(task["online_strategy"]),
                    execution_order=execution_order,
                )
                rows.append(
                    compare_event(
                        workload,
                        event_index,
                        event,
                        outcome.incremental_manager,
                        outcome.full_manager,
                        outcome.incremental_result,
                        outcome.full_result,
                        str(task["experiment_variant"]),
                        outcome.branch_clone_time,
                        outcome.execution_order,
                        event_subtype,
                        int(task["k"]),
                        bool(task["require_incremental_certificate"]),
                    )
                )
                # The online system commits the incremental decision. The full
                # solution remains an oracle evaluated from that same pre-event state.
                manager = outcome.incremental_manager

        result_queue.put(
            {
                "status": (
                    "OK"
                    if initialization_complete
                    else "INVALID" if not initial_valid
                    else "TIMEOUT" if initialization.timed_out
                    else "STATE_LIMIT" if initialization.hit_max_states
                    else "INCOMPLETE"
                ),
                "scenario": workload.get("scenario", "baseline"),
                "sweep": workload.get("sweep", "baseline"),
                "parameter": workload.get("parameter", "baseline"),
                "parameter_value": workload.get("parameter_value", 1.0),
                "dataset": workload["dataset"],
                "seed": workload["seed"],
                "k": task["k"],
                "experiment_variant": task["experiment_variant"],
                "online_strategy": task["online_strategy"],
                "periodic_recompute_interval": task["periodic_recompute_interval"],
                "preprocess_time": preprocess_time,
                "initialization_time": initialization.update_time,
                "initial_generated_schedules": initialization.generated_schedules,
                "initial_feasible_schedules": initialization.feasible_schedules,
                "initial_evicted_schedules": initialization.evicted_schedules,
                "initial_schedule_generation_capped": (
                    initialization.schedule_generation_capped
                ),
                "initial_error": initial_error,
                "wall_time": time.perf_counter() - started,
                "rows": rows,
            }
        )
    except MemoryError:
        result_queue.put(
            {
                "status": "OOM",
                "dataset": workload.get("dataset", ""),
                "seed": workload.get("seed", ""),
                "k": task.get("k", ""),
                "experiment_variant": task.get("experiment_variant", ""),
                "online_strategy": task.get("online_strategy", ""),
                "periodic_recompute_interval": task.get(
                    "periodic_recompute_interval", ""
                ),
                "error": "MemoryError",
                "rows": [],
            }
        )
    except BaseException as exc:
        result_queue.put(
            {
                "status": "ERROR",
                "dataset": workload.get("dataset", ""),
                "seed": workload.get("seed", ""),
                "k": task.get("k", ""),
                "experiment_variant": task.get("experiment_variant", ""),
                "online_strategy": task.get("online_strategy", ""),
                "periodic_recompute_interval": task.get(
                    "periodic_recompute_interval", ""
                ),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "rows": [],
            }
        )


def run_isolated(task: Mapping[str, object], timeout: float) -> Dict[str, object]:
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=paired_worker, args=(task, result_queue))
    process.start()
    workload = dict(task["workload"])  # type: ignore[arg-type]
    deadline = time.monotonic() + timeout
    result: Optional[Dict[str, object]] = None
    while result is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            process.terminate()
            process.join(10)
            result_queue.close()
            return {
                "status": "TIMEOUT",
                "dataset": workload["dataset"],
                "seed": workload["seed"],
                "k": task.get("k", ""),
                "experiment_variant": task.get("experiment_variant", ""),
                "online_strategy": task.get("online_strategy", ""),
                "periodic_recompute_interval": task.get(
                    "periodic_recompute_interval", ""
                ),
                "error": f"timeout after {timeout:g}s",
                "rows": [],
            }
        try:
            result = result_queue.get(timeout=min(1.0, remaining))
        except queue_module.Empty:
            if not process.is_alive():
                result = {
                    "status": "ERROR",
                    "dataset": workload["dataset"],
                    "seed": workload["seed"],
                    "k": task.get("k", ""),
                    "error": f"worker exited with code {process.exitcode}",
                    "rows": [],
                }
    process.join(10)
    if process.is_alive():
        process.terminate()
        process.join(10)
    result_queue.close()
    return result


def mean_numeric(rows: Sequence[Mapping[str, object]], field: str) -> object:
    values = [float(row[field]) for row in rows if finite(row.get(field))]
    return statistics.fmean(values) if values else ""


def sum_numeric(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return sum(float(row[field]) for row in rows if finite(row.get(field)))


def categorical_rate(
    rows: Sequence[Mapping[str, object]],
    field: str,
    value: object,
) -> object:
    if not rows:
        return ""
    return sum(row.get(field) == value for row in rows) / len(rows)


def percentile_numeric(
    rows: Sequence[Mapping[str, object]],
    field: str,
    percentile: float,
) -> object:
    values = sorted(float(row[field]) for row in rows if finite(row.get(field)))
    if not values:
        return ""
    position = (len(values) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def summarize(payloads: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    for payload in payloads:
        rows = list(payload.get("rows", []))  # type: ignore[arg-type]
        complete = [row for row in rows if row.get("status") == "COMPLETE"]
        regrets = [
            float(row["cost_regret_same_coverage"])
            for row in complete
            if finite(row.get("cost_regret_same_coverage"))
        ]
        full_topk_events = [
            row for row in complete if row.get("full_available_topk_complete")
        ]
        edge_updates = [
            row for row in complete if row.get("event_type") == "EdgeUpdate"
        ]
        incremental_tdsp_queries = sum_numeric(complete, "incremental_tdsp_queries")
        incremental_tdsp_hits = sum_numeric(complete, "incremental_tdsp_cache_hits")
        full_tdsp_queries = sum_numeric(complete, "full_tdsp_queries")
        full_tdsp_hits = sum_numeric(complete, "full_tdsp_cache_hits")
        incremental_expanded = sum_numeric(complete, "incremental_expanded_states")
        local_expanded = sum_numeric(
            complete, "incremental_local_reoptimization_expanded"
        )
        full_expanded = sum_numeric(complete, "full_expanded_states")
        summaries.append(
            {
                "scenario": payload.get("scenario", "baseline"),
                "sweep": payload.get("sweep", "baseline"),
                "parameter": payload.get("parameter", "baseline"),
                "parameter_value": payload.get("parameter_value", 1.0),
                "dataset": payload.get("dataset", ""),
                "seed": payload.get("seed", ""),
                "k": payload.get("k", ""),
                "experiment_variant": payload.get(
                    "experiment_variant", "FullIncremental"
                ),
                "online_strategy": payload.get("online_strategy", "incremental"),
                "online_method": online_method_name(
                    payload.get("online_strategy", "incremental"),
                    payload.get("periodic_recompute_interval", ""),
                ),
                "periodic_recompute_interval": payload.get(
                    "periodic_recompute_interval", ""
                ),
                "status": payload.get("status", "ERROR"),
                "event_count": len(rows),
                "complete_event_count": len(complete),
                "state_limit_count": sum(
                    row.get("status") == "STATE_LIMIT" for row in rows
                ),
                "timeout_count": sum(
                    row.get("status") == "TIMEOUT" for row in rows
                ),
                "incomplete_count": sum(
                    row.get("status") == "INCOMPLETE" for row in rows
                ),
                "invalid_count": sum(row.get("status") == "INVALID" for row in rows),
                "schedule_generation_capped_count": sum(
                    row.get("incremental_schedule_generation_capped") is True
                    for row in rows
                ),
                "local_search_iteration_capped_count": sum(
                    row.get("incremental_local_search_iteration_capped") is True
                    for row in rows
                ),
                "local_search_iterations": sum_numeric(
                    complete, "incremental_local_search_iterations"
                ),
                "local_search_moves": sum_numeric(
                    complete, "incremental_local_search_moves"
                ),
                "mean_candidate_pool_peak_size": mean_numeric(
                    complete, "incremental_candidate_pool_peak_size"
                ),
                "generated_schedules": sum_numeric(
                    complete, "incremental_generated_schedules"
                ),
                "feasible_schedules": sum_numeric(
                    complete, "incremental_feasible_schedules"
                ),
                "evicted_schedules": sum_numeric(
                    complete, "incremental_evicted_schedules"
                ),
                "incremental_first_count": sum(
                    row.get("execution_order") == "incremental_first"
                    for row in complete
                ),
                "full_first_count": sum(
                    row.get("execution_order") == "full_first"
                    for row in complete
                ),
                "mean_incremental_time": mean_numeric(complete, "incremental_time"),
                "p95_incremental_time": percentile_numeric(
                    complete, "incremental_time", 0.95
                ),
                "mean_full_recompute_time": mean_numeric(
                    complete, "full_recompute_time"
                ),
                "p95_full_recompute_time": percentile_numeric(
                    complete, "full_recompute_time", 0.95
                ),
                "paired_total_speedup": (
                    sum(float(row["full_recompute_time"]) for row in complete)
                    / sum(float(row["incremental_time"]) for row in complete)
                    if complete
                    and sum(float(row["incremental_time"]) for row in complete) > 0
                    else ""
                ),
                "coverage_match_rate": mean_numeric(complete, "coverage_match"),
                "objective_frontier_match_rate": mean_numeric(
                    complete, "objective_frontier_match"
                ),
                "service_set_match_rate": mean_numeric(complete, "service_set_match"),
                "event_plan_match_rate": mean_numeric(complete, "event_plan_match"),
                "physical_path_match_rate": mean_numeric(
                    complete, "physical_path_match"
                ),
                "concrete_topk_match_rate": mean_numeric(
                    complete, "concrete_topk_match"
                ),
                "mean_objective_recall": mean_numeric(complete, "objective_recall"),
                "mean_service_set_recall": mean_numeric(complete, "service_set_recall"),
                "mean_event_plan_recall": mean_numeric(
                    complete, "event_plan_recall"
                ),
                "mean_physical_path_recall": mean_numeric(
                    complete, "physical_path_recall"
                ),
                "mean_concrete_plan_recall": mean_numeric(
                    complete, "concrete_plan_recall"
                ),
                "mean_served_gap": mean_numeric(complete, "served_gap"),
                "mean_cost_regret_same_coverage": (
                    statistics.fmean(regrets) if regrets else ""
                ),
                "fallback_rate": mean_numeric(complete, "incremental_fallback"),
                "direct_certificate_acceptance_rate": categorical_rate(
                    complete, "incremental_certificate_stage", "incremental"
                ),
                "inherited_certificate_rate": categorical_rate(
                    complete, "incremental_certificate_stage", "inherited"
                ),
                "local_certificate_acceptance_rate": categorical_rate(
                    complete, "incremental_certificate_stage", "local"
                ),
                "full_recovery_rate": categorical_rate(
                    complete, "incremental_certificate_stage", "full"
                ),
                "local_reoptimization_rate": (
                    sum(
                        float(row.get("incremental_local_reoptimization_runs", 0))
                        > 0
                        for row in complete
                    )
                    / len(complete)
                    if complete
                    else ""
                ),
                "objective_certificate_rate": mean_numeric(
                    complete, "incremental_objective_complete"
                ),
                "concrete_certificate_rate": mean_numeric(
                    complete, "incremental_concrete_complete"
                ),
                "scope_complete_rate": mean_numeric(
                    complete, "incremental_scope_complete"
                ),
                "mean_unresolved_count": mean_numeric(
                    complete, "incremental_unresolved_count"
                ),
                "adaptive_recompute_rate": mean_numeric(
                    complete, "incremental_adaptive_recompute"
                ),
                "periodic_recompute_rate": mean_numeric(
                    complete, "periodic_recompute"
                ),
                "mean_incremental_archive_size": mean_numeric(
                    complete, "incremental_archive_size"
                ),
                "mean_available_route_count": mean_numeric(
                    complete, "incremental_available_route_count"
                ),
                "available_topk_complete_rate": mean_numeric(
                    complete, "incremental_available_topk_complete"
                ),
                "full_available_topk_complete_rate": mean_numeric(
                    complete, "full_available_topk_complete"
                ),
                "availability_preservation_rate": (
                    mean_numeric(full_topk_events, "incremental_available_topk_complete")
                    if full_topk_events
                    else 1.0
                ),
                "mean_backup_route_count": mean_numeric(
                    complete, "incremental_backup_route_count"
                ),
                "mean_backup_cost_regret": mean_numeric(
                    complete, "incremental_mean_backup_cost_regret"
                ),
                "mean_max_backup_cost_regret": mean_numeric(
                    complete, "incremental_max_backup_cost_regret"
                ),
                "mean_edge_affected_route_ratio": mean_numeric(
                    edge_updates, "incremental_affected_route_ratio"
                ),
                "incremental_tdsp_queries": incremental_tdsp_queries,
                "incremental_tdsp_cache_hits": incremental_tdsp_hits,
                "incremental_tdsp_cache_hit_rate": (
                    incremental_tdsp_hits / incremental_tdsp_queries
                    if incremental_tdsp_queries
                    else 0.0
                ),
                "incremental_tdsp_lower_path_reuses": sum_numeric(
                    complete, "incremental_tdsp_lower_path_reuses"
                ),
                "incremental_tdsp_dijkstra_runs": sum_numeric(
                    complete, "incremental_tdsp_dijkstra_runs"
                ),
                "full_tdsp_queries": full_tdsp_queries,
                "full_tdsp_cache_hits": full_tdsp_hits,
                "full_tdsp_cache_hit_rate": (
                    full_tdsp_hits / full_tdsp_queries
                    if full_tdsp_queries
                    else 0.0
                ),
                "full_tdsp_dijkstra_runs": sum_numeric(
                    complete, "full_tdsp_dijkstra_runs"
                ),
                "incremental_insertion_transitions_tested": sum_numeric(
                    complete, "incremental_insertion_transitions_tested"
                ),
                "incremental_insertion_labels_generated": sum_numeric(
                    complete, "incremental_insertion_labels_generated"
                ),
                "incremental_insertion_dominance_checks": sum_numeric(
                    complete, "incremental_insertion_dominance_checks"
                ),
                "incremental_dependency_lookups": sum_numeric(
                    complete, "incremental_dependency_lookups"
                ),
                "incremental_dependency_candidates": sum_numeric(
                    complete, "incremental_dependency_candidates"
                ),
                "incremental_plan_dag_builds": sum_numeric(
                    complete, "incremental_plan_dag_builds"
                ),
                "incremental_plan_dag_reuses": sum_numeric(
                    complete, "incremental_plan_dag_reuses"
                ),
                "incremental_plan_prefix_labels_evaluated": sum_numeric(
                    complete, "incremental_plan_prefix_labels_evaluated"
                ),
                "incremental_plan_prefix_transitions_saved": sum_numeric(
                    complete, "incremental_plan_prefix_transitions_saved"
                ),
                "incremental_plan_prefix_cache_hits": sum_numeric(
                    complete, "incremental_plan_prefix_cache_hits"
                ),
                "incremental_plan_prefix_labels_reopened": sum_numeric(
                    complete, "incremental_plan_prefix_labels_reopened"
                ),
                "incremental_plan_prefix_invalidations": sum_numeric(
                    complete, "incremental_plan_prefix_invalidations"
                ),
                "incremental_local_reoptimization_runs": sum_numeric(
                    complete, "incremental_local_reoptimization_runs"
                ),
                "incremental_local_reoptimization_requests": sum_numeric(
                    complete, "incremental_local_reoptimization_requests"
                ),
                "incremental_local_reoptimization_expanded": sum_numeric(
                    complete, "incremental_local_reoptimization_expanded"
                ),
                "mean_incremental_peak_open_size": mean_numeric(
                    complete, "incremental_peak_open_size"
                ),
                "mean_full_peak_open_size": mean_numeric(
                    complete, "full_peak_open_size"
                ),
                "mean_incremental_peak_active_dp_labels": mean_numeric(
                    complete, "incremental_peak_active_dp_labels"
                ),
                "mean_full_peak_active_dp_labels": mean_numeric(
                    complete, "full_peak_active_dp_labels"
                ),
                "incremental_dag_nodes_created": sum_numeric(
                    complete, "incremental_dag_nodes_created"
                ),
                "full_dag_nodes_created": sum_numeric(
                    complete, "full_dag_nodes_created"
                ),
                "full_expanded_states": full_expanded,
                "local_full_expanded_ratio": (
                    local_expanded / full_expanded if full_expanded > 0 else ""
                ),
                "total_incremental_full_expanded_ratio": (
                    incremental_expanded / full_expanded
                    if full_expanded > 0
                    else ""
                ),
                "preprocess_time": payload.get("preprocess_time", ""),
                "initialization_time": payload.get("initialization_time", ""),
                "initial_generated_schedules": payload.get(
                    "initial_generated_schedules", 0
                ),
                "initial_feasible_schedules": payload.get(
                    "initial_feasible_schedules", 0
                ),
                "initial_evicted_schedules": payload.get(
                    "initial_evicted_schedules", 0
                ),
                "initial_schedule_generation_capped": payload.get(
                    "initial_schedule_generation_capped", False
                ),
                "mean_branch_clone_time": mean_numeric(
                    complete, "branch_clone_time"
                ),
                "total_branch_clone_time": sum_numeric(
                    complete, "branch_clone_time"
                ),
                "wall_time": payload.get("wall_time", ""),
                "error": payload.get("error", payload.get("initial_error", "")),
            }
        )
    return summaries


def aggregate_events(
    event_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    """Aggregate paired metrics across seeds without mixing online policies."""
    grouped: Dict[Tuple[object, ...], List[Mapping[str, object]]] = {}
    for row in event_rows:
        key = (
            row.get("scenario", "baseline"),
            row.get("sweep", "baseline"),
            row.get("parameter", "baseline"),
            row.get("parameter_value", 1.0),
            row.get("dataset", ""),
            row.get("k", 5),
            row.get("experiment_variant", "FullIncremental"),
            row.get("online_strategy", "incremental"),
            row.get("periodic_recompute_interval", ""),
        )
        grouped.setdefault(key, []).append(row)

    aggregates: List[Dict[str, object]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        complete = [row for row in rows if row.get("status") == "COMPLETE"]
        regrets = [
            float(row["cost_regret_same_coverage"])
            for row in complete
            if finite(row.get("cost_regret_same_coverage"))
        ]
        full_topk_events = [
            row for row in complete if row.get("full_available_topk_complete")
        ]
        edge_updates = [
            row for row in complete if row.get("event_type") == "EdgeUpdate"
        ]
        online_total = sum(float(row["incremental_time"]) for row in complete)
        oracle_total = sum(float(row["full_recompute_time"]) for row in complete)
        incremental_tdsp_queries = sum_numeric(complete, "incremental_tdsp_queries")
        incremental_tdsp_hits = sum_numeric(complete, "incremental_tdsp_cache_hits")
        full_tdsp_queries = sum_numeric(complete, "full_tdsp_queries")
        full_tdsp_hits = sum_numeric(complete, "full_tdsp_cache_hits")
        incremental_expanded = sum_numeric(complete, "incremental_expanded_states")
        local_expanded = sum_numeric(
            complete, "incremental_local_reoptimization_expanded"
        )
        full_expanded = sum_numeric(complete, "full_expanded_states")
        aggregates.append(
            {
                "scenario": key[0],
                "sweep": key[1],
                "parameter": key[2],
                "parameter_value": key[3],
                "dataset": key[4],
                "k": key[5],
                "experiment_variant": key[6],
                "online_strategy": key[7],
                "online_method": online_method_name(key[7], key[8]),
                "periodic_recompute_interval": key[8],
                "seed_count": len({row.get("seed") for row in rows}),
                "event_count": len(rows),
                "complete_event_count": len(complete),
                "state_limit_count": sum(
                    row.get("status") == "STATE_LIMIT" for row in rows
                ),
                "timeout_count": sum(
                    row.get("status") == "TIMEOUT" for row in rows
                ),
                "incomplete_count": sum(
                    row.get("status") == "INCOMPLETE" for row in rows
                ),
                "invalid_count": sum(row.get("status") == "INVALID" for row in rows),
                "schedule_generation_capped_count": sum(
                    row.get("incremental_schedule_generation_capped") is True
                    for row in rows
                ),
                "local_search_iteration_capped_count": sum(
                    row.get("incremental_local_search_iteration_capped") is True
                    for row in rows
                ),
                "local_search_iterations": sum_numeric(
                    complete, "incremental_local_search_iterations"
                ),
                "local_search_moves": sum_numeric(
                    complete, "incremental_local_search_moves"
                ),
                "mean_candidate_pool_peak_size": mean_numeric(
                    complete, "incremental_candidate_pool_peak_size"
                ),
                "generated_schedules": sum_numeric(
                    complete, "incremental_generated_schedules"
                ),
                "feasible_schedules": sum_numeric(
                    complete, "incremental_feasible_schedules"
                ),
                "evicted_schedules": sum_numeric(
                    complete, "incremental_evicted_schedules"
                ),
                "mean_incremental_time": mean_numeric(
                    complete, "incremental_time"
                ),
                "p95_incremental_time": percentile_numeric(
                    complete, "incremental_time", 0.95
                ),
                "mean_full_recompute_time": mean_numeric(
                    complete, "full_recompute_time"
                ),
                "p95_full_recompute_time": percentile_numeric(
                    complete, "full_recompute_time", 0.95
                ),
                "paired_total_speedup": (
                    oracle_total / online_total if online_total > 0 else ""
                ),
                "coverage_match_rate": mean_numeric(complete, "coverage_match"),
                "objective_frontier_match_rate": mean_numeric(
                    complete, "objective_frontier_match"
                ),
                "service_set_match_rate": mean_numeric(
                    complete, "service_set_match"
                ),
                "concrete_topk_match_rate": mean_numeric(
                    complete, "concrete_topk_match"
                ),
                "mean_objective_recall": mean_numeric(
                    complete, "objective_recall"
                ),
                "mean_service_set_recall": mean_numeric(
                    complete, "service_set_recall"
                ),
                "event_plan_match_rate": mean_numeric(
                    complete, "event_plan_match"
                ),
                "physical_path_match_rate": mean_numeric(
                    complete, "physical_path_match"
                ),
                "mean_event_plan_recall": mean_numeric(
                    complete, "event_plan_recall"
                ),
                "mean_physical_path_recall": mean_numeric(
                    complete, "physical_path_recall"
                ),
                "mean_concrete_plan_recall": mean_numeric(
                    complete, "concrete_plan_recall"
                ),
                "mean_cost_regret_same_coverage": (
                    statistics.fmean(regrets) if regrets else ""
                ),
                "mean_route_count": mean_numeric(
                    complete, "incremental_route_count"
                ),
                "mean_available_route_count": mean_numeric(
                    complete, "incremental_available_route_count"
                ),
                "available_topk_complete_rate": mean_numeric(
                    complete, "incremental_available_topk_complete"
                ),
                "full_available_topk_complete_rate": mean_numeric(
                    complete, "full_available_topk_complete"
                ),
                "availability_preservation_rate": (
                    mean_numeric(full_topk_events, "incremental_available_topk_complete")
                    if full_topk_events
                    else 1.0
                ),
                "mean_backup_route_count": mean_numeric(
                    complete, "incremental_backup_route_count"
                ),
                "mean_backup_cost_regret": mean_numeric(
                    complete, "incremental_mean_backup_cost_regret"
                ),
                "mean_max_backup_cost_regret": mean_numeric(
                    complete, "incremental_max_backup_cost_regret"
                ),
                "mean_edge_affected_route_ratio": mean_numeric(
                    edge_updates, "incremental_affected_route_ratio"
                ),
                "fallback_rate": mean_numeric(complete, "incremental_fallback"),
                "direct_certificate_acceptance_rate": categorical_rate(
                    complete, "incremental_certificate_stage", "incremental"
                ),
                "inherited_certificate_rate": categorical_rate(
                    complete, "incremental_certificate_stage", "inherited"
                ),
                "local_certificate_acceptance_rate": categorical_rate(
                    complete, "incremental_certificate_stage", "local"
                ),
                "full_recovery_rate": categorical_rate(
                    complete, "incremental_certificate_stage", "full"
                ),
                "local_reoptimization_rate": (
                    sum(
                        float(row.get("incremental_local_reoptimization_runs", 0))
                        > 0
                        for row in complete
                    )
                    / len(complete)
                    if complete
                    else ""
                ),
                "objective_certificate_rate": mean_numeric(
                    complete, "incremental_objective_complete"
                ),
                "concrete_certificate_rate": mean_numeric(
                    complete, "incremental_concrete_complete"
                ),
                "scope_complete_rate": mean_numeric(
                    complete, "incremental_scope_complete"
                ),
                "mean_unresolved_count": mean_numeric(
                    complete, "incremental_unresolved_count"
                ),
                "adaptive_recompute_rate": mean_numeric(
                    complete, "incremental_adaptive_recompute"
                ),
                "periodic_recompute_rate": mean_numeric(
                    complete, "periodic_recompute"
                ),
                "incremental_tdsp_queries": incremental_tdsp_queries,
                "incremental_tdsp_cache_hits": incremental_tdsp_hits,
                "incremental_tdsp_cache_hit_rate": (
                    incremental_tdsp_hits / incremental_tdsp_queries
                    if incremental_tdsp_queries
                    else 0.0
                ),
                "incremental_tdsp_lower_path_reuses": sum_numeric(
                    complete, "incremental_tdsp_lower_path_reuses"
                ),
                "incremental_tdsp_dijkstra_runs": sum_numeric(
                    complete, "incremental_tdsp_dijkstra_runs"
                ),
                "full_tdsp_queries": full_tdsp_queries,
                "full_tdsp_cache_hits": full_tdsp_hits,
                "full_tdsp_cache_hit_rate": (
                    full_tdsp_hits / full_tdsp_queries
                    if full_tdsp_queries
                    else 0.0
                ),
                "full_tdsp_dijkstra_runs": sum_numeric(
                    complete, "full_tdsp_dijkstra_runs"
                ),
                "incremental_insertion_transitions_tested": sum_numeric(
                    complete, "incremental_insertion_transitions_tested"
                ),
                "incremental_insertion_labels_generated": sum_numeric(
                    complete, "incremental_insertion_labels_generated"
                ),
                "incremental_insertion_dominance_checks": sum_numeric(
                    complete, "incremental_insertion_dominance_checks"
                ),
                "incremental_dependency_lookups": sum_numeric(
                    complete, "incremental_dependency_lookups"
                ),
                "incremental_dependency_candidates": sum_numeric(
                    complete, "incremental_dependency_candidates"
                ),
                "incremental_plan_dag_builds": sum_numeric(
                    complete, "incremental_plan_dag_builds"
                ),
                "incremental_plan_dag_reuses": sum_numeric(
                    complete, "incremental_plan_dag_reuses"
                ),
                "incremental_plan_prefix_labels_evaluated": sum_numeric(
                    complete, "incremental_plan_prefix_labels_evaluated"
                ),
                "incremental_plan_prefix_transitions_saved": sum_numeric(
                    complete, "incremental_plan_prefix_transitions_saved"
                ),
                "incremental_plan_prefix_cache_hits": sum_numeric(
                    complete, "incremental_plan_prefix_cache_hits"
                ),
                "incremental_plan_prefix_labels_reopened": sum_numeric(
                    complete, "incremental_plan_prefix_labels_reopened"
                ),
                "incremental_plan_prefix_invalidations": sum_numeric(
                    complete, "incremental_plan_prefix_invalidations"
                ),
                "incremental_local_reoptimization_runs": sum_numeric(
                    complete, "incremental_local_reoptimization_runs"
                ),
                "incremental_local_reoptimization_requests": sum_numeric(
                    complete, "incremental_local_reoptimization_requests"
                ),
                "incremental_local_reoptimization_expanded": sum_numeric(
                    complete, "incremental_local_reoptimization_expanded"
                ),
                "mean_incremental_peak_open_size": mean_numeric(
                    complete, "incremental_peak_open_size"
                ),
                "mean_full_peak_open_size": mean_numeric(
                    complete, "full_peak_open_size"
                ),
                "mean_incremental_peak_active_dp_labels": mean_numeric(
                    complete, "incremental_peak_active_dp_labels"
                ),
                "mean_full_peak_active_dp_labels": mean_numeric(
                    complete, "full_peak_active_dp_labels"
                ),
                "incremental_dag_nodes_created": sum_numeric(
                    complete, "incremental_dag_nodes_created"
                ),
                "full_dag_nodes_created": sum_numeric(
                    complete, "full_dag_nodes_created"
                ),
                "full_expanded_states": full_expanded,
                "local_full_expanded_ratio": (
                    local_expanded / full_expanded if full_expanded > 0 else ""
                ),
                "total_incremental_full_expanded_ratio": (
                    incremental_expanded / full_expanded
                    if full_expanded > 0
                    else ""
                ),
            }
        )
    return aggregates


def aggregate_events_by_type(
    event_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for row in event_rows:
        grouped.setdefault(str(row.get("event_type", "Unknown")), []).append(row)
    aggregates: List[Dict[str, object]] = []
    for event_type, rows in sorted(grouped.items()):
        for aggregate in aggregate_events(rows):
            aggregates.append({"event_type": event_type, **aggregate})
    return aggregates


def aggregate_edge_updates_by_subtype(
    event_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    """Aggregate weight increases separately from decreases/restorations."""
    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for row in event_rows:
        if row.get("event_type") != "EdgeUpdate":
            continue
        subtype = str(row.get("event_subtype", "EdgeUpdate"))
        grouped.setdefault(subtype, []).append(row)
    aggregates: List[Dict[str, object]] = []
    for subtype, rows in sorted(grouped.items()):
        for aggregate in aggregate_events(rows):
            aggregates.append({"event_subtype": subtype, **aggregate})
    return aggregates


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
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


def code_fingerprint(workdir: Path) -> str:
    digest = hashlib.sha256()
    for filename in (
        "Dpkmor.py",
        "dynamic_kmor.py",
        "dynamic_kinetic_tree.py",
        "dynamic_rhls.py",
        "dynamic_td_insertion.py",
        "dynamic_experiment_suite.py",
        "dynamic_paired_experiment.py",
    ):
        digest.update(filename.encode("utf-8"))
        digest.update((workdir / filename).read_bytes())
    return digest.hexdigest()[:16]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--manifest", default="dynamic_scenarios/manifest.json")
    parser.add_argument("--output-dir", default="paper_results_dynamic_paired")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--scenarios", default="baseline")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--k-values",
        default="",
        help="Optional comma-separated k sweep; overrides --k when provided.",
    )
    parser.add_argument("--detour-ratio", type=float, default=1.4)
    parser.add_argument("--max-states", type=int, default=30000)
    parser.add_argument("--bucket-size", type=float, default=300.0)
    parser.add_argument("--archive-factor", type=int, default=3)
    parser.add_argument("--replacement-limit-per-route", type=int, default=2)
    parser.add_argument("--arrival-companion-limit", type=int, default=3)
    parser.add_argument("--arrival-batch-sequence-limit", type=int, default=512)
    parser.add_argument("--kinetic-tree-max-schedules", type=int, default=5000)
    parser.add_argument("--kinetic-tree-max-generated", type=int, default=100000)
    parser.add_argument("--rhls-pool-limit", type=int, default=500)
    parser.add_argument("--rhls-max-rounds", type=int, default=6)
    parser.add_argument("--rhls-max-generated", type=int, default=100000)
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
        "--variants",
        default="full",
        help=(
            "Comma-separated experiment variants: full, "
            "incremental_only, no_local_reoptimization, "
            "no_backup_maintenance"
        ),
    )
    parser.add_argument(
        "--full-recompute-warm-tdsp-cache",
        action="store_true",
        help=(
            "Let the FullRecompute oracle retain event-dependent TDSP query "
            "answers. The paper default clears them and retains only static "
            "road-network preprocessing."
        ),
    )
    parser.add_argument(
        "--online-strategies",
        "--online-strategy",
        dest="online_strategies",
        default=UpdateStrategy.INCREMENTAL.value,
        help=(
            "Comma-separated: incremental,periodic_recompute,kinetic_tree,"
            "td_insertion,td_topk_insertion,rhls_topk"
        ),
    )
    parser.add_argument(
        "--periodic-recompute-intervals",
        "--periodic-recompute-interval",
        dest="periodic_recompute_intervals",
        default="5",
        help="Comma-separated refresh intervals used by periodic_recompute",
    )
    parser.add_argument("--disable-request-replacement", action="store_true")
    parser.add_argument("--disable-cancel-refill", action="store_true")
    parser.add_argument("--disable-adaptive-recompute", action="store_true")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument(
        "--paired-order",
        choices=["alternating", "incremental_first", "full_first"],
        default="alternating",
        help=(
            "Execution order inside each paired event. Alternating is the "
            "paper default and counterbalances warm-cache effects."
        ),
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Diagnostic prefix length; 0 processes the complete event stream.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workdir = Path(args.workdir).resolve()
    manifest_path = (workdir / args.manifest).resolve()
    output_dir = (workdir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = {value.upper() for value in parse_values(args.datasets, str)}
    scenarios = {value.lower() for value in parse_values(args.scenarios, str)}
    seeds = set(parse_values(args.seeds, int))
    online_strategies = parse_values(args.online_strategies, str)
    variants = parse_values(args.variants, str)
    k_values = parse_values(args.k_values, int) or [args.k]
    if any(value <= 0 for value in k_values):
        raise ValueError("k values must be positive integers")
    periodic_intervals = parse_values(args.periodic_recompute_intervals, int)
    if not periodic_intervals or any(interval <= 0 for interval in periodic_intervals):
        raise ValueError("periodic recompute intervals must be positive integers")
    allowed_online = {
        UpdateStrategy.INCREMENTAL.value,
        UpdateStrategy.PERIODIC_RECOMPUTE.value,
        UpdateStrategy.KINETIC_TREE.value,
        UpdateStrategy.TD_INSERTION.value,
        UpdateStrategy.TD_TOPK_INSERTION.value,
        UpdateStrategy.RHLS_TOPK.value,
    }
    unknown_online = set(online_strategies) - allowed_online
    if unknown_online:
        raise ValueError(f"unsupported online strategy: {sorted(unknown_online)}")
    unknown_variants = set(variants) - set(EXPERIMENT_VARIANTS)
    if unknown_variants:
        raise ValueError(f"unsupported experiment variant: {sorted(unknown_variants)}")
    fingerprint = code_fingerprint(workdir)

    workloads = [
        workload
        for workload in manifest["workloads"]
        if (not datasets or str(workload["dataset"]).upper() in datasets)
        and (
            not scenarios
            or str(workload.get("scenario", "baseline")).lower() in scenarios
        )
        and (not seeds or int(workload["seed"]) in seeds)
    ]
    payloads: List[Dict[str, object]] = []
    for workload in workloads:
        strategy_settings = []
        for k_value in k_values:
            for online_strategy in online_strategies:
                intervals = (
                    periodic_intervals
                    if online_strategy == UpdateStrategy.PERIODIC_RECOMPUTE.value
                    else [0]
                )
                strategy_settings.extend(
                    (k_value, online_strategy, interval) for interval in intervals
                )
        for k_value, online_strategy, periodic_interval in strategy_settings:
            for variant in variants:
                variant_name = EXPERIMENT_VARIANTS[variant]
                variant_settings = variant_runtime_settings(
                    variant,
                    local_reoptimization_enabled=(
                        not args.disable_arrival_local_reoptimization
                    ),
                    backup_routes_enabled=not args.disable_backup_routes,
                    adaptive_recompute_enabled=(
                        not args.disable_adaptive_recompute
                    ),
                )
                if online_strategy in {
                    UpdateStrategy.KINETIC_TREE.value,
                    UpdateStrategy.TD_INSERTION.value,
                    UpdateStrategy.TD_TOPK_INSERTION.value,
                    UpdateStrategy.RHLS_TOPK.value,
                }:
                    if variant != "full":
                        raise ValueError(
                            f"{online_strategy} supports only --variants full"
                        )
                    variant_name = (
                        "KineticTreeTopK"
                        if online_strategy == UpdateStrategy.KINETIC_TREE.value
                        else "RHLSTopK"
                        if online_strategy == UpdateStrategy.RHLS_TOPK.value
                        else "TDTopKInsertion"
                        if online_strategy == UpdateStrategy.TD_TOPK_INSERTION.value
                        else "TDInsertion"
                    )
                    variant_settings.update(
                        enable_local_reoptimization=False,
                        enable_backup_routes=False,
                        enable_certified_recovery=False,
                        require_incremental_certificate=False,
                        enable_adaptive_recompute=False,
                    )
                task = {
                    "workdir": str(workdir),
                    "workload": workload,
                    "experiment_variant": variant_name,
                    "online_strategy": online_strategy,
                    "periodic_recompute_interval": periodic_interval,
                    "capacity": args.capacity,
                    "k": k_value,
                    "detour_ratio": args.detour_ratio,
                    "max_states": args.max_states,
                    "bucket_size": args.bucket_size,
                    "archive_factor": args.archive_factor,
                    "replacement_limit_per_route": args.replacement_limit_per_route,
                    "arrival_companion_limit": args.arrival_companion_limit,
                    "arrival_batch_sequence_limit": args.arrival_batch_sequence_limit,
                    "kinetic_tree_max_schedules": args.kinetic_tree_max_schedules,
                    "kinetic_tree_max_generated": args.kinetic_tree_max_generated,
                    "rhls_pool_limit": args.rhls_pool_limit,
                    "rhls_max_rounds": args.rhls_max_rounds,
                    "rhls_max_generated": args.rhls_max_generated,
                    "enable_arrival_local_reoptimization": variant_settings[
                        "enable_local_reoptimization"
                    ],
                    "enable_certified_recovery": variant_settings[
                        "enable_certified_recovery"
                    ],
                    "require_incremental_certificate": variant_settings[
                        "require_incremental_certificate"
                    ],
                    "arrival_local_reopt_request_limit": (
                        args.arrival_local_reopt_request_limit
                    ),
                    "arrival_local_reopt_max_states": (
                        args.arrival_local_reopt_max_states
                    ),
                    "refill_request_limit": args.refill_request_limit,
                    "cancel_refill_depth": args.cancel_refill_depth,
                    "enable_request_replacement": not args.disable_request_replacement,
                    "enable_cancel_refill": not args.disable_cancel_refill,
                    "adaptive_recompute_on_failed_arrival": (
                        variant_settings["enable_adaptive_recompute"]
                    ),
                    "adaptive_recompute_on_archive_depletion": (
                        variant_settings["enable_adaptive_recompute"]
                    ),
                    "archive_depletion_ratio": args.archive_depletion_ratio,
                    "enable_backup_routes": variant_settings[
                        "enable_backup_routes"
                    ],
                    "backup_coverage_gap": args.backup_coverage_gap,
                    "backup_cost_regret_limit": args.backup_cost_regret_limit,
                    "backup_deviation_attempts": args.backup_deviation_attempts,
                    "backup_event_order_attempts": args.backup_event_order_attempts,
                    "full_recompute_reuse_tdsp_cache": (
                        args.full_recompute_warm_tdsp_cache
                    ),
                    "paired_order": args.paired_order,
                    "max_events": args.max_events,
                    "code_fingerprint": fingerprint,
                }
                signature = json.dumps(task, sort_keys=True)
                digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
                checkpoint = checkpoint_dir / (
                    f"{workload.get('scenario', 'baseline')}__{workload['dataset']}"
                    f"__seed{workload['seed']}__{online_strategy}"
                    f"__{variant}__k{k_value}__p{periodic_interval}__{digest}.json"
                )
                label = (
                    f"[{workload.get('scenario', 'baseline')} {workload['dataset']} "
                    f"seed={workload['seed']} k={k_value} "
                    f"{online_strategy} {variant_name}"
                    f"{f' p={periodic_interval}' if periodic_interval else ''}]"
                )
                if args.resume and checkpoint.exists():
                    payload = json.loads(
                        checkpoint.read_text(encoding="utf-8")
                    )["payload"]
                    print(f"{label} resumed status={payload['status']}", flush=True)
                else:
                    print(f"{label} running paired oracle...", flush=True)
                    payload = run_isolated(task, args.timeout)
                    checkpoint.write_text(
                        json.dumps(
                            {"task_signature": signature, "payload": payload},
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    print(f"{label} status={payload['status']}", flush=True)
                payloads.append(payload)
                event_rows = [
                    row
                    for item in payloads
                    for row in item.get("rows", [])  # type: ignore[union-attr]
                ]
                write_csv(output_dir / "dynamic_paired_event_results.csv", event_rows)
                write_csv(
                    output_dir / "dynamic_paired_summary_results.csv",
                    summarize(payloads),
                )
                write_csv(
                    output_dir / "dynamic_paired_aggregate_results.csv",
                    aggregate_events(event_rows),
                )
                write_csv(
                    output_dir / "dynamic_paired_event_type_results.csv",
                    aggregate_events_by_type(event_rows),
                )
                write_csv(
                    output_dir / "dynamic_paired_edge_update_results.csv",
                    aggregate_edge_updates_by_subtype(event_rows),
                )
                diagnostic_mismatches = [
                    row
                    for row in event_rows
                    if row.get("experiment_variant") == "IncrementalOnly"
                    and row.get("status") == "COMPLETE"
                    and not (
                        row.get("coverage_match")
                        and row.get("objective_frontier_match")
                        and row.get("service_set_match")
                        and row.get("event_plan_match")
                        and row.get("physical_path_match")
                        and row.get("concrete_topk_match")
                    )
                ]
                write_csv(
                    output_dir / "dynamic_incremental_only_mismatches.csv",
                    diagnostic_mismatches,
                )
                kinetic_tree_mismatches = [
                    row
                    for row in event_rows
                    if row.get("online_strategy")
                    == UpdateStrategy.KINETIC_TREE.value
                    and row.get("status") == "COMPLETE"
                    and not (
                        row.get("coverage_match")
                        and row.get("objective_frontier_match")
                        and row.get("service_set_match")
                        and row.get("event_plan_match")
                        and row.get("physical_path_match")
                        and row.get("concrete_topk_match")
                    )
                ]
                write_csv(
                    output_dir / "dynamic_kinetic_tree_mismatches.csv",
                    kinetic_tree_mismatches,
                )
                rhls_mismatches = [
                    row
                    for row in event_rows
                    if row.get("online_strategy")
                    == UpdateStrategy.RHLS_TOPK.value
                    and row.get("status") == "COMPLETE"
                    and not (
                        row.get("coverage_match")
                        and row.get("objective_frontier_match")
                        and row.get("service_set_match")
                        and row.get("event_plan_match")
                        and row.get("physical_path_match")
                        and row.get("concrete_topk_match")
                    )
                ]
                write_csv(
                    output_dir / "dynamic_rhls_mismatches.csv",
                    rhls_mismatches,
                )

    metadata = {
        "manifest": str(manifest_path),
        "arguments": vars(args),
        "code_fingerprint": fingerprint,
        "completed_runs": len(payloads),
        "statuses": [
            {
                "dataset": payload.get("dataset", ""),
                "seed": payload.get("seed", ""),
                "k": payload.get("k", ""),
                "experiment_variant": payload.get("experiment_variant", ""),
                "online_strategy": payload.get("online_strategy", ""),
                "periodic_recompute_interval": payload.get(
                    "periodic_recompute_interval", ""
                ),
                "status": payload.get("status", ""),
                "error": payload.get("error", ""),
            }
            for payload in payloads
        ],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Saved paired dynamic results to {output_dir}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
