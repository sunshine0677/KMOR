#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event-driven dynamic KMOR.

The implementation reuses the existing static DPKMOR solver as a correctness
baseline. Edge updates use suffix repair, request arrivals use incremental DP
insertion, and cancellations remove stops and reconnect route suffixes. All
handlers fall back to FullRecompute only when no locally repaired route remains.
"""

from __future__ import annotations

import argparse
import copy
import csv
import heapq
import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from Dpkmor import (
    Candidate,
    Graph,
    Request,
    RideSharePlanner,
    SearchCertificate,
    candidate_identity,
    parse_requests_file,
    read_edge_list,
    select_top_k_candidates,
)


EdgeKey = Tuple[str, str]


@dataclass(frozen=True)
class EdgeUpdate:
    edge_id: str
    u: str
    v: str
    event_time: float
    new_weight: float


@dataclass(frozen=True)
class RequestArrival:
    request: Request
    event_time: float


@dataclass(frozen=True)
class RequestCancel:
    request_name: str
    event_time: float


DynamicEvent = Union[EdgeUpdate, RequestArrival, RequestCancel]


class UpdateStrategy(str, Enum):
    INCREMENTAL = "incremental"
    FULL_RECOMPUTE = "full_recompute"
    PERIODIC_RECOMPUTE = "periodic_recompute"
    KINETIC_TREE = "kinetic_tree"
    TD_INSERTION = "td_insertion"
    TD_TOPK_INSERTION = "td_topk_insertion"
    RHLS_TOPK = "rhls_topk"


@dataclass
class DynamicUpdateResult:
    event_type: str
    event_time: float
    active_requests: int
    route_count: int
    best_served: int
    best_cost: float
    update_time: float
    recompute_time: float
    expanded_states: int
    peak_open_size: int = 0
    peak_active_dp_labels: int = 0
    dag_nodes_created: int = 0
    strategy: str = ""
    affected_routes: int = 0
    retained_routes_before_event: int = 0
    affected_route_ratio: float = 0.0
    vehicle_node: str = ""
    planning_time: float = 0.0
    waiting_requests: int = 0
    onboard_requests: int = 0
    completed_requests: int = 0
    cancelled_requests: int = 0
    expired_requests: int = 0
    in_transit: bool = False
    update_mode: str = "FullRecompute"
    repaired_routes: int = 0
    reused_routes: int = 0
    inserted_routes: int = 0
    removed_stops: int = 0
    dp_pruned_states: int = 0
    hit_max_states: bool = False
    timed_out: bool = False
    search_exhausted: bool = False
    scope_complete: bool = False
    unresolved_count: int = 0
    objective_complete: bool = False
    concrete_complete: bool = False
    completed: bool = False
    certificate_stage: str = "none"
    fallback_recompute: bool = False
    archive_size: int = 0
    backup_route_count: int = 0
    available_route_count: int = 0
    available_topk_complete: bool = False
    mean_backup_cost_regret: float = 0.0
    max_backup_cost_regret: float = 0.0
    adaptive_recompute: bool = False
    tdsp_queries: int = 0
    tdsp_cache_hits: int = 0
    tdsp_lower_path_reuses: int = 0
    tdsp_dijkstra_runs: int = 0
    tdsp_cache_invalidations: int = 0
    lower_bound_queries: int = 0
    lower_bound_cache_hits: int = 0
    lower_bound_dijkstra_runs: int = 0
    deviation_queries: int = 0
    deviation_cache_hits: int = 0
    deviation_dijkstra_runs: int = 0
    insertion_transitions_tested: int = 0
    insertion_labels_generated: int = 0
    insertion_dominance_checks: int = 0
    insertion_resource_checks: int = 0
    dependency_lookups: int = 0
    dependency_candidates: int = 0
    plan_dag_builds: int = 0
    plan_dag_reuses: int = 0
    plan_prefix_labels_evaluated: int = 0
    plan_prefix_transitions_saved: int = 0
    plan_prefix_cache_hits: int = 0
    plan_prefix_labels_reopened: int = 0
    plan_prefix_invalidations: int = 0
    local_reoptimization_runs: int = 0
    local_reoptimization_requests: int = 0
    local_reoptimization_expanded: int = 0
    generated_schedules: int = 0
    feasible_schedules: int = 0
    evicted_schedules: int = 0
    schedule_generation_capped: bool = False
    local_search_iterations: int = 0
    local_search_moves: int = 0
    candidate_pool_peak_size: int = 0
    local_search_iteration_capped: bool = False
    note: str = ""


@dataclass
class DynamicSearchDiagnostics:
    """Low-overhead counters for one dynamic event."""

    tdsp_queries: int = 0
    tdsp_cache_hits: int = 0
    tdsp_lower_path_reuses: int = 0
    tdsp_dijkstra_runs: int = 0
    tdsp_cache_invalidations: int = 0
    lower_bound_queries: int = 0
    lower_bound_cache_hits: int = 0
    lower_bound_dijkstra_runs: int = 0
    deviation_queries: int = 0
    deviation_cache_hits: int = 0
    deviation_dijkstra_runs: int = 0
    insertion_transitions_tested: int = 0
    insertion_labels_generated: int = 0
    insertion_dominance_checks: int = 0
    insertion_resource_checks: int = 0
    dependency_lookups: int = 0
    dependency_candidates: int = 0
    plan_dag_builds: int = 0
    plan_dag_reuses: int = 0
    plan_prefix_labels_evaluated: int = 0
    plan_prefix_transitions_saved: int = 0
    plan_prefix_cache_hits: int = 0
    plan_prefix_labels_reopened: int = 0
    plan_prefix_invalidations: int = 0
    local_reoptimization_runs: int = 0
    local_reoptimization_requests: int = 0
    local_reoptimization_expanded: int = 0
    peak_open_size: int = 0
    peak_active_dp_labels: int = 0
    dag_nodes_created: int = 0

    def observe_planner_stats(self, stats) -> None:
        """Accumulate storage counters from planners used in one event."""
        self.peak_open_size = max(self.peak_open_size, stats.peak_open_size)
        self.peak_active_dp_labels = max(
            self.peak_active_dp_labels,
            stats.peak_active_dp_labels,
        )
        self.dag_nodes_created += stats.dag_nodes_created

    def attach(self, result: DynamicUpdateResult) -> DynamicUpdateResult:
        for name in self.__dataclass_fields__:
            setattr(result, name, getattr(self, name))
        return result


@dataclass(frozen=True)
class RouteSegment:
    u: str
    v: str
    edge_key: EdgeKey
    depart_time: float
    arrive_time: float
    travel_time: float


@dataclass(frozen=True)
class ScheduledStop:
    node: str
    action: str
    request_name: str
    event_time: float


@dataclass
class DynamicRouteRecord:
    route_index: int
    candidate: Candidate
    segments: List[RouteSegment]
    stops: List[ScheduledStop]

    @property
    def edge_keys(self) -> List[EdgeKey]:
        return [segment.edge_key for segment in self.segments]


@dataclass
class TimedEdge:
    to_node: str
    base_weight: float
    schedule: Dict[int, float] = field(default_factory=dict)

    def weight_at_bucket(self, bucket: int) -> float:
        if not self.schedule:
            return self.base_weight
        active_buckets = [start for start in self.schedule if start <= bucket]
        if not active_buckets:
            return self.base_weight
        return self.schedule[max(active_buckets)]

    def weight_at_time(self, depart_time: float, bucket_size: float) -> float:
        """Return travel time under the FIFO closure of the bucket schedule.

        For raw arrival function ``A(t) = t + w(t)``, the FIFO closure is its
        prefix maximum.  At a weight-change boundary the previous interval's
        right-limit must be considered; otherwise a congestion recovery can
        incorrectly let a later departure arrive before an earlier one.
        """
        if not self.schedule:
            return self.base_weight
        if bucket_size <= 0:
            return self.weight_at_bucket(0)

        departure = max(0.0, float(depart_time))
        current_weight = self.base_weight
        boundary_arrival = -math.inf
        for start_bucket in sorted(self.schedule):
            start_time = start_bucket * bucket_size
            if start_time > departure:
                break
            # The previous interval approaches this boundary from the left.
            # Its arrival value is part of the prefix envelope at and after it.
            boundary_arrival = max(boundary_arrival, start_time + current_weight)
            current_weight = self.schedule[start_bucket]
        arrival_time = max(departure + current_weight, boundary_arrival)
        return max(0.0, arrival_time - departure)


class RequestStatus(str, Enum):
    WAITING = "WAITING"
    ONBOARD = "ONBOARD"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


@dataclass
class DynamicVehicleState:
    """State at the next graph node where a routing decision can be made."""

    node: str
    current_time: float = 0.0
    planning_time: float = 0.0
    traveled_cost: float = 0.0
    in_transit: bool = False
    pickup_times: Dict[str, float] = field(default_factory=dict)


@dataclass
class InsertionLabel:
    base_index: int
    new_phase: int  # 0: waiting, 1: onboard, 2: dropped off
    node: str
    arrival_time: float
    load: int
    onboard: frozenset[str]
    key_path: Tuple[str, ...]
    full_path: Tuple[str, ...]
    route_events: Tuple[Tuple[str, str], ...]
    pickup_times: Dict[str, float]
    served_names: Tuple[str, ...]


@dataclass(eq=False)
class PlanPrefixNode:
    """One exact event-history prefix shared by archived plans."""

    children: Dict[Tuple[str, str], "PlanPrefixNode"] = field(
        default_factory=dict
    )
    terminal: bool = False
    cached_label: Optional[InsertionLabel] = None
    cache_context: Optional[Tuple[object, ...]] = None
    cache_ready: bool = False
    invalidated: bool = True
    dependency_edges: frozenset[EdgeKey] = frozenset()


@dataclass
class PersistentPlanDAG:
    signature: Tuple[Tuple[Tuple[str, str], ...], ...]
    root: PlanPrefixNode
    input_transition_count: int
    unique_transition_count: int


@dataclass
class PairedEventOutcome:
    incremental_manager: "DynamicKMORManager"
    full_manager: "DynamicKMORManager"
    incremental_result: DynamicUpdateResult
    full_result: DynamicUpdateResult
    branch_clone_time: float = 0.0
    execution_order: str = "incremental_first"


class TimeDependentGraph:
    """Bucketed time-dependent graph with FIFO-friendly non-negative weights."""

    def __init__(self, directed: bool = False, bucket_size: float = 300.0):
        self.directed = directed
        self.bucket_size = float(bucket_size)
        self.adj: Dict[str, List[TimedEdge]] = {}
        self._shared_nodes: set[str] = set()
        self.revision = 0
        self.lower_bound_revision = 0

    def fork(self) -> "TimeDependentGraph":
        """Create a cheap branch with node-level copy-on-write adjacency.

        Paired evaluation branches at every event. Copying the complete road
        network there is unnecessary because an event changes at most one
        edge. Both branches share TimedEdge objects until either branch writes
        an endpoint's adjacency list.
        """
        cloned = TimeDependentGraph(
            directed=self.directed,
            bucket_size=self.bucket_size,
        )
        cloned.adj = self.adj.copy()
        shared = set(self.adj)
        self._shared_nodes.update(shared)
        cloned._shared_nodes = shared
        cloned.revision = self.revision
        cloned.lower_bound_revision = self.lower_bound_revision
        return cloned

    def _ensure_node_owned(self, node: str) -> None:
        node = str(node)
        if node not in self._shared_nodes:
            return
        self.adj[node] = [
            TimedEdge(edge.to_node, edge.base_weight, dict(edge.schedule))
            for edge in self.adj.get(node, ())
        ]
        self._shared_nodes.discard(node)

    @classmethod
    def from_static_graph(cls, graph: Graph, bucket_size: float = 300.0) -> "TimeDependentGraph":
        td_graph = cls(directed=graph.directed, bucket_size=bucket_size)
        seen = set()
        for u, neighbors in graph.adj.items():
            for v, weight in neighbors:
                key = td_graph.edge_key(u, v)
                if not graph.directed and key in seen:
                    continue
                td_graph.add_edge(u, v, weight)
                seen.add(key)
        return td_graph

    def edge_key(self, u: str, v: str) -> EdgeKey:
        if self.directed:
            return (str(u), str(v))
        return tuple(sorted((str(u), str(v))))  # type: ignore[return-value]

    def time_bucket(self, t: float) -> int:
        if self.bucket_size <= 0:
            return 0
        return int(max(0.0, t) // self.bucket_size)

    def add_edge(self, u: str, v: str, weight: float) -> None:
        self._ensure_node_owned(str(u))
        self._ensure_node_owned(str(v))
        self.adj.setdefault(str(u), []).append(TimedEdge(str(v), float(weight)))
        self.adj.setdefault(str(v), [])
        if not self.directed:
            self.adj.setdefault(str(v), []).append(TimedEdge(str(u), float(weight)))

    def nodes(self) -> Iterable[str]:
        return self.adj.keys()

    def neighbors_at(self, u: str, depart_time: float) -> Iterable[Tuple[str, float]]:
        for edge in self.adj.get(str(u), []):
            yield edge.to_node, edge.weight_at_time(depart_time, self.bucket_size)

    def edge_weight_at(self, u: str, v: str, depart_time: float) -> float:
        for edge in self.adj.get(str(u), []):
            if edge.to_node == str(v):
                return edge.weight_at_time(depart_time, self.bucket_size)
        return math.inf

    def path_travel_time(
        self,
        path: Sequence[str],
        depart_time: float,
    ) -> Tuple[float, bool]:
        """Evaluate a path and report whether every edge remains at its lower bound."""
        current_time = float(depart_time)
        at_lower_bound = True
        for u, v in zip(path, path[1:]):
            weight = self.edge_weight_at(u, v, current_time)
            if math.isinf(weight):
                return math.inf, False
            minimum = self.minimum_edge_weight(u, v)
            if weight > minimum + 1e-9:
                at_lower_bound = False
            current_time += weight
        return current_time - float(depart_time), at_lower_bound

    def update_edge_weight(self, u: str, v: str, event_time: float, new_weight: float) -> None:
        old_minimum = self.minimum_edge_weight(u, v)
        bucket = self.time_bucket(event_time)
        self._update_directed_edge(u, v, bucket, new_weight)
        if not self.directed:
            self._update_directed_edge(v, u, bucket, new_weight)
        self.revision += 1
        if float(new_weight) < old_minimum:
            self.lower_bound_revision += 1

    def minimum_edge_weight(self, u: str, v: str) -> float:
        weights = [
            min([edge.base_weight, *edge.schedule.values()])
            for edge in self.adj.get(str(u), [])
            if edge.to_node == str(v)
        ]
        return min(weights, default=math.inf)

    def edge_weight_at(self, u: str, v: str, event_time: float) -> float:
        weights = [
            edge.weight_at_time(event_time, self.bucket_size)
            for edge in self.adj.get(str(u), [])
            if edge.to_node == str(v)
        ]
        return min(weights, default=math.inf)

    def _update_directed_edge(self, u: str, v: str, bucket: int, new_weight: float) -> None:
        self._ensure_node_owned(str(u))
        updated = False
        for edge in self.adj.get(str(u), []):
            if edge.to_node == str(v):
                edge.schedule[bucket] = float(new_weight)
                updated = True
        if updated:
            return
        self.adj.setdefault(str(u), []).append(TimedEdge(str(v), float(new_weight), {bucket: float(new_weight)}))
        self.adj.setdefault(str(v), [])

    def shortest_path(
        self,
        source: str,
        target: str,
        depart_time: float = 0.0,
        blocked_edges: Optional[Iterable[EdgeKey]] = None,
    ) -> Tuple[List[str], float]:
        """Time-dependent Dijkstra using edge weight at node departure time."""
        if source == target:
            return [str(source)], 0.0

        blocked = set(blocked_edges or ())
        pq: List[Tuple[float, str]] = [(float(depart_time), str(source))]
        arrival: Dict[str, float] = {str(source): float(depart_time)}
        prev: Dict[str, Optional[str]] = {str(source): None}
        visited = set()

        while pq:
            arr_time, node = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            if node == str(target):
                break

            for nxt, travel_time in self.neighbors_at(node, arr_time):
                if blocked and self.edge_key(node, nxt) in blocked:
                    continue
                if travel_time < 0:
                    continue
                nxt_arr = arr_time + travel_time
                if nxt_arr < arrival.get(nxt, math.inf):
                    arrival[nxt] = nxt_arr
                    prev[nxt] = node
                    heapq.heappush(pq, (nxt_arr, nxt))

        if str(target) not in arrival:
            return [], math.inf

        path: List[str] = []
        cur: Optional[str] = str(target)
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()
        return path, arrival[str(target)] - float(depart_time)

    def snapshot_graph(self, snapshot_time: float) -> Graph:
        """Convert current bucket to a static graph for the existing KMOR solver."""
        graph = Graph(directed=self.directed)
        seen = set()
        for u, neighbors in self.adj.items():
            for edge in neighbors:
                key = self.edge_key(u, edge.to_node)
                if not self.directed and key in seen:
                    continue
                graph.add_edge(
                    u,
                    edge.to_node,
                    edge.weight_at_time(snapshot_time, self.bucket_size),
                )
                seen.add(key)
        return graph

    def lower_bound_graph(self) -> Graph:
        """Static per-edge minima used only for admissible pruning bounds."""
        graph = Graph(directed=self.directed)
        seen = set()
        for u, neighbors in self.adj.items():
            for edge in neighbors:
                key = self.edge_key(u, edge.to_node)
                if not self.directed and key in seen:
                    continue
                minimum = min([edge.base_weight, *edge.schedule.values()])
                graph.add_edge(u, edge.to_node, minimum)
                seen.add(key)
        return graph


class DynamicKMORManager:
    """Maintains active requests and top-k routes under dynamic events."""

    def __init__(
        self,
        graph: TimeDependentGraph,
        source: str,
        target: str,
        initial_requests: Sequence[Request],
        **planner_params: object,
    ):
        self.graph = graph
        self.source = str(source)
        self.target = str(target)
        self.active_requests: Dict[str, Request] = {req.name: req for req in initial_requests}
        self.request_status: Dict[str, RequestStatus] = {
            req.name: RequestStatus.WAITING for req in initial_requests
        }
        strategy = planner_params.pop("strategy", UpdateStrategy.INCREMENTAL.value)
        strategy_value = strategy.value if isinstance(strategy, UpdateStrategy) else str(strategy)
        self.strategy = UpdateStrategy(strategy_value)
        self.planner_params = dict(planner_params)
        output_k = max(1, int(self.planner_params.get("k", 5)))
        archive_factor = max(
            1,
            int(self.planner_params.pop("archive_factor", 3)),
        )
        self.archive_limit = max(
            output_k,
            int(self.planner_params.pop("archive_limit", output_k * archive_factor)),
        )
        self.enable_request_replacement = bool(
            self.planner_params.pop("enable_request_replacement", True)
        )
        self.replacement_limit_per_route = max(
            0,
            int(self.planner_params.pop("replacement_limit_per_route", 2)),
        )
        self.arrival_companion_limit = max(
            0,
            int(self.planner_params.pop("arrival_companion_limit", min(3, output_k))),
        )
        self.arrival_batch_sequence_limit = max(
            1,
            int(self.planner_params.pop("arrival_batch_sequence_limit", 512)),
        )
        self.enable_arrival_local_reoptimization = bool(
            self.planner_params.pop("enable_arrival_local_reoptimization", True)
        )
        self.arrival_local_reopt_request_limit = max(
            1,
            int(self.planner_params.pop("arrival_local_reopt_request_limit", 12)),
        )
        default_local_states = min(
            50000,
            max(1, int(self.planner_params.get("max_states", 30000))),
        )
        self.arrival_local_reopt_max_states = max(
            1,
            int(
                self.planner_params.pop(
                    "arrival_local_reopt_max_states",
                    default_local_states,
                )
            ),
        )
        self.enable_cancel_refill = bool(
            self.planner_params.pop("enable_cancel_refill", True)
        )
        self.refill_request_limit = max(
            0,
            int(self.planner_params.pop("refill_request_limit", output_k)),
        )
        self.cancel_refill_depth = max(
            1,
            int(self.planner_params.pop("cancel_refill_depth", 2)),
        )
        self.adaptive_recompute_on_failed_arrival = bool(
            self.planner_params.pop("adaptive_recompute_on_failed_arrival", True)
        )
        self.adaptive_recompute_on_archive_depletion = bool(
            self.planner_params.pop(
                "adaptive_recompute_on_archive_depletion",
                True,
            )
        )
        self.archive_depletion_ratio = min(
            1.0,
            max(
                0.0,
                float(self.planner_params.pop("archive_depletion_ratio", 0.5)),
            ),
        )
        self.enable_backup_routes = bool(
            self.planner_params.pop("enable_backup_routes", True)
        )
        self.full_recompute_reuse_tdsp_cache = bool(
            self.planner_params.pop("full_recompute_reuse_tdsp_cache", False)
        )
        self.backup_coverage_gap = max(
            0,
            int(self.planner_params.pop("backup_coverage_gap", 1)),
        )
        self.backup_cost_regret_limit = max(
            0.0,
            float(self.planner_params.pop("backup_cost_regret_limit", 0.10)),
        )
        self.backup_deviation_attempts = max(
            0,
            int(
                self.planner_params.pop(
                    "backup_deviation_attempts",
                    max(24, output_k * 12),
                )
            ),
        )
        self.backup_event_order_attempts = max(
            0,
            int(
                self.planner_params.pop(
                    "backup_event_order_attempts",
                    max(12, output_k * 4),
                )
            ),
        )
        self.periodic_recompute_interval = max(
            1,
            int(self.planner_params.pop("periodic_recompute_interval", 5)),
        )
        self.processed_event_count = 0
        self.strict_static_semantics = bool(
            self.planner_params.pop("strict_static_semantics", True)
        )
        self.enable_certified_recovery = bool(
            self.planner_params.pop(
                "enable_certified_recovery",
                False,
            )
        )
        if self.strict_static_semantics:
            # The formal dynamic algorithm must solve the same constrained,
            # plan-level problem as the final static bundled implementation.
            self.planner_params.update(
                dp_mode="bundled",
                pareto_only=True,
                enforce_passenger_detour=True,
                enable_compatibility_pruning=False,
            )
            self.planner_params.pop("compatibility_ratio", None)
        else:
            # Explicit compatibility mode for reproducing the earlier dynamic
            # experiments; it is not the formal paper algorithm.
            self.planner_params.setdefault("pareto_only", False)
            self.planner_params.setdefault("enforce_passenger_detour", False)
            self.planner_params.setdefault("enable_compatibility_pruning", True)
            self.planner_params.setdefault(
                "compatibility_ratio",
                float(self.planner_params.get("detour_ratio", 1.8)),
            )
        self.current_time = 0.0
        self.vehicle = DynamicVehicleState(node=self.source)
        self.routes: List[Candidate] = []
        self.backup_routes: List[Candidate] = []
        self.available_routes: List[Candidate] = []
        # The public answer remains strict plan-level Pareto Top-k.  This
        # bounded internal archive preserves alternate concrete plans that can
        # become useful after later traffic or request events.
        self.route_archive: List[Candidate] = []
        self._last_local_search_certificate = SearchCertificate(
            scope_complete=False,
        )
        self._current_search_certificate = SearchCertificate(
            scope_complete=False,
        )
        # A full archive warns that plans just outside the retained prefix may
        # enter the Top-k after a later event. Incremental handlers use this as
        # a conservative frontier-completeness certificate.
        self.route_archive_saturated = False
        self.route_records: List[DynamicRouteRecord] = []
        self._route_record_by_identity: Dict[
            Tuple[object, ...], DynamicRouteRecord
        ] = {}
        self._route_keys_by_edge: Dict[
            EdgeKey, set[Tuple[object, ...]]
        ] = {}
        self._route_keys_by_request: Dict[
            str, set[Tuple[object, ...]]
        ] = {}
        self._persistent_plan_dag: Optional[PersistentPlanDAG] = None
        self._plan_nodes_by_edge: Dict[EdgeKey, set[PlanPrefixNode]] = {}
        self._lower_bound_graph_cache: Optional[Graph] = None
        self._lower_bound_cache_revision = -1
        self._lower_bound_shortest_path_cache: Dict[
            Tuple[str, str], Tuple[List[str], float]
        ] = {}
        self._corridor_path: List[str] = []
        self._corridor_index: Dict[str, int] = {}
        self._corridor_prefix_cost: List[float] = []
        self._dynamic_shortest_path_cache: Dict[
            Tuple[str, str, float], Tuple[List[str], float]
        ] = {}
        self._dynamic_path_keys_by_edge: Dict[
            EdgeKey, set[Tuple[str, str, float]]
        ] = {}
        self._deviation_shortest_path_cache: Dict[
            Tuple[str, str, frozenset[EdgeKey]],
            Tuple[List[str], float],
        ] = {}
        self.last_deviation_attempts = 0
        self.last_deviation_new_routes = 0
        self.last_backup_new_routes = 0
        self._event_diagnostics = DynamicSearchDiagnostics()

    def clone(
        self,
        strategy: Optional[Union[UpdateStrategy, str]] = None,
    ) -> "DynamicKMORManager":
        # Route candidates and cached paths are immutable by convention. A
        # shallow structural clone plus independent mutable indexes is enough,
        # while TimeDependentGraph.fork() isolates future edge updates.
        cloned = copy.copy(self)
        cloned.graph = self.graph.fork()
        cloned.active_requests = dict(self.active_requests)
        cloned.request_status = dict(self.request_status)
        cloned.planner_params = dict(self.planner_params)
        cloned.vehicle = copy.deepcopy(self.vehicle)
        cloned.routes = list(self.routes)
        cloned.backup_routes = list(self.backup_routes)
        cloned.available_routes = list(self.available_routes)
        cloned.route_archive = list(self.route_archive)
        cloned.route_records = list(self.route_records)
        cloned._route_record_by_identity = dict(self._route_record_by_identity)
        cloned._route_keys_by_edge = {
            key: set(values) for key, values in self._route_keys_by_edge.items()
        }
        cloned._route_keys_by_request = {
            key: set(values) for key, values in self._route_keys_by_request.items()
        }
        (
            cloned._persistent_plan_dag,
            cloned._plan_nodes_by_edge,
        ) = copy.deepcopy(
            (self._persistent_plan_dag, self._plan_nodes_by_edge)
        )
        # Static lower-bound graphs and path values are read-only. Cache
        # containers remain branch-local because an edge update may clear or
        # extend them.
        cloned._lower_bound_shortest_path_cache = dict(
            self._lower_bound_shortest_path_cache
        )
        cloned._corridor_path = list(self._corridor_path)
        cloned._corridor_index = dict(self._corridor_index)
        cloned._corridor_prefix_cost = list(self._corridor_prefix_cost)
        cloned._dynamic_shortest_path_cache = dict(
            self._dynamic_shortest_path_cache
        )
        cloned._dynamic_path_keys_by_edge = {
            key: set(values)
            for key, values in self._dynamic_path_keys_by_edge.items()
        }
        cloned._deviation_shortest_path_cache = dict(
            self._deviation_shortest_path_cache
        )
        cloned._event_diagnostics = DynamicSearchDiagnostics()
        if strategy is not None:
            strategy_value = (
                strategy.value if isinstance(strategy, UpdateStrategy) else str(strategy)
            )
            cloned.strategy = UpdateStrategy(strategy_value)
        return cloned

    def reset_event_diagnostics(self) -> None:
        self._event_diagnostics = DynamicSearchDiagnostics()

    def attach_event_diagnostics(
        self,
        result: DynamicUpdateResult,
    ) -> DynamicUpdateResult:
        return self._event_diagnostics.attach(result)

    def attach_event_context(
        self,
        result: DynamicUpdateResult,
        retained_routes_before_event: int,
    ) -> DynamicUpdateResult:
        result.retained_routes_before_event = retained_routes_before_event
        result.affected_route_ratio = (
            result.affected_routes / retained_routes_before_event
            if retained_routes_before_event > 0
            else 0.0
        )
        return self.attach_event_diagnostics(result)

    def cached_lower_bound_graph(self) -> Graph:
        revision = self.graph.lower_bound_revision
        if (
            self._lower_bound_graph_cache is None
            or self._lower_bound_cache_revision != revision
        ):
            self._lower_bound_graph_cache = self.graph.lower_bound_graph()
            self._lower_bound_cache_revision = revision
            self._lower_bound_shortest_path_cache.clear()
            self._deviation_shortest_path_cache.clear()
            self._corridor_path = []
            self._corridor_index.clear()
            self._corridor_prefix_cost = []
        return self._lower_bound_graph_cache

    def _ensure_corridor_index(self) -> None:
        if self._corridor_path:
            return
        base = self._lower_bound_shortest_path_cache.get((self.source, self.target))
        if base is None or not base[0]:
            return
        path = list(base[0])
        graph = self.cached_lower_bound_graph()
        prefix = [0.0]
        for left, right in zip(path, path[1:]):
            weights = [weight for node, weight in graph.neighbors(left) if node == right]
            if not weights:
                return
            prefix.append(prefix[-1] + min(weights))
        self._corridor_path = path
        self._corridor_index = {node: index for index, node in enumerate(path)}
        self._corridor_prefix_cost = prefix

    def _corridor_shortest_path(
        self,
        source: str,
        target: str,
    ) -> Optional[Tuple[List[str], float]]:
        self._ensure_corridor_index()
        left = self._corridor_index.get(str(source))
        right = self._corridor_index.get(str(target))
        if left is None or right is None:
            return None
        if left <= right:
            path = self._corridor_path[left : right + 1]
        else:
            path = list(reversed(self._corridor_path[right : left + 1]))
        cost = abs(
            self._corridor_prefix_cost[right] - self._corridor_prefix_cost[left]
        )
        return path, cost

    def cached_dynamic_shortest_path(
        self,
        source: str,
        target: str,
        depart_time: float,
    ) -> Tuple[List[str], float]:
        self._event_diagnostics.tdsp_queries += 1
        key = (
            str(source),
            str(target),
            round(float(depart_time), 6),
        )
        cached = self._dynamic_shortest_path_cache.get(key)
        if cached is not None:
            self._event_diagnostics.tdsp_cache_hits += 1
            return cached
        lower_path, _ = self.cached_lower_bound_shortest_path(source, target)
        lower_path_time, remains_optimal = self.graph.path_travel_time(
            lower_path,
            depart_time,
        )
        if lower_path and remains_optimal:
            self._event_diagnostics.tdsp_lower_path_reuses += 1
            result = (lower_path, lower_path_time)
        else:
            self._event_diagnostics.tdsp_dijkstra_runs += 1
            result = self.graph.shortest_path(
                str(source),
                str(target),
                float(depart_time),
            )
        if len(self._dynamic_shortest_path_cache) >= 100000:
            self._clear_dynamic_shortest_path_cache()
        self._store_dynamic_shortest_path(key, result)
        return result

    def _store_dynamic_shortest_path(
        self,
        key: Tuple[str, str, float],
        result: Tuple[List[str], float],
    ) -> None:
        self._dynamic_shortest_path_cache[key] = result
        path = result[0]
        for left, right in zip(path, path[1:]):
            edge = self.graph.edge_key(left, right)
            self._dynamic_path_keys_by_edge.setdefault(edge, set()).add(key)

    def _remove_dynamic_shortest_path(
        self,
        key: Tuple[str, str, float],
    ) -> bool:
        cached = self._dynamic_shortest_path_cache.pop(key, None)
        if cached is None:
            return False
        path = cached[0]
        for left, right in zip(path, path[1:]):
            edge = self.graph.edge_key(left, right)
            dependent_keys = self._dynamic_path_keys_by_edge.get(edge)
            if dependent_keys is None:
                continue
            dependent_keys.discard(key)
            if not dependent_keys:
                self._dynamic_path_keys_by_edge.pop(edge, None)
        return True

    def _clear_dynamic_shortest_path_cache(self) -> None:
        self._dynamic_shortest_path_cache.clear()
        self._dynamic_path_keys_by_edge.clear()

    def invalidate_dynamic_paths_for_edge(
        self,
        u: str,
        v: str,
        clear_all: bool = False,
    ) -> None:
        if clear_all:
            self._event_diagnostics.tdsp_cache_invalidations += len(
                self._dynamic_shortest_path_cache
            )
            self._clear_dynamic_shortest_path_cache()
            return

        updated_edge = self.graph.edge_key(str(u), str(v))
        stale_keys = tuple(self._dynamic_path_keys_by_edge.get(updated_edge, ()))
        invalidated = sum(
            self._remove_dynamic_shortest_path(key) for key in stale_keys
        )
        self._event_diagnostics.tdsp_cache_invalidations += invalidated

    def cached_deviation_shortest_path(
        self,
        source: str,
        target: str,
        blocked_edges: Iterable[EdgeKey],
    ) -> Tuple[List[str], float]:
        self._event_diagnostics.deviation_queries += 1
        graph = self.cached_lower_bound_graph()
        blocked = frozenset(blocked_edges)
        key = (
            str(source),
            str(target),
            blocked,
        )
        cached = self._deviation_shortest_path_cache.get(key)
        if cached is not None:
            self._event_diagnostics.deviation_cache_hits += 1
            return cached
        self._event_diagnostics.deviation_dijkstra_runs += 1
        result = graph.shortest_path_avoiding_edges(
            str(source),
            str(target),
            blocked,
        )
        if len(self._deviation_shortest_path_cache) >= 5000:
            self._deviation_shortest_path_cache.clear()
        self._deviation_shortest_path_cache[key] = result
        return result

    def apply_edge_update(self, event: EdgeUpdate) -> None:
        old_weight = self.graph.edge_weight_at(event.u, event.v, event.event_time)
        self.graph.update_edge_weight(
            event.u,
            event.v,
            event.event_time,
            event.new_weight,
        )
        # A faster edge can improve paths that did not previously use it.
        clear_all = float(event.new_weight) < old_weight
        if self.strategy != UpdateStrategy.FULL_RECOMPUTE:
            self.invalidate_plan_prefixes_for_edge(
                event.u,
                event.v,
                clear_all=clear_all,
            )
        else:
            self._persistent_plan_dag = None
            self._plan_nodes_by_edge.clear()
        self.invalidate_dynamic_paths_for_edge(
            event.u,
            event.v,
            clear_all=clear_all,
        )

    def cached_lower_bound_shortest_path(
        self,
        source: str,
        target: str,
    ) -> Tuple[List[str], float]:
        self._event_diagnostics.lower_bound_queries += 1
        graph = self.cached_lower_bound_graph()
        key = (str(source), str(target))
        cached = self._lower_bound_shortest_path_cache.get(key)
        if cached is not None:
            self._event_diagnostics.lower_bound_cache_hits += 1
            if key == (self.source, self.target):
                self._ensure_corridor_index()
            return cached

        canonical_key = key
        reverse_requested = False
        if not graph.directed and key[1] < key[0]:
            canonical_key = (key[1], key[0])
            reverse_requested = True
        canonical = self._lower_bound_shortest_path_cache.get(canonical_key)
        if canonical is not None:
            self._event_diagnostics.lower_bound_cache_hits += 1
        else:
            self._event_diagnostics.lower_bound_dijkstra_runs += 1
            # RideSharePlanner fills its shared cache with the single-source
            # implementation. Use the same routine here so equal-distance
            # alternatives have one canonical physical path regardless of
            # whether the incremental or full branch discovers a pair first.
            # A source-target corridor subpath can be equally short but select
            # a different tied trajectory, so it must not define this cache.
            canonical = graph.shortest_paths_from(
                canonical_key[0],
                [canonical_key[1]],
            )[canonical_key[1]]
            self._lower_bound_shortest_path_cache[canonical_key] = canonical

        reverse_key = (canonical_key[1], canonical_key[0])
        reverse_result = (list(reversed(canonical[0])), canonical[1])
        self._lower_bound_shortest_path_cache[reverse_key] = reverse_result
        result = reverse_result if reverse_requested else canonical
        self._lower_bound_shortest_path_cache[key] = result
        if key == (self.source, self.target):
            self._ensure_corridor_index()
        return result

    def warm_request_lower_bound_paths(self, request: Request) -> None:
        """Batch-cache static paths introduced by one arriving request."""
        graph = self.cached_lower_bound_graph()
        key_nodes = {self.vehicle.node, self.target}
        for active_request in self.active_requests.values():
            key_nodes.add(str(active_request.pickup))
            key_nodes.add(str(active_request.dropoff))

        for source in (str(request.pickup), str(request.dropoff)):
            missing_targets = sorted(
                target
                for target in key_nodes
                if target != source
                and (source, target) not in self._lower_bound_shortest_path_cache
                and (
                    graph.directed
                    or (target, source)
                    not in self._lower_bound_shortest_path_cache
                )
            )
            if not missing_targets:
                continue
            self._event_diagnostics.lower_bound_queries += len(missing_targets)
            self._event_diagnostics.lower_bound_dijkstra_runs += 1
            results = graph.shortest_paths_from(source, missing_targets)
            for target in missing_targets:
                path, distance = results[target]
                self._lower_bound_shortest_path_cache[(source, target)] = (
                    path,
                    distance,
                )
                if not graph.directed:
                    self._lower_bound_shortest_path_cache[(target, source)] = (
                        list(reversed(path)),
                        distance,
                    )

    def initialize(self, current_time: float = 0.0) -> DynamicUpdateResult:
        self.reset_event_diagnostics()
        result = self.full_recompute(
            current_time,
            "Initialize",
            note="initial solution",
        )
        return self.attach_event_diagnostics(result)

    def handle_event(self, event: DynamicEvent) -> DynamicUpdateResult:
        self.reset_event_diagnostics()
        retained_routes_before_event = len(self.current_candidate_pool())
        if self.strategy == UpdateStrategy.FULL_RECOMPUTE:
            return self.attach_event_context(
                self.handle_event_full_recompute(event),
                retained_routes_before_event,
            )
        if self.strategy == UpdateStrategy.PERIODIC_RECOMPUTE:
            self.processed_event_count += 1
            if self.processed_event_count % self.periodic_recompute_interval == 0:
                result = self.handle_event_full_recompute(event)
                result.update_mode = "PeriodicFullRecompute"
                return self.attach_event_context(
                    result,
                    retained_routes_before_event,
                )
        if isinstance(event, EdgeUpdate):
            update_start = time.perf_counter()
            self.current_time = event.event_time
            affected_records = self.find_affected_routes(
                event.u,
                event.v,
                event.event_time,
            )
            affected_route_keys = {
                candidate_identity(record.candidate) for record in affected_records
            }
            affected_routes = len(affected_records)
            old_weight = self.graph.edge_weight_at(
                event.u,
                event.v,
                event.event_time,
            )
            edge_became_faster = float(event.new_weight) < old_weight
            frontier_refresh_reasons: List[str] = []
            if edge_became_faster:
                frontier_refresh_reasons.append("edge_speedup")
            if affected_routes > 0 and self.route_archive_saturated:
                frontier_refresh_reasons.append("affected_saturated_archive")
            previous_routes = self.current_candidate_pool()
            previous_formal_count = len(
                self.route_plan_representatives(self.routes)
            )
            previous_decision_state = (
                self.vehicle.node,
                self.vehicle.planning_time,
            )
            self.advance_vehicle(event.event_time)
            state_advanced = previous_decision_state != (
                self.vehicle.node,
                self.vehicle.planning_time,
            )
            self.apply_edge_update(event)
            return self.attach_event_context(
                self.incremental_edge_update(
                    event.event_time,
                    previous_routes,
                    affected_routes=affected_routes,
                    affected_route_keys=affected_route_keys,
                    state_advanced=state_advanced,
                    previous_formal_count=previous_formal_count,
                    force_all_repairs=(
                        state_advanced or edge_became_faster
                    ),
                    frontier_refresh_reasons=frontier_refresh_reasons,
                    note=event.edge_id,
                    update_start=update_start,
                ),
                retained_routes_before_event,
            )
        if isinstance(event, RequestArrival):
            update_start = time.perf_counter()
            self.current_time = event.event_time
            previous_routes = self.current_candidate_pool()
            previous_decision_state = (
                self.vehicle.node,
                self.vehicle.planning_time,
            )
            self.advance_vehicle(event.event_time)
            state_advanced = previous_decision_state != (
                self.vehicle.node,
                self.vehicle.planning_time,
            )
            self.active_requests[event.request.name] = event.request
            self.request_status[event.request.name] = RequestStatus.WAITING
            self.warm_request_lower_bound_paths(event.request)
            return self.attach_event_context(
                self.incremental_request_arrival(
                    event.event_time,
                    event.request,
                    previous_routes,
                    update_start=update_start,
                    state_advanced=state_advanced,
                ),
                retained_routes_before_event,
            )
        if isinstance(event, RequestCancel):
            update_start = time.perf_counter()
            self.current_time = event.event_time
            previous_routes = self.current_candidate_pool()
            affected_routes = len(
                self.find_routes_serving_request(event.request_name)
            )
            previous_decision_state = (
                self.vehicle.node,
                self.vehicle.planning_time,
            )
            self.advance_vehicle(event.event_time)
            state_advanced = previous_decision_state != (
                self.vehicle.node,
                self.vehicle.planning_time,
            )
            status = self.request_status.get(event.request_name)
            if status == RequestStatus.WAITING:
                self.request_status[event.request_name] = RequestStatus.CANCELLED
                self.active_requests.pop(event.request_name, None)
                note = event.request_name
                cancellation_applied = True
            elif status == RequestStatus.ONBOARD:
                note = f"{event.request_name}:ignored_already_onboard"
                cancellation_applied = False
            else:
                note = f"{event.request_name}:ignored_{status.value.lower() if status else 'unknown'}"
                cancellation_applied = False
            return self.attach_event_context(
                self.incremental_request_cancel(
                    event.event_time,
                    event.request_name,
                    previous_routes,
                    cancellation_applied=cancellation_applied,
                    note=note,
                    update_start=update_start,
                    affected_routes=affected_routes,
                    state_advanced=state_advanced,
                ),
                retained_routes_before_event,
            )
        raise TypeError(f"unsupported event: {event!r}")

    def handle_event_full_recompute(self, event: DynamicEvent) -> DynamicUpdateResult:
        update_start = time.perf_counter()
        self.current_time = event.event_time
        affected_routes = 0

        if isinstance(event, EdgeUpdate):
            affected_routes = len(
                self.find_affected_routes(event.u, event.v, event.event_time)
            )
            self.advance_vehicle(event.event_time)
            self.apply_edge_update(event)
            event_type = "EdgeUpdate"
            note = event.edge_id
        elif isinstance(event, RequestArrival):
            self.advance_vehicle(event.event_time)
            self.active_requests[event.request.name] = event.request
            self.request_status[event.request.name] = RequestStatus.WAITING
            self.warm_request_lower_bound_paths(event.request)
            event_type = "RequestArrival"
            note = event.request.name
        elif isinstance(event, RequestCancel):
            affected_routes = len(
                self.find_routes_serving_request(event.request_name)
            )
            self.advance_vehicle(event.event_time)
            status = self.request_status.get(event.request_name)
            if status == RequestStatus.WAITING:
                self.request_status[event.request_name] = RequestStatus.CANCELLED
                self.active_requests.pop(event.request_name, None)
                note = event.request_name
            elif status == RequestStatus.ONBOARD:
                note = f"{event.request_name}:ignored_already_onboard"
            else:
                note = (
                    f"{event.request_name}:"
                    f"ignored_{status.value.lower() if status else 'unknown'}"
                )
            event_type = "RequestCancel"
        else:
            raise TypeError(f"unsupported event: {event!r}")

        return self.full_recompute(
            event.event_time,
            event_type,
            note=note,
            affected_routes=affected_routes,
            update_mode="FullRecomputeStrategy",
            update_start=update_start,
        )

    def trace_route(
        self,
        candidate: Candidate,
        route_index: int,
        start_time: float,
    ) -> DynamicRouteRecord:
        segments: List[RouteSegment] = []
        stops: List[ScheduledStop] = []
        current_time = float(start_time)
        pickup_times = {name: pickup for name, _, pickup, _ in candidate.pickup_info}
        full_path = list(candidate.full_path)
        path_cursor = 0

        for idx, (key_u, key_v) in enumerate(zip(candidate.key_path, candidate.key_path[1:])):
            path: List[str] = []
            if (
                path_cursor < len(full_path)
                and full_path[path_cursor] == key_u
            ):
                if key_u == key_v:
                    path = [str(key_u)]
                else:
                    try:
                        next_cursor = full_path.index(key_v, path_cursor + 1)
                    except ValueError:
                        next_cursor = -1
                    if next_cursor >= 0:
                        path = full_path[path_cursor : next_cursor + 1]
                        path_cursor = next_cursor
            if not path:
                path, _ = self.cached_dynamic_shortest_path(
                    key_u,
                    key_v,
                    current_time,
                )
            for u, v in zip(path, path[1:]):
                travel_time = self.graph.edge_weight_at(u, v, current_time)
                if math.isinf(travel_time):
                    continue
                depart_time = current_time
                arrive_time = current_time + travel_time
                segments.append(
                    RouteSegment(
                        u=str(u),
                        v=str(v),
                        edge_key=self.graph.edge_key(u, v),
                        depart_time=depart_time,
                        arrive_time=arrive_time,
                        travel_time=travel_time,
                    )
                )
                current_time = arrive_time

            action, request_name = (
                candidate.route_events[idx]
                if idx < len(candidate.route_events)
                else ("target", "")
            )
            if action == "pickup":
                current_time = max(current_time, pickup_times.get(request_name, current_time))
            stops.append(
                ScheduledStop(
                    node=str(key_v),
                    action=action,
                    request_name=request_name,
                    event_time=current_time,
                )
            )
        return DynamicRouteRecord(
            route_index=route_index,
            candidate=candidate,
            segments=segments,
            stops=stops,
        )

    def rebuild_route_records(self, event_time: float) -> None:
        candidates = list(self.routes)
        known = {candidate_identity(candidate) for candidate in candidates}
        for candidate in [*self.route_archive, *self.backup_routes]:
            identity = candidate_identity(candidate)
            if identity in known:
                continue
            known.add(identity)
            candidates.append(candidate)
        self.route_records = [
            self.trace_route(candidate, idx, event_time)
            for idx, candidate in enumerate(candidates)
        ]
        self._route_record_by_identity.clear()
        self._route_keys_by_edge.clear()
        self._route_keys_by_request.clear()
        for record in self.route_records:
            identity = candidate_identity(record.candidate)
            self._route_record_by_identity[identity] = record
            for edge in set(record.edge_keys):
                self._route_keys_by_edge.setdefault(edge, set()).add(identity)
            request_names = set(record.candidate.served_requests)
            request_names.update(
                request_name
                for action, request_name in record.candidate.route_events
                if action in {"pickup", "dropoff"} and request_name
            )
            for request_name in request_names:
                self._route_keys_by_request.setdefault(request_name, set()).add(
                    identity
                )

    def current_candidate_pool(self) -> List[Candidate]:
        # Backup plans are real, constraint-valid service plans.  Carry their
        # distinct stop sequences into the next event instead of discarding
        # and regenerating them after every update.  Suffix repair still
        # revalidates every plan from the current vehicle state.
        candidates: List[Candidate] = []
        known = set()
        for candidate in [
            *(self.route_archive or self.routes),
            *self.backup_routes,
        ]:
            identity = candidate_identity(candidate)
            if identity in known:
                continue
            known.add(identity)
            candidates.append(candidate)
        return candidates

    def find_affected_routes(self, u: str, v: str, event_time: float) -> List[DynamicRouteRecord]:
        edge_key = self.graph.edge_key(u, v)
        self._event_diagnostics.dependency_lookups += 1
        candidate_keys = self._route_keys_by_edge.get(edge_key, set())
        self._event_diagnostics.dependency_candidates += len(candidate_keys)
        affected: List[DynamicRouteRecord] = []
        records = (
            self._route_record_by_identity[key]
            for key in candidate_keys
            if key in self._route_record_by_identity
        )
        for record in records:
            for segment in record.segments:
                if segment.edge_key == edge_key and segment.depart_time >= event_time:
                    affected.append(record)
                    break
        affected.sort(key=lambda record: record.route_index)
        return affected

    def find_routes_serving_request(
        self,
        request_name: str,
    ) -> List[DynamicRouteRecord]:
        self._event_diagnostics.dependency_lookups += 1
        candidate_keys = self._route_keys_by_request.get(str(request_name), set())
        self._event_diagnostics.dependency_candidates += len(candidate_keys)
        records = [
            self._route_record_by_identity[key]
            for key in candidate_keys
            if key in self._route_record_by_identity
        ]
        records.sort(key=lambda record: record.route_index)
        return records

    def _apply_stop(self, stop: ScheduledStop) -> None:
        name = stop.request_name
        if stop.action == "pickup" and self.request_status.get(name) == RequestStatus.WAITING:
            self.request_status[name] = RequestStatus.ONBOARD
            self.vehicle.pickup_times[name] = stop.event_time
        elif stop.action == "dropoff" and self.request_status.get(name) == RequestStatus.ONBOARD:
            self.request_status[name] = RequestStatus.COMPLETED
            self.active_requests.pop(name, None)

    def advance_vehicle(self, event_time: float) -> None:
        """Advance the selected route to the event time without teleporting to source."""
        if event_time < self.vehicle.current_time:
            raise ValueError("dynamic events must be processed in nondecreasing time order")
        if not self.route_records:
            self.vehicle.current_time = event_time
            self.vehicle.planning_time = event_time
            self.vehicle.in_transit = False
            return

        selected = self.route_records[0]
        planning_time = max(event_time, self.vehicle.planning_time)
        decision_node = self.vehicle.node
        in_transit = event_time < self.vehicle.planning_time
        committed_segments: List[RouteSegment] = []

        for segment in selected.segments:
            if segment.arrive_time <= event_time:
                committed_segments.append(segment)
                decision_node = segment.v
                continue
            if segment.depart_time < event_time < segment.arrive_time:
                # The current edge is already committed. Replanning starts at its end.
                committed_segments.append(segment)
                decision_node = segment.v
                planning_time = segment.arrive_time
                in_transit = True
            break

        for segment in committed_segments:
            self.vehicle.traveled_cost += segment.travel_time
        for stop in selected.stops:
            if stop.event_time <= event_time:
                self._apply_stop(stop)

        self.vehicle.node = decision_node
        self.vehicle.current_time = event_time
        self.vehicle.planning_time = planning_time
        self.vehicle.in_transit = in_transit

    def _status_count(self, status: RequestStatus) -> int:
        return sum(1 for value in self.request_status.values() if value == status)

    def expire_waiting_requests(self, planning_time: float) -> None:
        for name, request in list(self.active_requests.items()):
            if self.request_status.get(name) != RequestStatus.WAITING:
                continue
            deadline = request.request_time + request.wait_limit
            if planning_time > deadline:
                self.request_status[name] = RequestStatus.EXPIRED
                self.active_requests.pop(name, None)

    def remaining_route_events(self, candidate: Candidate) -> List[Tuple[str, str]]:
        events: List[Tuple[str, str]] = []
        included_dropoffs = set()
        for action, name in candidate.route_events:
            status = self.request_status.get(name)
            if action == "pickup" and status == RequestStatus.WAITING:
                events.append((action, name))
            elif action == "dropoff" and status in {RequestStatus.WAITING, RequestStatus.ONBOARD}:
                events.append((action, name))
                included_dropoffs.add(name)

        # An alternative route may not have contained a passenger picked up on
        # the selected route. Such passengers are mandatory in every repaired suffix.
        for name, status in self.request_status.items():
            if status == RequestStatus.ONBOARD and name not in included_dropoffs:
                events.append(("dropoff", name))
        return events

    def route_plan_representatives(
        self,
        candidates: Sequence[Candidate],
    ) -> List[Candidate]:
        """Choose one route per remaining stop plan.

        Suffix repair reconstructs physical paths from the current vehicle
        state, so repairing several old physical variants of the same stop
        plan would perform identical work.
        """
        plans: Dict[Tuple[Tuple[str, str], ...], Candidate] = {}
        for candidate in candidates:
            key = tuple(self.remaining_route_events(candidate))
            representative = plans.get(key)
            if (
                representative is None
                or candidate.total_cost < representative.total_cost
            ):
                plans[key] = candidate
        return list(plans.values())

    def has_mixed_onboard_waiting_order(
        self,
        candidates: Sequence[Candidate],
    ) -> bool:
        """Return whether suffixes mix mandatory and optional event orders.

        After the vehicle commits part of a plan, an onboard dropoff can trade
        places with events of a still-waiting request. Retained Top-k plans do
        not certify that every such order was preserved, so insertion alone is
        insufficient for strict plan-level completeness in this case.
        """
        onboard = {
            name
            for name, status in self.request_status.items()
            if status == RequestStatus.ONBOARD
        }
        waiting = {
            name
            for name, status in self.request_status.items()
            if status == RequestStatus.WAITING
        }
        if not onboard or not waiting:
            return False
        for candidate in candidates:
            events = self.remaining_route_events(candidate)
            has_mandatory_dropoff = any(
                action == "dropoff" and name in onboard
                for action, name in events
            )
            has_waiting_event = any(name in waiting for _action, name in events)
            if has_mandatory_dropoff and has_waiting_event:
                return True
        return False

    def _persistent_plan_prefix_dag(
        self,
        candidates: Sequence[Candidate],
    ) -> PersistentPlanDAG:
        plans = {
            tuple(self.remaining_route_events(candidate))
            for candidate in candidates
        }
        signature = tuple(sorted(plans))
        cached = self._persistent_plan_dag
        if cached is not None and cached.signature == signature:
            self._event_diagnostics.plan_dag_reuses += 1
            return cached

        self._plan_nodes_by_edge.clear()
        root = PlanPrefixNode()
        input_transition_count = sum(len(plan) for plan in signature)
        unique_transition_count = 0
        for plan in signature:
            node = root
            for event in plan:
                child = node.children.get(event)
                if child is None:
                    child = PlanPrefixNode()
                    node.children[event] = child
                    unique_transition_count += 1
                node = child
            node.terminal = True
        cached = PersistentPlanDAG(
            signature=signature,
            root=root,
            input_transition_count=input_transition_count,
            unique_transition_count=unique_transition_count,
        )
        self._persistent_plan_dag = cached
        self._event_diagnostics.plan_dag_builds += 1
        return cached

    def _plan_prefix_cache_context(
        self,
        planning_time: float,
    ) -> Tuple[object, ...]:
        request_resources = tuple(
            sorted(
                (
                    name,
                    request.pickup,
                    request.dropoff,
                    request.demand,
                    request.request_time,
                    request.wait_limit,
                    self.request_status.get(name),
                )
                for name, request in self.active_requests.items()
            )
        )
        return (
            self.vehicle.node,
            round(float(planning_time), 9),
            tuple(sorted(self.vehicle.pickup_times.items())),
            request_resources,
        )

    def _unregister_plan_prefix_dependencies(
        self,
        node: PlanPrefixNode,
    ) -> None:
        for edge in node.dependency_edges:
            dependent_nodes = self._plan_nodes_by_edge.get(edge)
            if dependent_nodes is None:
                continue
            dependent_nodes.discard(node)
            if not dependent_nodes:
                self._plan_nodes_by_edge.pop(edge, None)
        node.dependency_edges = frozenset()

    def _cache_plan_prefix_label(
        self,
        node: PlanPrefixNode,
        context: Tuple[object, ...],
        parent: InsertionLabel,
        successor: InsertionLabel,
    ) -> None:
        self._unregister_plan_prefix_dependencies(node)
        appended_path = successor.full_path[len(parent.full_path) :]
        transition_path = (parent.node,) + appended_path
        dependencies = frozenset(
            self.graph.edge_key(left, right)
            for left, right in zip(transition_path, transition_path[1:])
        )
        node.cached_label = successor
        node.cache_context = context
        node.cache_ready = True
        node.invalidated = False
        node.dependency_edges = dependencies
        for edge in dependencies:
            self._plan_nodes_by_edge.setdefault(edge, set()).add(node)

    def _invalidate_plan_prefix_subtree(
        self,
        node: PlanPrefixNode,
        visited: set[PlanPrefixNode],
        context: Tuple[object, ...],
    ) -> None:
        if node in visited:
            return
        visited.add(node)
        if (
            node.cache_ready
            and node.cache_context == context
            and not node.invalidated
        ):
            self._event_diagnostics.plan_prefix_invalidations += 1
            node.invalidated = True
        for child in node.children.values():
            self._invalidate_plan_prefix_subtree(child, visited, context)

    def invalidate_plan_prefixes_for_edge(
        self,
        u: str,
        v: str,
        clear_all: bool = False,
    ) -> None:
        dag = self._persistent_plan_dag
        if dag is None:
            return
        context = self._plan_prefix_cache_context(
            max(self.current_time, self.vehicle.planning_time)
        )
        visited: set[PlanPrefixNode] = set()
        if clear_all:
            self._invalidate_plan_prefix_subtree(dag.root, visited, context)
            return
        edge = self.graph.edge_key(str(u), str(v))
        for node in tuple(self._plan_nodes_by_edge.get(edge, ())):
            self._invalidate_plan_prefix_subtree(node, visited, context)

    def _initial_plan_prefix_label(self, planning_time: float) -> InsertionLabel:
        onboard = {
            name
            for name, status in self.request_status.items()
            if status == RequestStatus.ONBOARD
        }
        load = sum(self.active_requests[name].demand for name in onboard)
        return InsertionLabel(
            base_index=0,
            new_phase=0,
            node=self.vehicle.node,
            arrival_time=planning_time,
            load=load,
            onboard=frozenset(onboard),
            key_path=(self.vehicle.node,),
            full_path=(self.vehicle.node,),
            route_events=(),
            pickup_times=dict(self.vehicle.pickup_times),
            served_names=(),
        )

    def _advance_plan_prefix(
        self,
        label: InsertionLabel,
        event: Tuple[str, str],
    ) -> Optional[InsertionLabel]:
        self._event_diagnostics.plan_prefix_labels_evaluated += 1
        action, request_name = event
        request = self.active_requests.get(request_name)
        if request is None:
            return None
        stop_node = request.pickup if action == "pickup" else request.dropoff
        path, travel_time = self.cached_dynamic_shortest_path(
            label.node,
            stop_node,
            label.arrival_time,
        )
        if not path or math.isinf(travel_time):
            return None

        arrival_time = label.arrival_time + travel_time
        load = label.load
        onboard = set(label.onboard)
        pickup_times = dict(label.pickup_times)
        served_names = list(label.served_names)

        if action == "pickup":
            if (
                self.request_status.get(request_name) != RequestStatus.WAITING
                or request_name in onboard
            ):
                return None
            arrival_time = max(arrival_time, request.request_time)
            if not self.onboard_detour_feasible(
                onboard,
                pickup_times,
                arrival_time,
            ):
                return None
            if arrival_time - request.request_time > request.wait_limit:
                return None
            if load + request.demand > int(
                self.planner_params.get("capacity", 4)
            ):
                return None
            load += request.demand
            onboard.add(request_name)
            pickup_times[request_name] = arrival_time
        elif action == "dropoff":
            if request_name not in onboard:
                return None
            if not self.onboard_detour_feasible(
                onboard,
                pickup_times,
                arrival_time,
            ):
                return None
            load -= request.demand
            onboard.remove(request_name)
            served_names.append(request_name)
        else:
            return None

        return InsertionLabel(
            base_index=0,
            new_phase=0,
            node=stop_node,
            arrival_time=arrival_time,
            load=load,
            onboard=frozenset(onboard),
            key_path=label.key_path + (stop_node,),
            full_path=label.full_path + tuple(path[1:]),
            route_events=label.route_events + (event,),
            pickup_times=pickup_times,
            served_names=tuple(served_names),
        )

    def _complete_plan_prefix(
        self,
        label: InsertionLabel,
        planning_time: float,
    ) -> Optional[Candidate]:
        if label.onboard:
            return None
        path, travel_time = self.cached_dynamic_shortest_path(
            label.node,
            self.target,
            label.arrival_time,
        )
        total_time = label.arrival_time + travel_time
        if (
            not path
            or math.isinf(travel_time)
            or total_time > float(self.planner_params.get("time_budget", math.inf))
        ):
            return None
        full_path = label.full_path + tuple(path[1:])
        pickup_info = []
        for name in label.served_names:
            request = self.active_requests.get(name)
            if request is None:
                continue
            pickup_time = label.pickup_times.get(name, planning_time)
            pickup_info.append(
                (
                    name,
                    request.request_time,
                    pickup_time,
                    pickup_time - request.request_time,
                )
            )
        pickup_info.sort(key=lambda item: item[2])
        served_names = list(label.served_names)
        served_count = len(served_names)
        return Candidate(
            key_path=list(label.key_path + (self.target,)),
            full_path=list(full_path),
            served_requests=served_names,
            pickup_info=pickup_info,
            total_cost=total_time,
            total_time=total_time,
            hops=max(0, len(full_path) - 1),
            served_count=served_count,
            score=served_count * 100000 - total_time,
            route_events=list(label.route_events + (("target", ""),)),
        )

    def repair_candidates_bundled(
        self,
        candidates: Sequence[Candidate],
        planning_time: float,
    ) -> List[Candidate]:
        """Repair exact plan histories while sharing identical event prefixes."""
        if not candidates:
            return []
        dag = self._persistent_plan_prefix_dag(candidates)
        cache_context = self._plan_prefix_cache_context(planning_time)
        self._event_diagnostics.plan_prefix_transitions_saved += max(
            0,
            dag.input_transition_count - dag.unique_transition_count,
        )
        repaired: List[Candidate] = []
        pending: List[Tuple[PlanPrefixNode, InsertionLabel]] = [
            (dag.root, self._initial_plan_prefix_label(planning_time))
        ]
        cursor = 0
        while cursor < len(pending):
            node, label = pending[cursor]
            cursor += 1
            if node.terminal:
                complete = self._complete_plan_prefix(label, planning_time)
                if complete is not None:
                    repaired.append(complete)
            for event, child in sorted(node.children.items()):
                if (
                    child.cache_ready
                    and not child.invalidated
                    and child.cache_context == cache_context
                ):
                    self._event_diagnostics.plan_prefix_cache_hits += 1
                    successor = child.cached_label
                else:
                    if (
                        child.cache_ready
                        and child.invalidated
                        and child.cache_context == cache_context
                    ):
                        self._event_diagnostics.plan_prefix_labels_reopened += 1
                    successor = self._advance_plan_prefix(label, event)
                    if successor is not None:
                        self._cache_plan_prefix_label(
                            child,
                            cache_context,
                            label,
                            successor,
                        )
                    else:
                        self._unregister_plan_prefix_dependencies(child)
                        child.cached_label = None
                        child.cache_context = cache_context
                        child.cache_ready = False
                        child.invalidated = True
                if successor is not None:
                    pending.append((child, successor))
        return repaired

    def onboard_detour_feasible(
        self,
        onboard: Iterable[str],
        pickup_times: Dict[str, float],
        at_time: float,
    ) -> bool:
        """Check the static paper's passenger-level beta constraint."""
        if not bool(self.planner_params.get("enforce_passenger_detour", True)):
            return True
        beta = float(self.planner_params.get("detour_ratio", 1.8))
        tolerance = float(self.planner_params.get("numeric_tolerance", 1e-9))
        for name in onboard:
            request = self.active_requests.get(name)
            pickup_time = pickup_times.get(name)
            if request is None or pickup_time is None:
                return False
            _, direct_cost = self.cached_lower_bound_shortest_path(
                request.pickup,
                request.dropoff,
            )
            if math.isinf(direct_cost):
                return False
            if at_time - pickup_time > beta * direct_cost + tolerance:
                return False
        return True

    def repair_candidate_suffix(
        self,
        graph: TimeDependentGraph,
        candidate: Candidate,
        planning_time: float,
        deviation_leg: Optional[int] = None,
        blocked_edges: Optional[Iterable[EdgeKey]] = None,
        preserve_physical_path: bool = False,
    ) -> Optional[Candidate]:
        route_events = self.remaining_route_events(candidate)
        preserved_paths = (
            self._candidate_leg_paths(candidate)
            if preserve_physical_path
            else {}
        )
        node = self.vehicle.node
        current_time = planning_time
        load = 0
        onboard = set()
        pickup_times = dict(self.vehicle.pickup_times)
        key_path = [node]
        full_path = [node]
        repaired_events: List[Tuple[str, str]] = []
        served_names: List[str] = []

        for name, status in self.request_status.items():
            if status != RequestStatus.ONBOARD:
                continue
            request = self.active_requests.get(name)
            if request is None:
                return None
            onboard.add(name)
            load += request.demand
        if load > int(self.planner_params.get("capacity", 4)):
            return None

        def preserved_leg_path(
            event_key: Tuple[str, str],
            source: str,
            target: str,
        ) -> List[str]:
            old_path = preserved_paths.get(event_key)
            if not old_path or old_path[-1] != str(target):
                return []
            if old_path[0] == str(source):
                return list(old_path)
            # The vehicle may have advanced into the first leg since this
            # concrete backup was produced. Retain only its untraversed suffix.
            try:
                source_index = old_path.index(str(source))
            except ValueError:
                return []
            return list(old_path[source_index:])

        for leg_index, (action, name) in enumerate(route_events):
            request = self.active_requests.get(name)
            if request is None:
                continue
            stop_node = request.pickup if action == "pickup" else request.dropoff
            if deviation_leg == leg_index and blocked_edges:
                path, _ = self.cached_deviation_shortest_path(
                    node,
                    stop_node,
                    blocked_edges,
                )
                travel_time, _ = graph.path_travel_time(path, current_time)
            else:
                path = preserved_leg_path((action, name), node, stop_node)
                if path:
                    travel_time, _ = graph.path_travel_time(path, current_time)
                else:
                    path, travel_time = self.cached_dynamic_shortest_path(
                        node,
                        stop_node,
                        current_time,
                    )
            if not path or math.isinf(travel_time):
                return None
            full_path.extend(path[1:])
            current_time += travel_time

            if action == "pickup":
                if self.request_status.get(name) != RequestStatus.WAITING or name in onboard:
                    return None
                current_time = max(current_time, request.request_time)
                if not self.onboard_detour_feasible(
                    onboard,
                    pickup_times,
                    current_time,
                ):
                    return None
                if current_time - request.request_time > request.wait_limit:
                    return None
                if load + request.demand > int(self.planner_params.get("capacity", 4)):
                    return None
                load += request.demand
                onboard.add(name)
                pickup_times[name] = current_time
            else:
                if name not in onboard:
                    return None
                if not self.onboard_detour_feasible(
                    onboard,
                    pickup_times,
                    current_time,
                ):
                    return None
                load -= request.demand
                onboard.remove(name)
                served_names.append(name)

            node = stop_node
            key_path.append(node)
            repaired_events.append((action, name))

        if onboard:
            return None
        if deviation_leg == len(route_events) and blocked_edges:
            path, _ = self.cached_deviation_shortest_path(
                node,
                self.target,
                blocked_edges,
            )
            travel_time, _ = graph.path_travel_time(path, current_time)
        else:
            path = preserved_leg_path(("target", ""), node, self.target)
            if path:
                travel_time, _ = graph.path_travel_time(path, current_time)
            else:
                path, travel_time = self.cached_dynamic_shortest_path(
                    node,
                    self.target,
                    current_time,
                )
        if not path or math.isinf(travel_time):
            return None
        full_path.extend(path[1:])
        current_time += travel_time
        if current_time > float(self.planner_params.get("time_budget", math.inf)):
            return None
        key_path.append(self.target)
        repaired_events.append(("target", ""))

        pickup_info = []
        for name in served_names:
            request = self.active_requests.get(name)
            if request is None:
                continue
            pickup_time = pickup_times.get(name, planning_time)
            pickup_info.append(
                (name, request.request_time, pickup_time, pickup_time - request.request_time)
            )
        pickup_info.sort(key=lambda item: item[2])
        served_count = len(served_names)
        return Candidate(
            key_path=key_path,
            full_path=full_path,
            served_requests=served_names,
            pickup_info=pickup_info,
            total_cost=current_time,
            total_time=current_time,
            hops=max(0, len(full_path) - 1),
            served_count=served_count,
            score=served_count * 100000 - current_time,
            route_events=repaired_events,
        )

    @staticmethod
    def _candidate_leg_paths(
        candidate: Candidate,
    ) -> Dict[Tuple[str, str], Tuple[str, ...]]:
        """Split a concrete plan into reusable physical paths per stop event."""
        key_path = [str(node) for node in candidate.key_path]
        full_path = [str(node) for node in candidate.full_path]
        paths: Dict[Tuple[str, str], Tuple[str, ...]] = {}
        cursor = 0
        for leg_index, event in enumerate(candidate.route_events):
            if leg_index + 1 >= len(key_path) or cursor >= len(full_path):
                break
            source = key_path[leg_index]
            target = key_path[leg_index + 1]
            if full_path[cursor] != source:
                try:
                    cursor = full_path.index(source, cursor)
                except ValueError:
                    continue
            if source == target:
                paths[event] = (source,)
                continue
            try:
                end = full_path.index(target, cursor + 1)
            except ValueError:
                continue
            paths[event] = tuple(full_path[cursor : end + 1])
            cursor = end
        return paths

    def select_top_k_repaired(self, candidates: Sequence[Candidate]) -> List[Candidate]:
        return select_top_k_candidates(
            candidates,
            int(self.planner_params.get("k", 5)),
            pareto_only=bool(self.planner_params.get("pareto_only", True)),
        )

    def select_route_archive(
        self,
        candidates: Sequence[Candidate],
    ) -> List[Candidate]:
        return select_top_k_candidates(
            candidates,
            self.archive_limit,
            pareto_only=False,
            numeric_tolerance=float(self.planner_params.get("numeric_tolerance", 1e-9)),
        )

    def route_archive_would_saturate(
        self,
        candidates: Sequence[Candidate],
    ) -> bool:
        """Return whether a bounded archive may hide another concrete plan."""
        if not self.active_requests:
            return False
        identities = {
            candidate_identity(candidate)
            for candidate in candidates
        }
        return len(identities) >= self.archive_limit

    @staticmethod
    def _jaccard_distance(left: Iterable[object], right: Iterable[object]) -> float:
        left_set = set(left)
        right_set = set(right)
        union = left_set | right_set
        if not union:
            return 0.0
        return 1.0 - len(left_set & right_set) / len(union)

    @staticmethod
    def _sequence_distance(
        left: Sequence[Tuple[str, str]],
        right: Sequence[Tuple[str, str]],
    ) -> float:
        if not left and not right:
            return 0.0
        previous = [0] * (len(right) + 1)
        for left_item in left:
            current = [0]
            for index, right_item in enumerate(right, start=1):
                if left_item == right_item:
                    current.append(previous[index - 1] + 1)
                else:
                    current.append(max(previous[index], current[-1]))
            previous = current
        return 1.0 - previous[-1] / max(len(left), len(right), 1)

    def route_diversity_distance(
        self,
        left: Candidate,
        right: Candidate,
    ) -> float:
        served_distance = self._jaccard_distance(
            left.served_requests,
            right.served_requests,
        )
        event_distance = self._sequence_distance(
            left.route_events,
            right.route_events,
        )
        left_edges = zip(left.full_path, left.full_path[1:])
        right_edges = zip(right.full_path, right.full_path[1:])
        path_distance = self._jaccard_distance(left_edges, right_edges)
        return 0.5 * served_distance + 0.3 * event_distance + 0.2 * path_distance

    @staticmethod
    def _remaining_cost(candidate: Candidate, planning_time: float) -> float:
        return max(0.0, float(candidate.total_cost) - float(planning_time))

    def backup_cost_regret(
        self,
        candidate: Candidate,
        candidates: Sequence[Candidate],
        planning_time: float,
    ) -> float:
        same_coverage = [
            self._remaining_cost(other, planning_time)
            for other in candidates
            if other.served_count == candidate.served_count
        ]
        if not same_coverage:
            return math.inf
        reference = min(same_coverage)
        candidate_cost = self._remaining_cost(candidate, planning_time)
        tolerance = float(self.planner_params.get("numeric_tolerance", 1e-9))
        if reference <= tolerance:
            return 0.0 if candidate_cost <= reference + tolerance else math.inf
        return max(0.0, (candidate_cost - reference) / reference)

    def select_backup_routes(
        self,
        formal_routes: Sequence[Candidate],
        candidates: Sequence[Candidate],
        planning_time: float,
    ) -> List[Candidate]:
        if not self.enable_backup_routes:
            return []
        k = int(self.planner_params.get("k", 5))
        remaining_slots = max(0, k - len(formal_routes))
        if remaining_slots == 0 or not candidates:
            return []

        formal_ids = {candidate_identity(candidate) for candidate in formal_routes}
        max_served = max(
            (candidate.served_count for candidate in formal_routes),
            default=max(candidate.served_count for candidate in candidates),
        )
        eligible: List[Tuple[Candidate, float]] = []
        for candidate in candidates:
            if candidate_identity(candidate) in formal_ids:
                continue
            if candidate.served_count < max_served - self.backup_coverage_gap:
                continue
            regret = self.backup_cost_regret(candidate, candidates, planning_time)
            if regret <= self.backup_cost_regret_limit + 1e-12:
                eligible.append((candidate, regret))

        selected = list(formal_routes)
        backups: List[Candidate] = []
        while eligible and len(backups) < remaining_slots:
            def choice_key(item: Tuple[Candidate, float]) -> Tuple[object, ...]:
                candidate, regret = item
                minimum_distance = min(
                    (
                        self.route_diversity_distance(candidate, old)
                        for old in selected
                    ),
                    default=1.0,
                )
                return (
                    -minimum_distance,
                    max_served - candidate.served_count,
                    regret,
                    self._remaining_cost(candidate, planning_time),
                    tuple(candidate.route_events),
                    tuple(candidate.full_path),
                )

            eligible.sort(key=choice_key)
            candidate, _ = eligible.pop(0)
            backups.append(candidate)
            selected.append(candidate)
        return backups

    def generate_backup_deviations(
        self,
        candidates: Sequence[Candidate],
        planning_time: float,
    ) -> List[Candidate]:
        if not self.enable_backup_routes or (
            self.backup_deviation_attempts <= 0
            and self.backup_event_order_attempts <= 0
        ):
            return []
        k = int(self.planner_params.get("k", 5))
        pool = list(candidates)
        known = {candidate_identity(candidate) for candidate in pool}
        generated: List[Candidate] = []
        deviation_generated = 0
        attempts = 0
        formal = self.select_top_k_repaired(candidates)

        def register_deviation(
            seed: Candidate,
            leg_index: int,
            blocked: Iterable[EdgeKey],
        ) -> bool:
            nonlocal attempts, deviation_generated
            if attempts >= self.backup_deviation_attempts:
                return False
            attempts += 1
            variant = self.repair_candidate_suffix(
                self.graph,
                seed,
                planning_time,
                deviation_leg=leg_index,
                blocked_edges=blocked,
            )
            if variant is None:
                return False
            identity = candidate_identity(variant)
            if identity in known:
                return False
            known.add(identity)
            generated.append(variant)
            deviation_generated += 1
            pool.append(variant)
            backups = self.select_backup_routes(formal, pool, planning_time)
            return len(formal) + len(backups) >= k

        # A backup may deliberately serve one fewer waiting request.  This is
        # an executable degradation plan, not a fabricated route: both stops
        # are removed and the complete suffix is rebuilt under all constraints.
        for seed in self.route_plan_representatives(pool):
            removable = sorted(
                {
                    name
                    for action, name in seed.route_events
                    if action == "pickup"
                    and self.request_status.get(name) == RequestStatus.WAITING
                }
            )
            for request_name in removable:
                reduced = copy.copy(seed)
                reduced.route_events = [
                    (action, name)
                    for action, name in seed.route_events
                    if name != request_name
                ]
                variant = self.repair_candidate_suffix(
                    self.graph,
                    reduced,
                    planning_time,
                )
                if variant is None:
                    continue
                identity = candidate_identity(variant)
                if identity in known:
                    continue
                known.add(identity)
                generated.append(variant)
                pool.append(variant)

                backups = self.select_backup_routes(
                    formal,
                    pool,
                    planning_time,
                )
                if len(formal) + len(backups) >= k:
                    self.last_backup_new_routes = len(generated)
                    return generated

        # Sparse road networks may offer no useful physical detour even when
        # another feasible pickup/dropoff order is available.  Generate a
        # bounded set of precedence-safe event plans and validate them through
        # the same bundled suffix DP used by the formal dynamic algorithm.
        order_stubs: List[Candidate] = []
        known_orders = {
            tuple(self.remaining_route_events(candidate))
            for candidate in pool
        }
        queued_orders = set(known_orders)
        order_attempts = 0

        def precedence_safe(events: Sequence[Tuple[str, str]]) -> bool:
            pickup_positions = {
                name: index
                for index, (action, name) in enumerate(events)
                if action == "pickup"
            }
            for index, (action, name) in enumerate(events):
                if action != "dropoff":
                    continue
                status = self.request_status.get(name)
                if status == RequestStatus.ONBOARD:
                    continue
                pickup_index = pickup_positions.get(name)
                if pickup_index is None or pickup_index >= index:
                    return False
            return True

        if self.backup_event_order_attempts > 0:
            for seed in self.route_plan_representatives(pool):
                base_events = tuple(self.remaining_route_events(seed))
                if len(base_events) < 2:
                    continue
                for source_index in range(len(base_events)):
                    event = base_events[source_index]
                    for target_index in range(len(base_events)):
                        if target_index == source_index:
                            continue
                        reordered = list(base_events)
                        reordered.pop(source_index)
                        reordered.insert(target_index, event)
                        order = tuple(reordered)
                        if order in queued_orders or not precedence_safe(order):
                            continue
                        queued_orders.add(order)
                        stub = copy.copy(seed)
                        stub.route_events = list(order)
                        order_stubs.append(stub)
                        order_attempts += 1
                        if order_attempts >= self.backup_event_order_attempts:
                            break
                    if order_attempts >= self.backup_event_order_attempts:
                        break
                if order_attempts >= self.backup_event_order_attempts:
                    break

        if order_stubs:
            for variant in self.repair_candidates_bundled(
                order_stubs,
                planning_time,
            ):
                identity = candidate_identity(variant)
                if identity in known:
                    continue
                known.add(identity)
                generated.append(variant)
                pool.append(variant)
            backups = self.select_backup_routes(formal, pool, planning_time)
            if len(formal) + len(backups) >= k:
                self.last_backup_new_routes = len(generated)
                return generated

        deviation_jobs = []
        for seed in self.route_plan_representatives(pool):
            for leg_index, (left, right) in enumerate(
                zip(seed.key_path, seed.key_path[1:])
            ):
                base_path, _ = self.cached_lower_bound_shortest_path(left, right)
                base_edges = list(zip(base_path, base_path[1:]))
                if not base_edges:
                    continue
                middle = len(base_edges) // 2
                edge_order = sorted(
                    range(len(base_edges)),
                    key=lambda index: (abs(index - middle), index),
                )
                deviation_jobs.append(
                    {
                        "seed": seed,
                        "leg_index": leg_index,
                        "left": left,
                        "right": right,
                        "base_path": base_path,
                        "base_edges": base_edges,
                        "edge_order": edge_order,
                    }
                )

        # Probe all route legs round-robin.  Spending the complete budget on
        # the first long leg performs poorly on sparse networks because many
        # of its edges are bridges, while another leg may have a cheap detour.
        independent_budget = max(1, self.backup_deviation_attempts * 2 // 3)
        independent_used = 0
        rank = 0
        while (
            attempts < self.backup_deviation_attempts
            and independent_used < independent_budget
        ):
            progressed = False
            for job in deviation_jobs:
                edge_order = job["edge_order"]
                if rank >= len(edge_order):
                    continue
                progressed = True
                edge_index = edge_order[rank]
                edge = job["base_edges"][edge_index]
                if register_deviation(
                    job["seed"],
                    job["leg_index"],
                    {self.graph.edge_key(*edge)},
                ):
                    self.last_deviation_attempts = attempts
                    self.last_deviation_new_routes = deviation_generated
                    self.last_backup_new_routes = len(generated)
                    return generated
                independent_used += 1
                if (
                    attempts >= self.backup_deviation_attempts
                    or independent_used >= independent_budget
                ):
                    break
            if not progressed:
                break
            rank += 1

        cumulative_states = [
            {
                **job,
                "alternate_path": list(job["base_path"]),
                "blocked": set(),
            }
            for job in deviation_jobs
        ]
        while attempts < self.backup_deviation_attempts:
            progressed = False
            for state in cumulative_states:
                alternate_path = state["alternate_path"]
                if len(alternate_path) <= 1:
                    continue
                progressed = True
                edges = list(zip(alternate_path, alternate_path[1:]))
                middle_edge = edges[len(edges) // 2]
                state["blocked"].add(self.graph.edge_key(*middle_edge))
                if register_deviation(
                    state["seed"],
                    state["leg_index"],
                    state["blocked"],
                ):
                    self.last_deviation_attempts = attempts
                    self.last_deviation_new_routes = deviation_generated
                    self.last_backup_new_routes = len(generated)
                    return generated
                alternate_path, _ = self.cached_deviation_shortest_path(
                    state["left"],
                    state["right"],
                    state["blocked"],
                )
                state["alternate_path"] = alternate_path
                if attempts >= self.backup_deviation_attempts:
                    break
            if not progressed:
                break

        self.last_deviation_attempts = attempts
        self.last_deviation_new_routes = deviation_generated
        self.last_backup_new_routes = len(generated)
        return generated

    def refresh_route_layers(
        self,
        candidates: Sequence[Candidate],
        planning_time: float,
    ) -> List[Candidate]:
        archive = self.select_route_archive(candidates)
        self.route_archive_saturated = self.route_archive_would_saturate(
            candidates
        )
        formal = self.select_top_k_repaired(archive)
        backup_pool = list(archive)
        if self.strategy != UpdateStrategy.FULL_RECOMPUTE:
            # Concrete backup paths are independent executable alternatives.
            # Revalidate and carry them forward before asking for new graph
            # deviations; this avoids regenerating the same physical routes at
            # every event while preserving every hard KMOR constraint.
            known = {candidate_identity(candidate) for candidate in backup_pool}
            for old_backup in self.backup_routes:
                repaired_backup = self.repair_candidate_suffix(
                    self.graph,
                    old_backup,
                    planning_time,
                    preserve_physical_path=True,
                )
                if repaired_backup is None:
                    continue
                identity = candidate_identity(repaired_backup)
                if identity in known:
                    continue
                known.add(identity)
                backup_pool.append(repaired_backup)
        backups = self.select_backup_routes(formal, backup_pool, planning_time)
        k = int(self.planner_params.get("k", 5))
        if len(formal) + len(backups) < k:
            generated = self.generate_backup_deviations(archive, planning_time)
            if generated:
                backup_pool.extend(generated)
                backups = self.select_backup_routes(
                    formal,
                    backup_pool,
                    planning_time,
                )

        self.route_archive = archive
        self.backup_routes = backups
        self.available_routes = [*formal, *backups][:k]
        return formal

    def route_layer_result_fields(self, planning_time: float) -> Dict[str, object]:
        regrets = [
            self.backup_cost_regret(
                candidate,
                [*self.route_archive, *self.backup_routes],
                planning_time,
            )
            for candidate in self.backup_routes
        ]
        k = int(self.planner_params.get("k", 5))
        return {
            "backup_route_count": len(self.backup_routes),
            "available_route_count": len(self.available_routes),
            "available_topk_complete": len(self.available_routes) >= k,
            "mean_backup_cost_regret": (
                sum(regrets) / len(regrets) if regrets else 0.0
            ),
            "max_backup_cost_regret": max(regrets, default=0.0),
        }

    def maintain_top_k_routes(
        self,
        candidates: Sequence[Candidate],
        planning_time: float,
    ) -> List[Candidate]:
        """Fill a shrinking incremental route pool with feasible subroutes."""
        k = int(self.planner_params.get("k", 5))
        self.last_deviation_attempts = 0
        self.last_deviation_new_routes = 0
        self.last_backup_new_routes = 0
        if self.strict_static_semantics:
            return self.refresh_route_layers(candidates, planning_time)

        pool = list(candidates)
        known = {candidate_identity(candidate) for candidate in pool}
        queue = list(pool)
        cursor = 0

        while cursor < len(queue) and len(known) < k:
            base = queue[cursor]
            cursor += 1
            removable = {
                name
                for action, name in base.route_events
                if action == "pickup"
                and self.request_status.get(name) == RequestStatus.WAITING
            }
            for request_name in sorted(removable):
                reduced = copy.copy(base)
                reduced.route_events = [
                    (action, name)
                    for action, name in base.route_events
                    if name != request_name
                ]
                repaired = self.repair_candidate_suffix(
                    self.graph,
                    reduced,
                    planning_time,
                )
                if repaired is None:
                    continue
                identity = candidate_identity(repaired)
                if identity in known:
                    continue
                known.add(identity)
                pool.append(repaired)
                queue.append(repaired)
                if len(known) >= k:
                    break

        deviation_attempts = 0
        max_deviation_attempts = int(
            self.planner_params.get("topk_deviation_attempts", max(12, k * 8))
        )
        stagnation_limit = int(
            self.planner_params.get(
                "topk_stagnation_limit",
                max_deviation_attempts,
            )
        )
        stagnant_attempts = 0
        seed_cursor = 0
        while (
            seed_cursor < len(pool)
            and len(known) < k
            and deviation_attempts < max_deviation_attempts
            and stagnant_attempts < stagnation_limit
        ):
            seed = pool[seed_cursor]
            seed_cursor += 1
            for leg_index, (left, right) in enumerate(
                zip(seed.key_path, seed.key_path[1:])
            ):
                base_path, _ = self.cached_lower_bound_shortest_path(left, right)
                base_edges = list(zip(base_path, base_path[1:]))
                middle = len(base_edges) // 2
                independent_order = sorted(
                    range(len(base_edges)),
                    key=lambda index: (abs(index - middle), index),
                )
                for edge_index in independent_order:
                    deviation_attempts += 1
                    variant = self.repair_candidate_suffix(
                        self.graph,
                        seed,
                        planning_time,
                        deviation_leg=leg_index,
                        blocked_edges={
                            self.graph.edge_key(*base_edges[edge_index])
                        },
                    )
                    if variant is not None:
                        identity = candidate_identity(variant)
                        if identity not in known:
                            known.add(identity)
                            pool.append(variant)
                            self.last_deviation_new_routes += 1
                            stagnant_attempts = 0
                            if len(known) >= k:
                                break
                        else:
                            stagnant_attempts += 1
                    else:
                        stagnant_attempts += 1
                    self.last_deviation_attempts = deviation_attempts
                    if deviation_attempts >= max_deviation_attempts:
                        break
                    if stagnant_attempts >= stagnation_limit:
                        break
                if (
                    len(known) >= k
                    or deviation_attempts >= max_deviation_attempts
                    or stagnant_attempts >= stagnation_limit
                ):
                    break

                alternate_path = base_path
                blocked: set[EdgeKey] = set()
                while (
                    len(alternate_path) > 1
                    and stagnant_attempts < stagnation_limit
                ):
                    edges = list(zip(alternate_path, alternate_path[1:]))
                    middle_edge = edges[len(edges) // 2]
                    blocked.add(self.graph.edge_key(*middle_edge))
                    deviation_attempts += 1
                    variant = self.repair_candidate_suffix(
                        self.graph,
                        seed,
                        planning_time,
                        deviation_leg=leg_index,
                        blocked_edges=blocked,
                    )
                    if variant is not None:
                        identity = candidate_identity(variant)
                        if identity not in known:
                            known.add(identity)
                            pool.append(variant)
                            self.last_deviation_new_routes += 1
                            stagnant_attempts = 0
                            if len(known) >= k:
                                break
                        else:
                            stagnant_attempts += 1
                    else:
                        stagnant_attempts += 1
                    self.last_deviation_attempts = deviation_attempts
                    alternate_path, _ = self.cached_deviation_shortest_path(
                        left,
                        right,
                        blocked,
                    )
                    if deviation_attempts >= max_deviation_attempts:
                        break
                if (
                    len(known) >= k
                    or deviation_attempts >= max_deviation_attempts
                    or stagnant_attempts >= stagnation_limit
                ):
                    break

        selected = self.select_top_k_repaired(pool)
        self.route_archive_saturated = self.route_archive_would_saturate(pool)
        self.route_archive = list(selected)
        self.backup_routes = []
        self.available_routes = list(selected)
        return selected

    def _insertion_transition(
        self,
        graph: TimeDependentGraph,
        label: InsertionLabel,
        action: str,
        request_name: str,
        base_index: int,
        new_phase: int,
    ) -> Optional[InsertionLabel]:
        self._event_diagnostics.insertion_transitions_tested += 1
        request = self.active_requests.get(request_name)
        if request is None:
            return None
        stop_node = request.pickup if action == "pickup" else request.dropoff
        path, travel_time = self.cached_dynamic_shortest_path(
            label.node,
            stop_node,
            label.arrival_time,
        )
        if not path or math.isinf(travel_time):
            return None

        arrival_time = label.arrival_time + travel_time
        load = label.load
        onboard = set(label.onboard)
        pickup_times = dict(label.pickup_times)
        served_names = list(label.served_names)

        if action == "pickup":
            if request_name in onboard:
                return None
            arrival_time = max(arrival_time, request.request_time)
            if not self.onboard_detour_feasible(
                onboard,
                pickup_times,
                arrival_time,
            ):
                return None
            if arrival_time - request.request_time > request.wait_limit:
                return None
            if load + request.demand > int(self.planner_params.get("capacity", 4)):
                return None
            load += request.demand
            onboard.add(request_name)
            pickup_times[request_name] = arrival_time
        else:
            if request_name not in onboard:
                return None
            if not self.onboard_detour_feasible(
                onboard,
                pickup_times,
                arrival_time,
            ):
                return None
            load -= request.demand
            onboard.remove(request_name)
            served_names.append(request_name)

        _, lower_bound = self.cached_dynamic_shortest_path(
            stop_node,
            self.target,
            arrival_time,
        )
        if (
            math.isinf(lower_bound)
            or arrival_time + lower_bound
            > float(self.planner_params.get("time_budget", math.inf))
        ):
            return None

        self._event_diagnostics.insertion_labels_generated += 1
        return InsertionLabel(
            base_index=base_index,
            new_phase=new_phase,
            node=stop_node,
            arrival_time=arrival_time,
            load=load,
            onboard=frozenset(onboard),
            key_path=label.key_path + (stop_node,),
            full_path=label.full_path + tuple(path[1:]),
            route_events=label.route_events + ((action, request_name),),
            pickup_times=pickup_times,
            served_names=tuple(served_names),
        )

    @staticmethod
    def _insertion_signature(
        label: InsertionLabel,
    ) -> Tuple[int, int, str, int, frozenset[str]]:
        return (
            label.base_index,
            label.new_phase,
            label.node,
            label.load,
            label.onboard,
        )

    def _same_insertion_resources(
        self,
        left: InsertionLabel,
        right: InsertionLabel,
    ) -> bool:
        self._event_diagnostics.insertion_resource_checks += 1
        # Nearby times can cross a release or traffic breakpoint. Only
        # identical resources may share a dynamic insertion history pool.
        if left.arrival_time != right.arrival_time:
            return False
        return all(
            left.pickup_times[name] == right.pickup_times[name]
            for name in left.onboard
        )

    @staticmethod
    def _insertion_history_key(label: InsertionLabel) -> Tuple[object, ...]:
        return (
            len(label.full_path) - 1,
            label.route_events,
            label.full_path,
            label.key_path,
        )

    def dp_insert_request(
        self,
        graph: TimeDependentGraph,
        base_candidate: Candidate,
        new_request: Request,
        planning_time: float,
    ) -> Tuple[List[Candidate], int, int]:
        base_events = self.remaining_route_events(base_candidate)
        onboard = {
            name
            for name, status in self.request_status.items()
            if status == RequestStatus.ONBOARD
        }
        load = sum(self.active_requests[name].demand for name in onboard)
        initial = InsertionLabel(
            base_index=0,
            new_phase=0,
            node=self.vehicle.node,
            arrival_time=planning_time,
            load=load,
            onboard=frozenset(onboard),
            key_path=(self.vehicle.node,),
            full_path=(self.vehicle.node,),
            route_events=(),
            pickup_times=dict(self.vehicle.pickup_times),
            served_names=(),
        )

        pending = [initial]
        cursor = 0
        initial_signature = self._insertion_signature(initial)
        best_time: Dict[Tuple[int, int, str, int, frozenset[str]], float] = {
            initial_signature: planning_time
        }
        frontiers: Dict[
            Tuple[int, int, str, int, frozenset[str]],
            List[InsertionLabel],
        ] = {initial_signature: [initial]}
        completed: List[Candidate] = []
        expanded = 0
        dp_pruned = 0
        max_states = int(self.planner_params.get("max_states", 200000))

        def enqueue(label: Optional[InsertionLabel]) -> None:
            nonlocal dp_pruned
            if label is None:
                return
            signature = self._insertion_signature(label)
            if self.strict_static_semantics:
                frontier = frontiers.setdefault(signature, [])
                history_key = self._insertion_history_key(label)
                if any(
                    self._insertion_history_key(existing) == history_key
                    for existing in frontier
                ):
                    dp_pruned += 1
                    return
                # An earlier label can lose its strict cost advantage after a
                # future release-time wait, so retain unequal TD resources.
                survivors = list(frontier)
                equal_resources = [
                    existing
                    for existing in survivors
                    if self._same_insertion_resources(existing, label)
                ]
                if equal_resources:
                    history_limit = int(self.planner_params.get("k", 5))
                    ranked = sorted(
                        equal_resources + [label],
                        key=self._insertion_history_key,
                    )
                    retained_equal = ranked[:history_limit]
                    retained_ids = {id(existing) for existing in retained_equal}
                    survivors = [
                        existing
                        for existing in survivors
                        if existing not in equal_resources or id(existing) in retained_ids
                    ]
                    if id(label) not in retained_ids:
                        frontiers[signature] = survivors
                        dp_pruned += 1
                        return
                    survivors.append(label)
                    frontiers[signature] = survivors
                    pending.append(label)
                    return

                survivors.append(label)
                frontiers[signature] = survivors
                pending.append(label)
                return

            old_time = best_time.get(signature)
            if old_time is not None and old_time <= label.arrival_time:
                dp_pruned += 1
                return
            best_time[signature] = label.arrival_time
            pending.append(label)

        while cursor < len(pending) and expanded < max_states:
            label = pending[cursor]
            cursor += 1
            signature = self._insertion_signature(label)
            if self.strict_static_semantics:
                if not any(
                    existing is label for existing in frontiers.get(signature, [])
                ):
                    continue
            elif label.arrival_time > best_time.get(signature, math.inf):
                dp_pruned += 1
                continue
            expanded += 1

            if label.base_index == len(base_events) and label.new_phase == 2:
                if label.onboard:
                    continue
                path, travel_time = self.cached_dynamic_shortest_path(
                    label.node,
                    self.target,
                    label.arrival_time,
                )
                total_time = label.arrival_time + travel_time
                if not path or math.isinf(travel_time):
                    continue
                if total_time > float(self.planner_params.get("time_budget", math.inf)):
                    continue
                key_path = list(label.key_path + (self.target,))
                full_path = list(label.full_path + tuple(path[1:]))
                route_events = list(label.route_events + (("target", ""),))
                pickup_info = []
                for name in label.served_names:
                    request = self.active_requests.get(name)
                    if request is None:
                        continue
                    pickup_time = label.pickup_times.get(name, planning_time)
                    pickup_info.append(
                        (
                            name,
                            request.request_time,
                            pickup_time,
                            pickup_time - request.request_time,
                        )
                    )
                pickup_info.sort(key=lambda item: item[2])
                served_names = list(label.served_names)
                completed.append(
                    Candidate(
                        key_path=key_path,
                        full_path=full_path,
                        served_requests=served_names,
                        pickup_info=pickup_info,
                        total_cost=total_time,
                        total_time=total_time,
                        hops=max(0, len(full_path) - 1),
                        served_count=len(served_names),
                        score=len(served_names) * 100000 - total_time,
                        route_events=route_events,
                    )
                )
                continue

            if label.base_index < len(base_events):
                action, name = base_events[label.base_index]
                enqueue(
                    self._insertion_transition(
                        graph,
                        label,
                        action,
                        name,
                        label.base_index + 1,
                        label.new_phase,
                    )
                )
            if label.new_phase == 0:
                enqueue(
                    self._insertion_transition(
                        graph,
                        label,
                        "pickup",
                        new_request.name,
                        label.base_index,
                        1,
                    )
                )
            elif label.new_phase == 1:
                enqueue(
                    self._insertion_transition(
                        graph,
                        label,
                        "dropoff",
                        new_request.name,
                        label.base_index,
                        2,
                    )
                )

        self._last_dp_insert_hit_max_states = cursor < len(pending)
        return completed, expanded, dp_pruned

    def dp_insert_request_batch(
        self,
        base_candidates: Sequence[Candidate],
        new_request: Request,
        planning_time: float,
    ) -> Tuple[List[Candidate], int, int, bool]:
        """Insert one request into many plans through one shared prefix DAG.

        Every pickup/dropoff position is enumerated. Equal event sequences are
        collapsed, and common prefixes are evaluated once by
        ``repair_candidates_bundled``. The batch may retain intermediate plans
        that the per-plan DP would strictly dominate, but Pareto Top-k
        selection removes them and therefore preserves the formal answer.
        """
        unique_sequences: Dict[
            Tuple[Tuple[str, str], ...], Candidate
        ] = {}
        pickup = ("pickup", new_request.name)
        dropoff = ("dropoff", new_request.name)
        for base_candidate in self.route_plan_representatives(base_candidates):
            base_events = tuple(self.remaining_route_events(base_candidate))
            if any(name == new_request.name for _, name in base_events):
                continue
            for pickup_position in range(len(base_events) + 1):
                with_pickup = (
                    base_events[:pickup_position]
                    + (pickup,)
                    + base_events[pickup_position:]
                )
                for dropoff_position in range(
                    pickup_position + 1,
                    len(with_pickup) + 1,
                ):
                    sequence = (
                        with_pickup[:dropoff_position]
                        + (dropoff,)
                        + with_pickup[dropoff_position:]
                    )
                    unique_sequences.setdefault(sequence, base_candidate)

        if not unique_sequences:
            return [], 0, 0, False

        if len(unique_sequences) > self.arrival_batch_sequence_limit:
            # Large sequence families benefit more from the resource-label
            # dominance in the original insertion DP than from prefix sharing.
            # This is an exact execution-path switch, not candidate truncation.
            repaired_by_identity: Dict[Tuple[object, ...], Candidate] = {}
            expanded = 0
            dp_pruned = 0
            hit_max_states = False
            for base_candidate in self.route_plan_representatives(base_candidates):
                candidates, route_expanded, route_pruned = self.dp_insert_request(
                    self.graph,
                    base_candidate,
                    new_request,
                    planning_time,
                )
                expanded += route_expanded
                dp_pruned += route_pruned
                hit_max_states = hit_max_states or bool(
                    getattr(self, "_last_dp_insert_hit_max_states", False)
                )
                for candidate in candidates:
                    repaired_by_identity.setdefault(
                        candidate_identity(candidate),
                        candidate,
                    )
            return (
                list(repaired_by_identity.values()),
                expanded,
                dp_pruned,
                hit_max_states,
            )

        selected_sequences = sorted(unique_sequences)
        hit_max_states = False
        stubs: List[Candidate] = []
        for sequence in selected_sequences:
            stub = copy.copy(unique_sequences[sequence])
            stub.route_events = list(sequence)
            stubs.append(stub)

        before_evaluated = self._event_diagnostics.plan_prefix_labels_evaluated
        before_saved = self._event_diagnostics.plan_prefix_transitions_saved
        repaired = self.repair_candidates_bundled(stubs, planning_time)
        expanded = (
            self._event_diagnostics.plan_prefix_labels_evaluated
            - before_evaluated
        )
        dp_pruned = (
            self._event_diagnostics.plan_prefix_transitions_saved
            - before_saved
        )
        self._event_diagnostics.insertion_transitions_tested += sum(
            len(sequence) for sequence in selected_sequences
        )
        self._event_diagnostics.insertion_labels_generated += len(
            selected_sequences
        )

        # The insertion DAG may contain thousands of temporary alternatives;
        # retaining it would make every paired branch clone unnecessarily
        # expensive. Formal repaired plans are preserved in the route archive.
        self._persistent_plan_dag = None
        self._plan_nodes_by_edge.clear()
        return repaired, expanded, dp_pruned, hit_max_states

    def select_local_frontier_requests(
        self,
        candidates: Sequence[Candidate],
        focus_request: Optional[Request] = None,
    ) -> Tuple[List[Request], bool]:
        """Build a deterministic request neighborhood for frontier refresh.

        Onboard requests are mandatory. An optional arriving request is
        followed by requests represented near the front of the current
        archive, then by the most urgent dormant requests. When the active set
        fits inside the configured limit, this is an exact refresh over every
        active request.
        """
        active_names = set(self.active_requests)
        mandatory = {
            name
            for name, status in self.request_status.items()
            if status == RequestStatus.ONBOARD and name in active_names
        }
        focus_name = focus_request.name if focus_request is not None else None
        if focus_name in active_names:
            mandatory.add(focus_name)

        represented_rank: Dict[str, int] = {}
        for route_index, candidate in enumerate(candidates):
            for name in candidate.served_requests:
                if name in active_names:
                    represented_rank.setdefault(name, route_index)

        def priority(name: str) -> Tuple[object, ...]:
            request = self.active_requests[name]
            status = self.request_status.get(name)
            return (
                0 if name == focus_name else 1,
                0 if status == RequestStatus.ONBOARD else 1,
                0 if name in represented_rank else 1,
                represented_rank.get(name, len(candidates) + 1),
                request.request_time + request.wait_limit,
                request.request_time,
                name,
            )

        limit = max(self.arrival_local_reopt_request_limit, len(mandatory))
        if len(active_names) <= limit:
            # Match the full-recompute request order when the local window is
            # exhaustive. The order does not change the exact answer, but it
            # strongly affects incumbent discovery and branch-and-bound cost.
            return (
                [
                    request
                    for name, request in self.active_requests.items()
                    if name in active_names
                ],
                True,
            )

        ordered = sorted(active_names, key=priority)
        selected_names = set(ordered[:limit]) | mandatory
        selected = [
            self.active_requests[name]
            for name in ordered
            if name in selected_names
        ]
        return selected, len(selected_names) == len(active_names)

    def select_arrival_local_requests(
        self,
        new_request: Request,
        candidates: Sequence[Candidate],
    ) -> Tuple[List[Request], bool]:
        """Compatibility wrapper for arrival-focused local refreshes."""
        return self.select_local_frontier_requests(
            candidates,
            focus_request=new_request,
        )

    def local_reoptimize_frontier(
        self,
        candidates: Sequence[Candidate],
        planning_time: float,
        focus_request: Optional[Request] = None,
        neighborhood_candidates: Optional[Sequence[Candidate]] = None,
    ) -> Tuple[List[Candidate], bool, int, int, bool]:
        """Exactly rebuild a bounded event-local request neighborhood."""
        requests, exhaustive = self.select_local_frontier_requests(
            neighborhood_candidates
            if neighborhood_candidates is not None
            else candidates,
            focus_request=focus_request,
        )
        self._event_diagnostics.local_reoptimization_runs += 1
        self._event_diagnostics.local_reoptimization_requests += len(requests)

        onboard_names = [
            request.name
            for request in requests
            if self.request_status.get(request.name) == RequestStatus.ONBOARD
        ]
        request_names = {request.name for request in requests}
        initial_candidates = [
            candidate
            for candidate in candidates
            if set(candidate.served_requests) <= request_names
        ]
        planner_params = dict(self.planner_params)
        planner_params.update(
            k=self.archive_limit,
            pareto_only=False,
            max_states=self.arrival_local_reopt_max_states,
            initial_time=planning_time,
            initial_onboard_names=onboard_names,
            time_dependent_shortest_path=self.cached_dynamic_shortest_path,
            shortest_path_cache=self._lower_bound_shortest_path_cache,
            canonical_undirected_paths=True,
            initial_pickup_times={
                name: self.vehicle.pickup_times.get(name, planning_time)
                for name in onboard_names
            },
            initial_candidates=initial_candidates,
            include_initial_candidates_in_output=False,
        )
        planner = RideSharePlanner(
            self.cached_lower_bound_graph(),
            self.vehicle.node,
            self.target,
            requests,
            **planner_params,
        )
        routes = planner.solve()
        self._event_diagnostics.observe_planner_stats(planner.stats)
        self._last_local_search_certificate = (
            planner.search_certificate.for_scope(exhaustive)
        )
        expanded = planner.stats.expanded_states
        self._event_diagnostics.local_reoptimization_expanded += expanded
        return (
            routes,
            exhaustive,
            expanded,
            planner.stats.dp_pruned_states,
            planner.stats.hit_max_states,
        )

    @staticmethod
    def certificate_result_fields(
        certificate: SearchCertificate,
    ) -> Dict[str, object]:
        return {
            "timed_out": certificate.timed_out,
            "search_exhausted": certificate.search_exhausted,
            "scope_complete": certificate.scope_complete,
            "unresolved_count": certificate.unresolved_count,
            "objective_complete": certificate.objective_complete,
            "concrete_complete": certificate.concrete_complete,
            "completed": certificate.completed,
        }

    def local_reoptimize_arrival(
        self,
        new_request: Request,
        candidates: Sequence[Candidate],
        planning_time: float,
        neighborhood_candidates: Optional[Sequence[Candidate]] = None,
    ) -> Tuple[List[Candidate], bool, int, int, bool]:
        """Compatibility wrapper for arrival-focused local refreshes."""
        return self.local_reoptimize_frontier(
            candidates,
            planning_time,
            focus_request=new_request,
            neighborhood_candidates=neighborhood_candidates,
        )

    def incremental_request_arrival(
        self,
        event_time: float,
        new_request: Request,
        previous_routes: Sequence[Candidate],
        update_start: float,
        state_advanced: bool = False,
    ) -> DynamicUpdateResult:
        planning_time = max(event_time, self.vehicle.planning_time)
        previous_archive_saturated = self.route_archive_saturated
        active_before_expiry = set(self.active_requests)
        self.expire_waiting_requests(planning_time)
        expired_now = active_before_expiry - set(self.active_requests)
        if self.request_status.get(new_request.name) == RequestStatus.EXPIRED:
            previous_routes = list(previous_routes)
        insertion_start = time.perf_counter()

        plan_representatives = self.route_plan_representatives(previous_routes)
        arrival_compatible = (
            self.request_status.get(new_request.name) == RequestStatus.WAITING
            and self.is_request_compatible(new_request, planning_time)
        )
        direct_frontier_refresh = (
            self.enable_arrival_local_reoptimization
            and previous_archive_saturated
        )
        # A saturated archive requires an exact local refresh regardless of
        # insertion outcome. Repair only the formal Top-k as incumbents; the
        # complete old archive is still used to choose the same neighborhood.
        repair_inputs = (
            self.route_plan_representatives(self.routes)
            if direct_frontier_refresh
            else plan_representatives
        )
        unchanged = self.repair_candidates_bundled(
            repair_inputs,
            planning_time,
        )
        inserted: List[Candidate] = []
        expanded = 0
        dp_pruned = 0
        hit_max_states = False
        local_reoptimization_used = False
        local_neighborhood_proven = False
        certified_local_candidates: Optional[List[Candidate]] = None
        local_reoptimization_reasons: List[str] = []

        unchanged_best = max(
            (candidate.served_count for candidate in unchanged),
            default=0,
        )
        if arrival_compatible and not direct_frontier_refresh:
            insertion_bases: List[Candidate] = []
            for old_route in plan_representatives:
                insertion_bases.append(old_route)

                if not self.enable_request_replacement:
                    continue
                removable = sorted(
                    {
                        name
                        for action, name in old_route.route_events
                        if action == "pickup"
                        and name != new_request.name
                        and self.request_status.get(name) == RequestStatus.WAITING
                    },
                    key=lambda name: (
                        self.active_requests[name].request_time
                        + self.active_requests[name].wait_limit,
                        name,
                    ),
                    reverse=True,
                )[: self.replacement_limit_per_route]
                for removed_name in removable:
                    reduced = copy.copy(old_route)
                    reduced.route_events = [
                        (action, name)
                        for action, name in old_route.route_events
                        if name != removed_name
                    ]
                    insertion_bases.append(reduced)

            (
                inserted,
                route_expanded,
                route_pruned,
                route_hit_max_states,
            ) = self.dp_insert_request_batch(
                insertion_bases,
                new_request,
                planning_time,
            )
            expanded += route_expanded
            dp_pruned += route_pruned
            hit_max_states = hit_max_states or route_hit_max_states

            inserted_best = max(
                (candidate.served_count for candidate in inserted),
                default=0,
            )
            if (
                inserted
                and inserted_best <= unchanged_best
                and self.arrival_companion_limit > 0
                and not self.enable_arrival_local_reoptimization
            ):
                # A replacement can fit the arriving request without raising
                # coverage while a second dormant request is needed to reach
                # the best new service set. Explore exactly one additional,
                # archive-bounded insertion layer instead of recomputing all
                # event orders.
                companions = sorted(
                    (
                        request
                        for name, request in self.active_requests.items()
                        if name != new_request.name
                        and self.request_status.get(name) == RequestStatus.WAITING
                    ),
                    key=lambda request: (
                        request.request_time + request.wait_limit,
                        request.name,
                    ),
                )[: self.arrival_companion_limit]
                companion_layer: List[Candidate] = []
                companion_frontier = self.route_plan_representatives(
                    self.select_route_archive(inserted)
                )
                for request in companions:
                    companion_bases: List[Candidate] = []
                    for base_route in companion_frontier:
                        served = set(base_route.served_requests)
                        if request.name in served or not self.is_request_compatible(
                            request,
                            planning_time,
                        ):
                            continue
                        companion_bases.append(base_route)
                    if not companion_bases:
                        continue
                    (
                        candidates,
                        route_expanded,
                        route_pruned,
                        route_hit_max_states,
                    ) = self.dp_insert_request_batch(
                        companion_bases,
                        request,
                        planning_time,
                    )
                    companion_layer.extend(candidates)
                    expanded += route_expanded
                    dp_pruned += route_pruned
                    hit_max_states = hit_max_states or route_hit_max_states
                if companion_layer:
                    inserted.extend(self.select_route_archive(companion_layer))

        local_pool = [*unchanged, *inserted]
        local_best = max(
            (candidate.served_count for candidate in local_pool),
            default=0,
        )
        if arrival_compatible and local_best <= unchanged_best:
            local_reoptimization_reasons.append("coverage_stagnation")
        if previous_archive_saturated:
            local_reoptimization_reasons.append("prior_archive_saturated")
        if self.route_archive_would_saturate(local_pool):
            local_reoptimization_reasons.append("candidate_pool_saturated")

        # Insertion preserves each retained old event order. Once the vehicle
        # has advanced, a mandatory onboard dropoff may need to trade places
        # with a waiting request's events; the old Top-k archive cannot certify
        # those suffix orders. Refresh this mixed state exactly when its
        # executable plan set is still underfilled. Existing feasible backups
        # count toward this certificate; unlike the formal Pareto frontier,
        # they preserve concrete histories that can seed the next insertion.
        output_k = int(self.planner_params.get("k", 5))
        local_formal = self.select_top_k_repaired(local_pool)
        local_backups = self.select_backup_routes(
            local_formal,
            local_pool,
            planning_time,
        )
        local_available_count = len(local_formal) + len(local_backups)
        exact_plan_refresh_required = (
            self.strict_static_semantics
            and arrival_compatible
            and state_advanced
            and len(self.active_requests) > 1
            and local_available_count < output_k
            and self.has_mixed_onboard_waiting_order(local_pool)
        )
        if exact_plan_refresh_required:
            local_reoptimization_reasons.append(
                "mixed_onboard_waiting_order"
            )
        certificate_required = (
            self.enable_certified_recovery
            and (arrival_compatible or state_advanced or bool(expired_now))
        )
        if certificate_required:
            local_reoptimization_reasons.append("certificate_required")

        local_reoptimization_reasons = list(
            dict.fromkeys(local_reoptimization_reasons)
        )

        if (
            self.enable_arrival_local_reoptimization
            and local_reoptimization_reasons
        ):
            local_reoptimization_used = True
            (
                local_candidates,
                local_exhaustive,
                local_expanded,
                local_pruned,
                local_hit_max_states,
            ) = self.local_reoptimize_arrival(
                new_request,
                local_pool,
                planning_time,
                neighborhood_candidates=(
                    previous_routes if direct_frontier_refresh else None
                ),
            )
            inserted.extend(local_candidates)
            expanded += local_expanded
            dp_pruned += local_pruned
            hit_max_states = hit_max_states or local_hit_max_states
            local_neighborhood_proven = (
                self._last_local_search_certificate.completed
            )
            if local_neighborhood_proven:
                certified_local_candidates = list(local_candidates)
                self._current_search_certificate = (
                    self._last_local_search_certificate
                )

        self.routes = self.maintain_top_k_routes(
            (
                certified_local_candidates
                if certified_local_candidates is not None
                else unchanged + inserted
            ),
            planning_time,
        )
        insertion_time = time.perf_counter() - insertion_start
        if not self.routes:
            return self.full_recompute(
                event_time,
                "RequestArrival",
                note=f"{new_request.name}:fallback",
                update_mode="FullRecomputeFallback",
                fallback_recompute=True,
                update_start=update_start,
            )

        self.rebuild_route_records(planning_time)
        best = self.routes[0]
        projected_served = self._status_count(RequestStatus.COMPLETED) + best.served_count
        projected_cost = (
            self.vehicle.traveled_cost + max(0.0, best.total_cost - planning_time)
        )
        selected_insertions = sum(
            1 for candidate in self.routes if new_request.name in candidate.served_requests
        )
        archived_insertions = sum(
            1
            for candidate in self.route_archive
            if new_request.name in candidate.served_requests
        )
        best_remaining_coverage = best.served_count
        local_still_stagnant = (
            local_reoptimization_used
            and best_remaining_coverage <= unchanged_best
            and not local_neighborhood_proven
        )
        if (
            self.adaptive_recompute_on_failed_arrival
            and arrival_compatible
            and (
                (archived_insertions == 0 and not local_neighborhood_proven)
                or local_still_stagnant
                or (
                    exact_plan_refresh_required
                    and not local_neighborhood_proven
                )
            )
        ):
            return self.full_recompute(
                event_time,
                "RequestArrival",
                note=f"{new_request.name}:local_neighborhood_exhausted",
                update_mode="AdaptiveFullRecompute",
                fallback_recompute=True,
                update_start=update_start,
            )
        if certificate_required and not local_neighborhood_proven:
            return self.full_recompute(
                event_time,
                "RequestArrival",
                note=f"{new_request.name}:certificate_recovery",
                update_mode="CertifiedFullRecompute",
                fallback_recompute=True,
                update_start=update_start,
            )
        return DynamicUpdateResult(
            event_type="RequestArrival",
            event_time=event_time,
            active_requests=len(self.active_requests),
            route_count=len(self.routes),
            best_served=projected_served,
            best_cost=projected_cost,
            update_time=time.perf_counter() - update_start,
            recompute_time=insertion_time,
            expanded_states=expanded,
            strategy=self.strategy.value,
            affected_routes=0,
            vehicle_node=self.vehicle.node,
            planning_time=planning_time,
            waiting_requests=self._status_count(RequestStatus.WAITING),
            onboard_requests=self._status_count(RequestStatus.ONBOARD),
            completed_requests=self._status_count(RequestStatus.COMPLETED),
            cancelled_requests=self._status_count(RequestStatus.CANCELLED),
            expired_requests=self._status_count(RequestStatus.EXPIRED),
            in_transit=self.vehicle.in_transit,
            update_mode=(
                "IncrementalLocalReoptimization"
                if local_reoptimization_used
                else "IncrementalDPInsertion"
            ),
            repaired_routes=len(self.routes),
            inserted_routes=selected_insertions,
            dp_pruned_states=dp_pruned,
            hit_max_states=hit_max_states,
            fallback_recompute=False,
            archive_size=len(self.route_archive),
            **self.route_layer_result_fields(planning_time),
            **self.certificate_result_fields(
                self._last_local_search_certificate
                if local_reoptimization_used
                else self._current_search_certificate
            ),
            certificate_stage=(
                "local"
                if local_reoptimization_used and local_neighborhood_proven
                else "inherited"
                if self.enable_certified_recovery
                and self._current_search_certificate.completed
                else "none"
            ),
            note=(
                f"{new_request.name}:frontier_refresh["
                f"{'+'.join(local_reoptimization_reasons)}]"
                if local_reoptimization_used
                else new_request.name
            ),
        )

    def is_request_compatible(self, request: Request, planning_time: float) -> bool:
        if request.demand > int(self.planner_params.get("capacity", 4)):
            return False
        _, base_cost = self.cached_lower_bound_shortest_path(
            self.vehicle.node,
            self.target,
        )
        _, pickup_cost = self.cached_lower_bound_shortest_path(
            self.vehicle.node,
            request.pickup,
        )
        _, direct_cost = self.cached_lower_bound_shortest_path(
            request.pickup,
            request.dropoff,
        )
        _, tail_cost = self.cached_lower_bound_shortest_path(
            request.dropoff,
            self.target,
        )
        if any(math.isinf(value) for value in (base_cost, pickup_cost, direct_cost, tail_cost)):
            return False
        if planning_time + pickup_cost - request.request_time > request.wait_limit:
            return False
        if (
            planning_time + pickup_cost + direct_cost + tail_cost
            > float(self.planner_params.get("time_budget", math.inf))
        ):
            return False
        if not bool(
            self.planner_params.get("enable_compatibility_pruning", False)
        ):
            return True
        if base_cost <= 0:
            return True
        corridor_ratio = (pickup_cost + direct_cost + tail_cost) / base_cost
        threshold = (
            float(self.planner_params.get("detour_ratio", 1.8))
            * float(self.planner_params.get("compatibility_scale", 1.0))
        )
        return corridor_ratio <= threshold

    def incremental_request_cancel(
        self,
        event_time: float,
        request_name: str,
        previous_routes: Sequence[Candidate],
        cancellation_applied: bool,
        note: str,
        update_start: float,
        affected_routes: int = 0,
        state_advanced: bool = False,
    ) -> DynamicUpdateResult:
        planning_time = max(event_time, self.vehicle.planning_time)
        previous_archive_saturated = self.route_archive_saturated
        previous_archive_size = len(self.route_archive or self.routes)
        active_before_expiry = set(self.active_requests)
        self.expire_waiting_requests(planning_time)
        expired_now = active_before_expiry - set(self.active_requests)
        repair_start = time.perf_counter()
        removed_stops = 0
        if cancellation_applied:
            removed_stops = sum(
                1
                for candidate in previous_routes
                for action, name in candidate.route_events
                if name == request_name and action in {"pickup", "dropoff"}
            )

        repaired = self.repair_candidates_bundled(
            self.route_plan_representatives(previous_routes),
            planning_time,
        )
        refill_candidates: List[Candidate] = []
        expanded = 0
        dp_pruned = 0
        hit_max_states = False
        local_reoptimization_used = False
        local_neighborhood_proven = False
        certified_local_candidates: Optional[List[Candidate]] = None
        local_reoptimization_reasons: List[str] = []
        refill_names: set[str] = set()
        if cancellation_applied and self.enable_cancel_refill and repaired:
            waiting = sorted(
                (
                    request
                    for name, request in self.active_requests.items()
                    if self.request_status.get(name) == RequestStatus.WAITING
                    and self.is_request_compatible(request, planning_time)
                ),
                key=lambda request: (
                    request.request_time + request.wait_limit,
                    request.name,
                ),
            )[: self.refill_request_limit]
            attempted_refills: set[
                Tuple[str, Tuple[Tuple[str, str], ...]]
            ] = set()

            def refill_once(
                base_route: Candidate,
                request: Request,
            ) -> Tuple[List[Candidate], int, int, bool]:
                # Every refill layer starts from the same current vehicle
                # resources. Equal remaining stop plans therefore define the
                # same insertion DP even when they came from different
                # cancelled-plan histories.
                remaining = tuple(self.remaining_route_events(base_route))
                cache_key = (request.name, remaining)
                if cache_key in attempted_refills:
                    return [], 0, 0, False
                attempted_refills.add(cache_key)
                candidates, route_expanded, route_pruned = self.dp_insert_request(
                    self.graph,
                    base_route,
                    request,
                    planning_time,
                )
                return (
                    candidates,
                    route_expanded,
                    route_pruned,
                    bool(getattr(self, "_last_dp_insert_hit_max_states", False)),
                )

            frontier = self.route_plan_representatives(repaired)[: self.archive_limit]
            for _ in range(self.cancel_refill_depth):
                layer: List[Candidate] = []
                for base_route in frontier:
                    served = set(base_route.served_requests)
                    for request in waiting:
                        if request.name in served:
                            continue
                        (
                            candidates,
                            route_expanded,
                            route_pruned,
                            route_hit_max_states,
                        ) = refill_once(
                            base_route,
                            request,
                        )
                        if candidates:
                            refill_names.add(request.name)
                            layer.extend(candidates)
                        expanded += route_expanded
                        dp_pruned += route_pruned
                        hit_max_states = hit_max_states or route_hit_max_states
                if not layer:
                    break
                # Bound every depth by the internal archive width. Distinct
                # event histories remain available while combinatorial growth
                # cannot escape the configured dynamic-search budget.
                selected_layer = self.select_route_archive(layer)
                refill_candidates.extend(selected_layer)
                frontier = self.route_plan_representatives(selected_layer)

        if (
            cancellation_applied
            and affected_routes > 0
            and previous_archive_saturated
        ):
            local_reoptimization_reasons.append(
                "cancelled_saturated_archive_request"
            )
        certificate_required = (
            self.enable_certified_recovery
            and (cancellation_applied or state_advanced or bool(expired_now))
        )
        if certificate_required:
            local_reoptimization_reasons.append("certificate_required")
        local_reoptimization_reasons = list(
            dict.fromkeys(local_reoptimization_reasons)
        )
        if (
            self.enable_arrival_local_reoptimization
            and local_reoptimization_reasons
        ):
            local_reoptimization_used = True
            local_pool = [*repaired, *refill_candidates]
            (
                local_candidates,
                local_exhaustive,
                local_expanded,
                local_pruned,
                local_hit_max_states,
            ) = self.local_reoptimize_frontier(
                local_pool,
                planning_time,
            )
            refill_candidates.extend(local_candidates)
            expanded += local_expanded
            dp_pruned += local_pruned
            hit_max_states = hit_max_states or local_hit_max_states
            local_neighborhood_proven = (
                self._last_local_search_certificate.completed
            )
            if local_neighborhood_proven:
                certified_local_candidates = list(local_candidates)
                self._current_search_certificate = (
                    self._last_local_search_certificate
                )

        self.routes = self.maintain_top_k_routes(
            (
                certified_local_candidates
                if certified_local_candidates is not None
                else repaired + refill_candidates
            ),
            planning_time,
        )
        repair_time = time.perf_counter() - repair_start
        if not self.routes:
            return self.full_recompute(
                event_time,
                "RequestCancel",
                note=f"{note}:fallback",
                update_mode="FullRecomputeFallback",
                fallback_recompute=True,
                update_start=update_start,
            )

        retained_threshold = min(
            int(self.planner_params.get("k", 5)),
            max(1, math.ceil(previous_archive_size * self.archive_depletion_ratio)),
        )
        if (
            cancellation_applied
            and self.adaptive_recompute_on_archive_depletion
            and not local_neighborhood_proven
            and previous_archive_size > 0
            and len(self.route_archive) < retained_threshold
        ):
            return self.full_recompute(
                event_time,
                "RequestCancel",
                note=(
                    f"{note}:archive_depleted_"
                    f"{previous_archive_size}_to_{len(self.route_archive)}"
                ),
                update_mode="AdaptiveFullRecompute",
                fallback_recompute=True,
                update_start=update_start,
            )
        if certificate_required and not local_neighborhood_proven:
            return self.full_recompute(
                event_time,
                "RequestCancel",
                note=f"{note}:certificate_recovery",
                update_mode="CertifiedFullRecompute",
                fallback_recompute=True,
                update_start=update_start,
            )

        self.rebuild_route_records(planning_time)
        best = self.routes[0]
        projected_served = self._status_count(RequestStatus.COMPLETED) + best.served_count
        projected_cost = (
            self.vehicle.traveled_cost + max(0.0, best.total_cost - planning_time)
        )
        return DynamicUpdateResult(
            event_type="RequestCancel",
            event_time=event_time,
            active_requests=len(self.active_requests),
            route_count=len(self.routes),
            best_served=projected_served,
            best_cost=projected_cost,
            update_time=time.perf_counter() - update_start,
            recompute_time=repair_time,
            expanded_states=expanded,
            strategy=self.strategy.value,
            affected_routes=affected_routes,
            vehicle_node=self.vehicle.node,
            planning_time=planning_time,
            waiting_requests=self._status_count(RequestStatus.WAITING),
            onboard_requests=self._status_count(RequestStatus.ONBOARD),
            completed_requests=self._status_count(RequestStatus.COMPLETED),
            cancelled_requests=self._status_count(RequestStatus.CANCELLED),
            expired_requests=self._status_count(RequestStatus.EXPIRED),
            in_transit=self.vehicle.in_transit,
            update_mode=(
                "IncrementalCancellationFrontierRefresh"
                if local_reoptimization_used
                else "IncrementalCancellationRefill"
                if any(
                    name in candidate.served_requests
                    for candidate in self.routes
                    for name in refill_names
                )
                else "IncrementalCancellationRepair"
            ),
            repaired_routes=len(self.routes),
            inserted_routes=sum(
                1
                for candidate in self.routes
                if any(name in candidate.served_requests for name in refill_names)
            ),
            removed_stops=removed_stops,
            dp_pruned_states=dp_pruned,
            hit_max_states=hit_max_states,
            fallback_recompute=False,
            archive_size=len(self.route_archive),
            **self.route_layer_result_fields(planning_time),
            **self.certificate_result_fields(
                self._last_local_search_certificate
                if local_reoptimization_used
                else self._current_search_certificate
            ),
            certificate_stage=(
                "local"
                if local_reoptimization_used and local_neighborhood_proven
                else "inherited"
                if self.enable_certified_recovery
                and self._current_search_certificate.completed
                else "none"
            ),
            note=(
                f"{note}:frontier_refresh["
                f"{'+'.join(local_reoptimization_reasons)}]"
                if local_reoptimization_used
                else note
            ),
        )

    def incremental_edge_update(
        self,
        event_time: float,
        previous_routes: Sequence[Candidate],
        affected_routes: int,
        note: str,
        update_start: float,
        affected_route_keys: Optional[set[Tuple[object, ...]]] = None,
        state_advanced: bool = False,
        previous_formal_count: Optional[int] = None,
        force_all_repairs: bool = True,
        frontier_refresh_reasons: Optional[Sequence[str]] = None,
    ) -> DynamicUpdateResult:
        planning_time = max(event_time, self.vehicle.planning_time)
        active_before_expiry = set(self.active_requests)
        self.expire_waiting_requests(planning_time)
        expired_now = active_before_expiry - set(self.active_requests)
        repair_start = time.perf_counter()
        refresh_reasons = list(frontier_refresh_reasons or ())
        if self.enable_certified_recovery:
            refresh_reasons.append("certificate_required")
        previous_plan_count = len(
            self.route_plan_representatives(previous_routes)
        )
        all_retained_routes_affected = (
            bool(self.active_requests)
            and bool(previous_routes)
            and affected_routes >= len(previous_routes)
        )
        if all_retained_routes_affected:
            refresh_reasons.append("all_retained_routes_affected")
        candidates: List[Candidate] = []
        reused_count = 0
        routes_to_repair: List[Candidate] = []
        affected_keys = affected_route_keys or set()
        repair_source = self.routes if refresh_reasons else previous_routes
        for old_route in self.route_plan_representatives(repair_source):
            route_names = {
                name
                for action, name in old_route.route_events
                if action in {"pickup", "dropoff"}
            }
            needs_repair = (
                force_all_repairs
                or candidate_identity(old_route) in affected_keys
                or bool(route_names & expired_now)
            )
            if not needs_repair:
                candidates.append(old_route)
                reused_count += 1
                continue
            routes_to_repair.append(old_route)

        repaired_routes = self.repair_candidates_bundled(
            routes_to_repair,
            planning_time,
        )
        candidates.extend(repaired_routes)
        repaired_count = len(repaired_routes)

        repaired_plan_count = len(self.route_plan_representatives(candidates))
        if (
            self.strict_static_semantics
            and state_advanced
            and repaired_plan_count < previous_plan_count
        ):
            # Committing a route prefix can merge or invalidate retained plan
            # histories. The bounded pre-event archive then no longer proves
            # that every plan-level frontier member is represented.
            refresh_reasons.append("committed_prefix_archive_contraction")

        if previous_formal_count is None:
            previous_formal_count = len(
                self.route_plan_representatives(self.routes)
            )
        repaired_formal_count = len(self.select_top_k_repaired(candidates))
        if (
            self.strict_static_semantics
            and self.active_requests
            and repaired_formal_count < previous_formal_count
        ):
            # A bounded archive is not a completeness certificate after an
            # update removes formal frontier members: plans ranked below the
            # retained archive can become newly non-dominated.
            refresh_reasons.append("formal_frontier_contraction")

        refresh_reasons = list(dict.fromkeys(refresh_reasons))
        frontier_uncertainty = any(
            reason in {
                "all_retained_routes_affected",
                "committed_prefix_archive_contraction",
                "formal_frontier_contraction",
            }
            for reason in refresh_reasons
        )

        local_reoptimization_used = False
        local_neighborhood_proven = False
        expanded = 0
        dp_pruned = 0
        hit_max_states = False
        if self.enable_arrival_local_reoptimization and refresh_reasons:
            local_reoptimization_used = True
            (
                local_candidates,
                local_exhaustive,
                expanded,
                dp_pruned,
                hit_max_states,
            ) = self.local_reoptimize_frontier(
                candidates,
                planning_time,
                neighborhood_candidates=previous_routes,
            )
            candidates.extend(local_candidates)
            local_neighborhood_proven = (
                self._last_local_search_certificate.completed
            )
            if local_neighborhood_proven:
                # A certified exhaustive local solve defines the canonical
                # formal frontier. Repaired historical candidates remain useful
                # as warm starts, but mixing them back into the output can alter
                # objective-equivalent plan-level Top-k tie breaking.
                candidates = list(local_candidates)
                self._current_search_certificate = (
                    self._last_local_search_certificate
                )

        if (
            local_reoptimization_used
            and frontier_uncertainty
            and not local_neighborhood_proven
        ):
            return self.full_recompute(
                event_time,
                "EdgeUpdate",
                note=(
                    f"{note}:uncertain_frontier["
                    f"{'+'.join(refresh_reasons)}]"
                ),
                affected_routes=affected_routes,
                update_mode="AdaptiveFullRecompute",
                fallback_recompute=True,
                update_start=update_start,
            )
        if self.enable_certified_recovery and not local_neighborhood_proven:
            return self.full_recompute(
                event_time,
                "EdgeUpdate",
                note=(
                    f"{note}:certificate_recovery["
                    f"{'+'.join(refresh_reasons)}]"
                ),
                affected_routes=affected_routes,
                update_mode="CertifiedFullRecompute",
                fallback_recompute=True,
                update_start=update_start,
            )

        self.routes = self.maintain_top_k_routes(candidates, planning_time)
        repair_time = time.perf_counter() - repair_start

        if not self.routes:
            return self.full_recompute(
                event_time,
                "EdgeUpdate",
                note=f"{note}:fallback",
                affected_routes=affected_routes,
                update_mode="FullRecomputeFallback",
                fallback_recompute=True,
                update_start=update_start,
            )

        self.rebuild_route_records(planning_time)
        best = self.routes[0]
        projected_served = self._status_count(RequestStatus.COMPLETED) + best.served_count
        projected_cost = (
            self.vehicle.traveled_cost + max(0.0, best.total_cost - planning_time)
        )
        return DynamicUpdateResult(
            event_type="EdgeUpdate",
            event_time=event_time,
            active_requests=len(self.active_requests),
            route_count=len(self.routes),
            best_served=projected_served,
            best_cost=projected_cost,
            update_time=time.perf_counter() - update_start,
            recompute_time=repair_time,
            expanded_states=expanded,
            strategy=self.strategy.value,
            affected_routes=affected_routes,
            vehicle_node=self.vehicle.node,
            planning_time=planning_time,
            waiting_requests=self._status_count(RequestStatus.WAITING),
            onboard_requests=self._status_count(RequestStatus.ONBOARD),
            completed_requests=self._status_count(RequestStatus.COMPLETED),
            cancelled_requests=self._status_count(RequestStatus.CANCELLED),
            expired_requests=self._status_count(RequestStatus.EXPIRED),
            in_transit=self.vehicle.in_transit,
            update_mode=(
                "IncrementalEdgeFrontierRefresh"
                if local_reoptimization_used
                else "IncrementalSuffixRepair"
            ),
            repaired_routes=repaired_count,
            reused_routes=reused_count,
            dp_pruned_states=dp_pruned,
            hit_max_states=hit_max_states,
            fallback_recompute=False,
            archive_size=len(self.route_archive),
            **self.route_layer_result_fields(planning_time),
            **self.certificate_result_fields(
                self._last_local_search_certificate
                if local_reoptimization_used
                else self._current_search_certificate
            ),
            certificate_stage=(
                "local"
                if local_reoptimization_used and local_neighborhood_proven
                else "inherited"
                if self.enable_certified_recovery
                and self._current_search_certificate.completed
                else "none"
            ),
            note=(
                f"{note}:frontier_refresh[{'+'.join(refresh_reasons)}]"
                if local_reoptimization_used
                else note
            ),
        )

    def full_recompute(
        self,
        event_time: float,
        event_type: str,
        note: str = "",
        affected_routes: int = 0,
        update_mode: str = "FullRecompute",
        fallback_recompute: bool = False,
        update_start: Optional[float] = None,
    ) -> DynamicUpdateResult:
        start = update_start if update_start is not None else time.perf_counter()
        if (
            self.strategy == UpdateStrategy.FULL_RECOMPUTE
            and not self.full_recompute_reuse_tdsp_cache
        ):
            # FullRecompute retains only graph-level static preprocessing. In
            # particular, it must not inherit the incremental algorithm's
            # event-dependent TDSP answers or route-prefix dependency DAG.
            self._clear_dynamic_shortest_path_cache()
            self._persistent_plan_dag = None
            self._plan_nodes_by_edge.clear()
        planning_time = max(event_time, self.vehicle.planning_time)
        self.expire_waiting_requests(planning_time)
        lower_bound_graph = self.cached_lower_bound_graph()
        onboard_names = [
            name for name, status in self.request_status.items()
            if status == RequestStatus.ONBOARD
        ]
        planner_params = dict(self.planner_params)
        planner_params.update(
            k=self.archive_limit,
            pareto_only=False,
            initial_time=planning_time,
            initial_onboard_names=onboard_names,
            time_dependent_shortest_path=self.cached_dynamic_shortest_path,
            shortest_path_cache=self._lower_bound_shortest_path_cache,
            canonical_undirected_paths=True,
            initial_pickup_times={
                name: self.vehicle.pickup_times.get(name, planning_time)
                for name in onboard_names
            },
        )
        planner = RideSharePlanner(
            lower_bound_graph,
            self.vehicle.node,
            self.target,
            list(self.active_requests.values()),
            **planner_params,
        )
        recompute_start = time.perf_counter()
        self.routes = self.maintain_top_k_routes(
            planner.solve(),
            planning_time,
        )
        self._event_diagnostics.observe_planner_stats(planner.stats)
        self._current_search_certificate = planner.search_certificate
        recompute_time = time.perf_counter() - recompute_start
        self.rebuild_route_records(planning_time)
        update_time = time.perf_counter() - start

        best = self.routes[0] if self.routes else None
        projected_served = self._status_count(RequestStatus.COMPLETED)
        projected_cost = self.vehicle.traveled_cost
        if best is not None:
            projected_served += best.served_count
            projected_cost += max(0.0, best.total_cost - planning_time)
        return DynamicUpdateResult(
            event_type=event_type,
            event_time=event_time,
            active_requests=len(self.active_requests),
            route_count=len(self.routes),
            best_served=projected_served,
            best_cost=projected_cost if best else math.inf,
            update_time=update_time,
            recompute_time=recompute_time,
            expanded_states=planner.stats.expanded_states,
            peak_open_size=planner.stats.peak_open_size,
            peak_active_dp_labels=planner.stats.peak_active_dp_labels,
            dag_nodes_created=planner.stats.dag_nodes_created,
            strategy=self.strategy.value,
            affected_routes=affected_routes,
            vehicle_node=self.vehicle.node,
            planning_time=planning_time,
            waiting_requests=self._status_count(RequestStatus.WAITING),
            onboard_requests=self._status_count(RequestStatus.ONBOARD),
            completed_requests=self._status_count(RequestStatus.COMPLETED),
            cancelled_requests=self._status_count(RequestStatus.CANCELLED),
            expired_requests=self._status_count(RequestStatus.EXPIRED),
            in_transit=self.vehicle.in_transit,
            update_mode=update_mode,
            repaired_routes=0,
            inserted_routes=0,
            dp_pruned_states=planner.stats.dp_pruned_states,
            hit_max_states=planner.stats.hit_max_states,
            fallback_recompute=fallback_recompute,
            archive_size=len(self.route_archive),
            **self.route_layer_result_fields(planning_time),
            **self.certificate_result_fields(planner.search_certificate),
            certificate_stage="full",
            adaptive_recompute=update_mode == "AdaptiveFullRecompute",
            note=note,
        )


def run_paired_event(
    base_manager: DynamicKMORManager,
    event: DynamicEvent,
    commit_incremental: bool = False,
    online_strategy: Optional[Union[UpdateStrategy, str]] = None,
    execution_order: str = "incremental_first",
) -> PairedEventOutcome:
    """Evaluate one event from identical pre-event states.

    ``commit_incremental=True`` uses the input manager as the online branch and
    therefore needs only one full-graph clone for the oracle.  The default
    preserves the earlier non-mutating API for callers that need both copies.
    """
    strategy = online_strategy or UpdateStrategy.INCREMENTAL
    strategy = (
        strategy
        if isinstance(strategy, UpdateStrategy)
        else UpdateStrategy(str(strategy))
    )
    if strategy == UpdateStrategy.FULL_RECOMPUTE:
        raise ValueError("paired online strategy must not be full_recompute")
    if execution_order not in {"incremental_first", "full_first"}:
        raise ValueError(
            "execution_order must be incremental_first or full_first"
        )

    clone_start = time.perf_counter()
    full = base_manager.clone(UpdateStrategy.FULL_RECOMPUTE)
    if commit_incremental:
        incremental = base_manager
        incremental.strategy = strategy
    else:
        incremental = base_manager.clone(strategy)
    branch_clone_time = time.perf_counter() - clone_start
    if execution_order == "full_first":
        full_result = full.handle_event(event)
        incremental_result = incremental.handle_event(event)
    else:
        incremental_result = incremental.handle_event(event)
        full_result = full.handle_event(event)
    return PairedEventOutcome(
        incremental_manager=incremental,
        full_manager=full,
        incremental_result=incremental_result,
        full_result=full_result,
        branch_clone_time=branch_clone_time,
        execution_order=execution_order,
    )


def parse_event_file(path: Path) -> List[DynamicEvent]:
    events: List[DynamicEvent] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(row for row in f if row.strip() and not row.lstrip().startswith("#"))
        for row in reader:
            kind = row.get("type", "").strip()
            if kind == "EdgeUpdate":
                events.append(
                    EdgeUpdate(
                        edge_id=row.get("edge_id", ""),
                        u=str(row["u"]),
                        v=str(row["v"]),
                        event_time=float(row["event_time"]),
                        new_weight=float(row["new_weight"]),
                    )
                )
            elif kind == "RequestArrival":
                events.append(
                    RequestArrival(
                        event_time=float(row["event_time"]),
                        request=Request(
                            name=str(row["name"]),
                            pickup=str(row["pickup"]),
                            dropoff=str(row["dropoff"]),
                            demand=int(row.get("demand", 1)),
                            request_time=float(row.get("request_time", row["event_time"])),
                            wait_limit=float(row.get("wait_limit", math.inf)),
                        ),
                    )
                )
            elif kind == "RequestCancel":
                events.append(RequestCancel(request_name=str(row["name"]), event_time=float(row["event_time"])))
    events.sort(key=lambda item: item.event_time)
    return events


def write_results(path: Path, rows: Sequence[DynamicUpdateResult]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].__dict__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run event-driven dynamic KMOR")
    ap.add_argument("--input", required=True, help="Edge list file")
    ap.add_argument("--requests-file", required=True, help="Initial requests")
    ap.add_argument("--events-file", required=True, help="CSV event stream")
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--output", default="paper_results/dynamic_kmor_results.csv")
    ap.add_argument("--ignore-first-col", action="store_true")
    ap.add_argument("--directed", action="store_true")
    ap.add_argument("--bucket-size", type=float, default=300.0)
    ap.add_argument("--capacity", type=int, default=4)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--time-budget", type=float, default=math.inf)
    ap.add_argument("--detour-ratio", type=float, default=1.4)
    ap.add_argument("--max-states", type=int, default=30000)
    ap.add_argument(
        "--legacy-dynamic-semantics",
        action="store_true",
        help=(
            "Use the earlier dynamic Top-k filling behavior without the "
            "passenger-level beta constraint"
        ),
    )
    ap.add_argument(
        "--strategy",
        choices=[item.value for item in UpdateStrategy],
        default=UpdateStrategy.INCREMENTAL.value,
    )
    return ap


def main() -> int:
    args = build_parser().parse_args()
    static_graph = read_edge_list(args.input, directed=args.directed, ignore_first_col=args.ignore_first_col)
    td_graph = TimeDependentGraph.from_static_graph(static_graph, bucket_size=args.bucket_size)
    requests = parse_requests_file(args.requests_file)
    events = parse_event_file(Path(args.events_file))

    manager = DynamicKMORManager(
        td_graph,
        args.source,
        args.target,
        requests,
        capacity=args.capacity,
        k=args.k,
        time_budget=args.time_budget,
        detour_ratio=args.detour_ratio,
        max_states=args.max_states,
        strategy=args.strategy,
        strict_static_semantics=not args.legacy_dynamic_semantics,
        enable_certified_recovery=not args.legacy_dynamic_semantics,
    )

    rows = [manager.initialize(0.0)]
    for event in events:
        rows.append(manager.handle_event(event))
    write_results(Path(args.output), rows)
    print(f"Saved dynamic KMOR results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
