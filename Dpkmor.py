#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DPKMOR:
在真实路网上做乘客拼车实验，支持：
1. 乘客 pickup/dropoff 顺序约束
2. 车辆容量约束
3. 乘客等待时间约束（request_time 之后 wait_limit 内必须接到）
4. 乘客上车后的 beta 绕行约束
5. 最多 k 条互不支配方案输出
6. 基于插入 DP 的候选评估 + Pareto 安全剪枝的搜索
"""

from __future__ import annotations

import argparse
import heapq
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple


BUNDLED_IMPLEMENTATION_VERSION = "plan-dag-v1"


class Graph:
    def __init__(self, directed: bool = False):
        self.directed = directed
        self.adj: Dict[str, List[Tuple[str, float]]] = {}

    def add_edge(self, u: str, v: str, w: float = 1.0) -> None:
        self.adj.setdefault(u, []).append((v, float(w)))
        self.adj.setdefault(v, [])
        if not self.directed:
            self.adj.setdefault(v, []).append((u, float(w)))

    def neighbors(self, u: str) -> Iterable[Tuple[str, float]]:
        return self.adj.get(u, [])

    def nodes(self) -> Iterable[str]:
        return self.adj.keys()

    def shortest_path(self, source: str, target: str) -> Tuple[List[str], float]:
        if not self.directed:
            return self._bidirectional_shortest_path(str(source), str(target))
        return self.shortest_paths_from(source, [target])[str(target)]

    def shortest_path_avoiding_edges(
        self,
        source: str,
        target: str,
        blocked_edges: Iterable[Tuple[str, str]],
    ) -> Tuple[List[str], float]:
        source = str(source)
        target = str(target)
        blocked = {
            (str(u), str(v))
            if self.directed
            else tuple(sorted((str(u), str(v))))
            for u, v in blocked_edges
        }
        if source == target:
            return [source], 0.0
        if not self.directed:
            return self._bidirectional_shortest_path(
                source,
                target,
                blocked_edges=blocked,
            )

        queue: List[Tuple[float, str]] = [(0.0, source)]
        distances: Dict[str, float] = {source: 0.0}
        previous: Dict[str, Optional[str]] = {source: None}
        settled = set()
        while queue:
            distance, node = heapq.heappop(queue)
            if node in settled:
                continue
            settled.add(node)
            if node == target:
                break
            for nxt, weight in self.neighbors(node):
                edge_key = (
                    (node, nxt)
                    if self.directed
                    else tuple(sorted((node, nxt)))
                )
                if edge_key in blocked:
                    continue
                next_distance = distance + weight
                if next_distance < distances.get(nxt, math.inf):
                    distances[nxt] = next_distance
                    previous[nxt] = node
                    heapq.heappush(queue, (next_distance, nxt))

        if target not in distances:
            return [], math.inf
        path = []
        node: Optional[str] = target
        while node is not None:
            path.append(node)
            node = previous.get(node)
        path.reverse()
        return path, distances[target]

    def _bidirectional_shortest_path(
        self,
        source: str,
        target: str,
        blocked_edges: Iterable[Tuple[str, str]] = (),
    ) -> Tuple[List[str], float]:
        """Exact bidirectional Dijkstra for non-negative undirected graphs."""
        if source == target:
            return [source], 0.0
        if source not in self.adj or target not in self.adj:
            return [], math.inf
        blocked = {
            tuple(sorted((str(u), str(v))))
            for u, v in blocked_edges
        }

        forward_queue: List[Tuple[float, str]] = [(0.0, source)]
        backward_queue: List[Tuple[float, str]] = [(0.0, target)]
        forward_dist: Dict[str, float] = {source: 0.0}
        backward_dist: Dict[str, float] = {target: 0.0}
        forward_prev: Dict[str, Optional[str]] = {source: None}
        backward_next: Dict[str, Optional[str]] = {target: None}
        forward_settled = set()
        backward_settled = set()
        best_distance = math.inf
        meeting_node: Optional[str] = None

        while forward_queue and backward_queue:
            if forward_queue[0][0] + backward_queue[0][0] >= best_distance:
                break

            expand_forward = forward_queue[0][0] <= backward_queue[0][0]
            if expand_forward:
                curr_dist, node = heapq.heappop(forward_queue)
                if node in forward_settled or curr_dist != forward_dist.get(node):
                    continue
                forward_settled.add(node)
                if node in backward_dist:
                    candidate = curr_dist + backward_dist[node]
                    if candidate < best_distance:
                        best_distance = candidate
                        meeting_node = node
                for nxt, weight in self.neighbors(node):
                    if tuple(sorted((node, nxt))) in blocked:
                        continue
                    next_dist = curr_dist + weight
                    if next_dist < forward_dist.get(nxt, math.inf):
                        forward_dist[nxt] = next_dist
                        forward_prev[nxt] = node
                        heapq.heappush(forward_queue, (next_dist, nxt))
                    if nxt in backward_dist:
                        candidate = forward_dist[nxt] + backward_dist[nxt]
                        if candidate < best_distance:
                            best_distance = candidate
                            meeting_node = nxt
            else:
                curr_dist, node = heapq.heappop(backward_queue)
                if node in backward_settled or curr_dist != backward_dist.get(node):
                    continue
                backward_settled.add(node)
                if node in forward_dist:
                    candidate = curr_dist + forward_dist[node]
                    if candidate < best_distance:
                        best_distance = candidate
                        meeting_node = node
                for nxt, weight in self.neighbors(node):
                    if tuple(sorted((node, nxt))) in blocked:
                        continue
                    next_dist = curr_dist + weight
                    if next_dist < backward_dist.get(nxt, math.inf):
                        backward_dist[nxt] = next_dist
                        backward_next[nxt] = node
                        heapq.heappush(backward_queue, (next_dist, nxt))
                    if nxt in forward_dist:
                        candidate = backward_dist[nxt] + forward_dist[nxt]
                        if candidate < best_distance:
                            best_distance = candidate
                            meeting_node = nxt

        if meeting_node is None:
            return [], math.inf

        prefix: List[str] = []
        node: Optional[str] = meeting_node
        while node is not None:
            prefix.append(node)
            node = forward_prev.get(node)
        prefix.reverse()

        suffix: List[str] = []
        node = backward_next.get(meeting_node)
        while node is not None:
            suffix.append(node)
            node = backward_next.get(node)
        return prefix + suffix, best_distance

    def shortest_paths_from(
        self,
        source: str,
        targets: Iterable[str],
    ) -> Dict[str, Tuple[List[str], float]]:
        source = str(source)
        target_set = {str(target) for target in targets}
        if not target_set:
            return {}

        pq: List[Tuple[float, str]] = [(0.0, source)]
        dist: Dict[str, float] = {source: 0.0}
        prev: Dict[str, Optional[str]] = {source: None}
        visited = set()
        remaining = set(target_set)

        while pq and remaining:
            curr_dist, node = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            remaining.discard(node)

            for nxt, weight in self.neighbors(node):
                nd = curr_dist + weight
                if nd < dist.get(nxt, math.inf):
                    dist[nxt] = nd
                    prev[nxt] = node
                    heapq.heappush(pq, (nd, nxt))

        results: Dict[str, Tuple[List[str], float]] = {}
        for target in target_set:
            if target not in dist:
                results[target] = ([], math.inf)
                continue
            path = []
            cur: Optional[str] = target
            while cur is not None:
                path.append(cur)
                cur = prev.get(cur)
            path.reverse()
            results[target] = (path, dist[target])
        return results


def read_edge_list(
    path: str,
    delimiter: Optional[str] = None,
    directed: bool = False,
    comment: str = "#",
    ignore_first_col: bool = False,
) -> Graph:
    graph = Graph(directed=directed)
    sep = None
    if delimiter:
        if delimiter == "space":
            sep = None
        elif delimiter == "tab":
            sep = "\t"
        elif delimiter == "comma":
            sep = ","
        else:
            sep = delimiter

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(comment):
                continue
            parts = s.split(sep) if sep is not None else s.split()
            if ignore_first_col:
                if len(parts) < 4:
                    continue
                u, v, w_str = parts[1], parts[2], parts[3]
            else:
                if len(parts) < 2:
                    continue
                u = parts[0]
                v = parts[1]
                w_str = parts[2] if len(parts) >= 3 else "1.0"
            try:
                w = float(w_str)
            except ValueError:
                w = 1.0
            graph.add_edge(str(u), str(v), w)
    return graph


@dataclass(frozen=True)
class Request:
    name: str
    pickup: str
    dropoff: str
    demand: int = 1
    request_time: float = 0.0
    wait_limit: float = math.inf


@dataclass
class Candidate:
    key_path: List[str]
    full_path: List[str]
    served_requests: List[str]
    pickup_info: List[Tuple[str, float, float, float]]
    total_cost: float
    total_time: float
    hops: int
    served_count: int
    score: float
    route_events: List[Tuple[str, str]] = field(default_factory=list)


def candidate_identity(
    candidate: Candidate,
) -> Tuple[
    Tuple[Tuple[str, str], ...],
    Tuple[str, ...],
    Tuple[str, ...],
    Tuple[str, ...],
]:
    """Identity used to keep genuinely different route recommendations."""
    return (
        tuple(candidate.route_events),
        tuple(candidate.key_path),
        tuple(sorted(candidate.served_requests)),
        tuple(candidate.full_path),
    )


def select_top_k_candidates(
    candidates: Iterable[Candidate],
    k: int,
    target_served_count: Optional[int] = None,
    pareto_only: bool = False,
    numeric_tolerance: float = 1e-9,
) -> List[Candidate]:
    """Return up to k distinct routes.

    Static Pareto KMOR sets ``pareto_only=True`` and may therefore return fewer
    than k routes.  The compatibility mode remains available for the dynamic
    route-diversity experiments, which historically filled all k slots.
    """
    values = list(candidates)
    if target_served_count is not None:
        filtered = [
            candidate
            for candidate in values
            if candidate.served_count == target_served_count
        ]
        if filtered:
            values = filtered

    unique: Dict[
        Tuple[
            Tuple[Tuple[str, str], ...],
            Tuple[str, ...],
            Tuple[str, ...],
            Tuple[str, ...],
        ],
        Candidate,
    ] = {}
    for candidate in values:
        key = candidate_identity(candidate)
        old = unique.get(key)
        if old is None or candidate.total_cost < old.total_cost:
            unique[key] = candidate

    def rank(item: Candidate) -> Tuple[Any, ...]:
        cost_rank: Any
        if numeric_tolerance > 0:
            cost_rank = round(item.total_cost / numeric_tolerance)
        else:
            cost_rank = item.total_cost
        return (
            -item.served_count,
            cost_rank,
            item.hops,
            tuple(item.route_events),
            tuple(item.key_path),
        )

    ranked = sorted(unique.values(), key=rank)
    pareto: List[Candidate] = []
    best_cost_at_greater_coverage = math.inf
    start = 0
    while start < len(ranked):
        served_count = ranked[start].served_count
        end = start + 1
        while end < len(ranked) and ranked[end].served_count == served_count:
            end += 1
        group = ranked[start:end]
        group_min_cost = min(item.total_cost for item in group)
        for candidate in group:
            lower_cost_in_same_group = (
                candidate.total_cost
                > group_min_cost + numeric_tolerance
            )
            dominated_by_greater_coverage = (
                best_cost_at_greater_coverage
                <= candidate.total_cost + numeric_tolerance
            )
            if not lower_cost_in_same_group and not dominated_by_greater_coverage:
                pareto.append(candidate)
        best_cost_at_greater_coverage = min(
            best_cost_at_greater_coverage,
            group_min_cost,
        )
        start = end

    selected = pareto[: max(0, int(k))]
    if pareto_only:
        selected.sort(key=rank)
        return selected

    selected_ids = {candidate_identity(candidate) for candidate in selected}
    if len(selected) < k:
        for candidate in ranked:
            identity = candidate_identity(candidate)
            if identity in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(identity)
            if len(selected) >= k:
                break
    selected.sort(key=rank)
    return selected


@dataclass
class PlannerStats:
    generated_states: int = 0
    popped_states: int = 0
    expanded_states: int = 0
    feasibility_checks: int = 0
    dp_pruned_states: int = 0
    feasibility_pruned_states: int = 0
    heuristic_pruned_states: int = 0
    # Kept for compatibility with the newer experiment report schema.
    state_bound_pruned_states: int = 0
    pareto_bound_pruned_states: int = 0
    compatibility_pruned_requests: int = 0
    # Insertion-DP accounting. These counters measure evaluation work, not
    # search-space pruning, so the ablation has a clean interpretation.
    dp_insertion_evaluations: int = 0
    dp_insertion_position_checks: int = 0
    dp_insertion_cache_hits: int = 0
    naive_insertion_evaluations: int = 0
    naive_insertion_position_checks: int = 0
    complete_candidates: int = 0
    peak_open_size: int = 0
    active_dp_labels: int = 0
    peak_active_dp_labels: int = 0
    dag_nodes_created: int = 0
    dag_edges_created: int = 0
    hit_max_states: bool = False
    timed_out: bool = False


@dataclass(frozen=True)
class SearchCertificate:
    """Auditable completeness result for one route-search invocation.

    Objective completeness permits an unresolved state's optimistic objective
    to equal an existing frontier point.  Concrete-plan completeness does not:
    an objective-equivalent state may still produce a distinct executable plan.
    """

    search_exhausted: bool = False
    hit_state_limit: bool = False
    timed_out: bool = False
    scope_complete: bool = True
    unresolved_count: int = 0
    objective_complete: bool = False
    concrete_complete: bool = False

    @property
    def satisfied(self) -> bool:
        return self.objective_complete and self.concrete_complete

    @property
    def completed(self) -> bool:
        return (
            self.satisfied
            and not self.hit_state_limit
            and not self.timed_out
        )

    def for_scope(self, scope_complete: bool) -> "SearchCertificate":
        scope_complete = bool(scope_complete)
        return SearchCertificate(
            search_exhausted=self.search_exhausted,
            hit_state_limit=self.hit_state_limit,
            timed_out=self.timed_out,
            scope_complete=scope_complete,
            unresolved_count=self.unresolved_count,
            objective_complete=(
                scope_complete and self.objective_complete
            ),
            concrete_complete=(
                scope_complete and self.concrete_complete
            ),
        )


@dataclass(frozen=True)
class SearchState:
    node: str
    picked_mask: int
    served_mask: int
    load: int
    time_used: float
    key_path: Tuple[str, ...]
    pickup_times: Tuple[float, ...]
    route_events: Tuple[Tuple[str, str], ...] = ()


@dataclass
class RouteHistory:
    """Compact persistent route history used by dynamic label search.

    Prefix tuples are materialized lazily. During search, a transition only
    creates a parent pointer and a compact route id instead of copying the
    complete event/path tuples.
    """

    parent: Optional["RouteHistory"]
    node: str
    event: Tuple[str, str]
    request_idx: Optional[int]
    pickup_time: Optional[float]
    prefix_hops: int
    route_id: int
    base_key_path: Tuple[str, ...] = ()
    base_pickup_times: Tuple[float, ...] = ()
    base_route_events: Tuple[Tuple[str, str], ...] = ()
    _key_path_cache: Optional[Tuple[str, ...]] = field(default=None, init=False, repr=False)
    _pickup_times_cache: Optional[Tuple[float, ...]] = field(default=None, init=False, repr=False)
    _route_events_cache: Optional[Tuple[Tuple[str, str], ...]] = field(default=None, init=False, repr=False)

    @property
    def key_path(self) -> Tuple[str, ...]:
        if self._key_path_cache is None:
            if self.parent is None:
                self._key_path_cache = self.base_key_path
            else:
                self._key_path_cache = self.parent.key_path + (self.node,)
        return self._key_path_cache

    @property
    def pickup_times(self) -> Tuple[float, ...]:
        if self._pickup_times_cache is None:
            if self.parent is None:
                self._pickup_times_cache = self.base_pickup_times
            elif self.request_idx is None:
                self._pickup_times_cache = self.parent.pickup_times
            else:
                values = list(self.parent.pickup_times)
                values[self.request_idx] = self.pickup_time
                self._pickup_times_cache = tuple(values)
        return self._pickup_times_cache

    @property
    def route_events(self) -> Tuple[Tuple[str, str], ...]:
        if self._route_events_cache is None:
            if self.parent is None:
                self._route_events_cache = self.base_route_events
            else:
                self._route_events_cache = self.parent.route_events + (self.event,)
        return self._route_events_cache


@dataclass(frozen=True)
class DPTransition:
    successor: SearchState
    request_idx: Optional[int]
    segment_hops: int


@dataclass(slots=True)
class DPNode:
    state: SearchState
    histories: Dict[int, None]
    priority: float
    representative_history_id: int = 0
    propagated_history_keys: Set[int] = field(default_factory=set)
    transitions: Optional[Tuple[DPTransition, ...]] = None
    incoming: List[Tuple["DPNode", DPTransition]] = field(default_factory=list)
    incoming_keys: Set[Tuple[int, Tuple[str, str]]] = field(default_factory=set)
    queued: bool = False
    active: bool = True


@dataclass(eq=False, slots=True)
class FastDPLabel:
    """Resource-safe static DP label with a compact parent pointer."""

    state: SearchState
    parent: Optional["FastDPLabel"]
    event: Optional[Tuple[str, str]]
    depth: int
    onboard_pickup_times: Tuple[float, ...]
    optimistic_served: int
    lower_cost: float
    priority: float
    active: bool = True


def parse_requests_file(path: str) -> List[Request]:
    requests: List[Request] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split(",")] if "," in line else line.split()
            if line_no == 1 and parts and parts[0].lower() == "name":
                continue
            if len(parts) < 3:
                continue

            name = parts[0]
            pickup = parts[1]
            dropoff = parts[2]

            try:
                demand = int(parts[3]) if len(parts) >= 4 else 1
            except ValueError:
                demand = 1

            request_time = 0.0
            wait_limit = math.inf
            if len(parts) >= 6:
                try:
                    request_time = float(parts[4])
                except ValueError:
                    request_time = 0.0
                try:
                    wait_limit = float(parts[5])
                except ValueError:
                    wait_limit = math.inf
            elif len(parts) >= 5:
                try:
                    wait_limit = float(parts[4])
                except ValueError:
                    wait_limit = math.inf

            requests.append(
                Request(
                    name=name,
                    pickup=str(pickup),
                    dropoff=str(dropoff),
                    demand=demand,
                    request_time=request_time,
                    wait_limit=wait_limit,
                )
            )
    return requests


def parse_kv_list(kv_list: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in kv_list:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        try:
            if "." in value:
                out[key] = float(value)
            else:
                out[key] = int(value)
        except ValueError:
            low = value.lower()
            if low in {"true", "false"}:
                out[key] = low == "true"
            else:
                out[key] = value
    return out


class RideSharePlanner:
    def __init__(
        self,
        graph: Graph,
        source: str,
        target: str,
        requests: List[Request],
        **params: Any,
    ):
        self.graph = graph
        self.source = source
        self.target = target
        self.requests = requests
        self.capacity = int(params.get("capacity", 4))
        self.k = int(params.get("k", 5))
        self.max_events = int(params.get("max_events", max(2 * len(requests) + 2, 4)))
        self.max_states = int(params.get("max_states", 200000))
        self.search_timeout_seconds = float(
            params.get("search_timeout_seconds", math.inf)
        )
        self.time_budget = float(params.get("time_budget", math.inf))
        self.initial_time = float(params.get("initial_time", 0.0))
        self.initial_onboard_names = {
            str(name) for name in params.get("initial_onboard_names", ())
        }
        self.initial_pickup_times = {
            str(name): float(value)
            for name, value in dict(params.get("initial_pickup_times", {})).items()
        }
        self.detour_ratio = float(params.get("detour_ratio", 1.8))
        self.enforce_passenger_detour = bool(
            params.get("enforce_passenger_detour", True)
        )
        self.compatibility_scale = float(params.get("compatibility_scale", 1.0))
        compatibility_ratio = params.get("compatibility_ratio")
        self.compatibility_ratio = (
            float(compatibility_ratio)
            if compatibility_ratio is not None
            else math.inf
        )
        self.enable_compatibility_pruning = bool(
            params.get("enable_compatibility_pruning", compatibility_ratio is not None)
        )
        self.pareto_only = bool(params.get("pareto_only", True))
        self.numeric_tolerance = float(params.get("numeric_tolerance", 1e-9))
        self.max_candidates = int(params.get("max_candidates", self.k * 8))
        self.expensive_bound_slack_ratio = float(params.get("expensive_bound_slack_ratio", 0.05))
        self.expensive_bound_min_onboard = int(params.get("expensive_bound_min_onboard", 2))
        legacy_state_bounds = params.get("aggressive_served_pruning")
        self.enable_state_bounds = bool(
            params.get(
                "enable_state_bounds",
                legacy_state_bounds if legacy_state_bounds is not None else False,
            )
        )
        self.aggressive_served_pruning = bool(legacy_state_bounds or False)
        target_served = params.get("target_served_count")
        self.target_served_count = int(target_served) if target_served is not None else None
        self.enable_pruning = bool(params.get("enable_pruning", True))
        self.enable_dp = bool(params.get("enable_dp", True))
        self.shortest_path_cache = params.get("shortest_path_cache")
        # Dynamic paired runs need equal-cost undirected paths to be independent
        # of planner construction order. Static experiments keep their original
        # source-selection policy unless this option is explicitly enabled.
        self.canonical_undirected_paths = bool(
            params.get("canonical_undirected_paths", False)
        )
        self.time_dependent_shortest_path: Optional[
            Callable[[str, str, float], Tuple[List[str], float]]
        ] = params.get("time_dependent_shortest_path")
        default_dp_mode = (
            "bundled"
            if self.time_dependent_shortest_path is not None
            or self.initial_onboard_names
            else "exact_fast"
        )
        self.dp_mode = str(params.get("dp_mode", default_dp_mode)).strip().lower()
        if self.dp_mode not in {"exact_fast", "bundled"}:
            raise ValueError(f"unsupported dp_mode: {self.dp_mode!r}")
        # The original DPKMOR algorithm is the event-state DP solver.  Keep
        # route insertion only as an explicitly selected experimental mode.
        self.use_insertion_solver = bool(params.get("use_insertion_solver", False))
        # Optional feasible plans used only as a safe Pareto incumbent set.
        # The default is empty, so existing static experiments are unchanged.
        self.initial_candidates = list(params.get("initial_candidates", ()))
        self.include_initial_candidates_in_output = bool(
            params.get("include_initial_candidates_in_output", True)
        )
        self.stats = PlannerStats()
        self.search_certificate = SearchCertificate()

        self.key_nodes = self._build_key_nodes()
        self.dist_cache: Dict[Tuple[str, str], float] = {}
        self.path_cache: Dict[Tuple[str, str], List[str]] = {}
        self._dynamic_travel_cache: Dict[
            Tuple[str, str, float], Tuple[List[str], float]
        ] = {}
        self.state_lb_cache: Dict[Tuple[str, int, int, int], float] = {}
        self._mandatory_dropoff_lb_cache: Dict[Tuple[str, int], float] = {}
        self.coverage_ub_cache_limit = max(
            0,
            int(params.get("coverage_ub_cache_limit", 100000)),
        )
        self.enable_pair_coverage_bound = bool(
            params.get(
                "enable_pair_coverage_bound",
                self.time_dependent_shortest_path is not None,
            )
        )
        self.enable_seed_coverage_certificate = bool(
            params.get(
                "enable_seed_coverage_certificate",
                self.time_dependent_shortest_path is not None,
            )
        )
        self._coverage_ub_cache: Dict[Tuple[str, int, int, float], int] = {}
        self._pair_conflict_masks: List[int] = []
        self._pair_compatible_count_cache: Dict[int, int] = {0: 0}
        self._prefix_hop_cache: Dict[Tuple[str, ...], int] = {}
        self._dp_domination_counts: Dict[int, int] = {}
        self._dp_rank_cache: Dict[int, Tuple[Any, ...]] = {}
        self._insertion_cost_cache: Dict[
            Tuple[Tuple[str, ...], int], Tuple[Tuple[float, ...], Tuple[Tuple[int, int, float], ...]]
        ] = {}
        self._insertion_upper_cache: Dict[
            Tuple[Tuple[str, ...], Tuple[Tuple[str, str], ...]], int
        ] = {}
        self._insertion_replay_cache: Dict[
            Tuple[Tuple[str, ...], Tuple[Tuple[str, str], ...]], Optional[SearchState]
        ] = {}
        self._history_intern: Dict[
            Tuple[int, Tuple[str, str], str, Optional[int], Optional[float]],
            int,
        ] = {}
        self._history_records: Dict[
            int,
            Tuple[Optional[int], str, Tuple[str, str], Optional[int], Optional[float], int],
        ] = {}
        self._history_materialized_cache: Dict[
            int,
            Tuple[Tuple[str, ...], Tuple[float, ...], Tuple[Tuple[str, str], ...]],
        ] = {}
        self._history_base_key_path: Tuple[str, ...] = ()
        self._history_base_pickup_times: Tuple[float, ...] = ()
        self._history_base_route_events: Tuple[Tuple[str, str], ...] = ()
        self._history_base_prefix_hops = 0
        self._history_rank_cache: Dict[int, Tuple[Any, ...]] = {}
        self._max_unpicked_release_cache: Dict[int, float] = {}
        self._bundle_resource_index: Dict[Tuple[Any, ...], DPNode] = {}
        self._next_history_id = 1
        self.last_complete_frontier: List[Candidate] = []
        self._precompute_shortest_paths()
        self.base_route = self.path_cache.get((self.source, self.target), [])
        self.base_cost = self.dist_cache.get((self.source, self.target), math.inf)
        self.drop_target_costs = [self.dist(req.dropoff, self.target) for req in self.requests]
        self.compatible = self._build_compatibility_mask()
        self._build_pair_pickup_conflicts()
        self._certified_coverage_cap = self._seed_coverage_cap()

    def _optimistic_objective_is_covered(
        self,
        state: SearchState,
        frontier: Sequence[Candidate],
        *,
        strict: bool,
    ) -> bool:
        """Return whether a frontier plan certifies one unresolved state.

        ``_best_possible_served`` is an admissible coverage upper bound and
        ``time_used + _cheap_lower_bound`` is an admissible final-cost lower
        bound.  Weak coverage is enough for objective profiles; concrete plans
        require strict objective dominance to exclude an equivalent history.
        """
        upper_coverage = self._best_possible_served(state)
        completion_lb = self._cheap_lower_bound(state)
        lower_cost = state.time_used + completion_lb
        if math.isinf(lower_cost):
            return True

        tolerance = self.numeric_tolerance
        for candidate in frontier:
            weakly_covers = (
                candidate.served_count >= upper_coverage
                and candidate.total_cost <= lower_cost + tolerance
            )
            if not weakly_covers:
                continue
            if not strict:
                return True
            if (
                candidate.served_count > upper_coverage
                or candidate.total_cost < lower_cost - tolerance
            ):
                return True
        return False

    def _build_search_certificate(
        self,
        unresolved_states: Iterable[SearchState],
        frontier: Sequence[Candidate],
        *,
        scope_complete: bool = True,
        allow_resource_bounds: bool = True,
    ) -> SearchCertificate:
        # Queue entries can share a state object after lazy invalidation.  One
        # unresolved resource state only needs to be certified once.
        unique: Dict[int, SearchState] = {}
        for state in unresolved_states:
            unique[id(state)] = state
        unresolved = list(unique.values())
        search_exhausted = (
            not unresolved
            and not self.stats.hit_max_states
            and not self.stats.timed_out
        )

        objective_complete = search_exhausted
        concrete_complete = search_exhausted
        if allow_resource_bounds and unresolved:
            objective_complete = all(
                self._optimistic_objective_is_covered(
                    state,
                    frontier,
                    strict=False,
                )
                for state in unresolved
            )
            concrete_complete = all(
                self._optimistic_objective_is_covered(
                    state,
                    frontier,
                    strict=True,
                )
                for state in unresolved
            )

        certificate = SearchCertificate(
            search_exhausted=search_exhausted,
            hit_state_limit=self.stats.hit_max_states,
            timed_out=self.stats.timed_out,
            scope_complete=bool(scope_complete),
            unresolved_count=len(unresolved),
            objective_complete=(
                bool(scope_complete) and objective_complete
            ),
            concrete_complete=(
                bool(scope_complete) and concrete_complete
            ),
        )
        self.search_certificate = certificate
        return certificate

    def _build_key_nodes(self) -> List[str]:
        nodes = {self.source, self.target}
        for req in self.requests:
            nodes.add(req.pickup)
            nodes.add(req.dropoff)
        return sorted(nodes)

    def _precompute_shortest_paths(self) -> None:
        unresolved: Dict[str, List[str]] = {}
        for u in self.key_nodes:
            for v in self.key_nodes:
                cache_key = (u, v)
                if u == v:
                    self.dist_cache[cache_key] = 0.0
                    self.path_cache[cache_key] = [u]
                    continue
                if self.shortest_path_cache is not None and cache_key in self.shortest_path_cache:
                    path, dist = self.shortest_path_cache[cache_key]
                    self.dist_cache[cache_key] = dist
                    self.path_cache[cache_key] = path
                    continue
                reverse_key = (v, u)
                if (
                    not self.graph.directed
                    and self.shortest_path_cache is not None
                    and reverse_key in self.shortest_path_cache
                ):
                    reverse_path, dist = self.shortest_path_cache[reverse_key]
                    path = list(reversed(reverse_path))
                    self.dist_cache[cache_key] = dist
                    self.path_cache[cache_key] = path
                    self.shortest_path_cache[cache_key] = (path, dist)

        while True:
            unresolved.clear()
            for u in self.key_nodes:
                missing = [
                    v
                    for v in self.key_nodes
                    if u != v and (u, v) not in self.dist_cache
                ]
                if missing:
                    unresolved[u] = missing
            if not unresolved:
                break

            if self.canonical_undirected_paths and not self.graph.directed:
                # Always compute an unordered pair from its lexicographically
                # smaller endpoint. This makes equal-cost path selection stable
                # across incremental and full-recompute planner instances.
                u = min(unresolved)
                missing_targets = sorted(unresolved[u])
            else:
                # On undirected graphs, processing a newly added key node first
                # fills both directions and avoids one search from every old node.
                u = max(unresolved, key=lambda node: len(unresolved[node]))
                missing_targets = unresolved[u]
            results = self.graph.shortest_paths_from(u, missing_targets)
            for v in missing_targets:
                path, dist = results[v]
                cache_key = (u, v)
                self.dist_cache[cache_key] = dist
                self.path_cache[cache_key] = path
                if self.shortest_path_cache is not None:
                    self.shortest_path_cache[cache_key] = (path, dist)
                if not self.graph.directed:
                    reverse_key = (v, u)
                    reverse_path = list(reversed(path))
                    self.dist_cache[reverse_key] = dist
                    self.path_cache[reverse_key] = reverse_path
                    if self.shortest_path_cache is not None:
                        self.shortest_path_cache[reverse_key] = (
                            reverse_path,
                            dist,
                        )

    def dist(self, u: str, v: str) -> float:
        return self.dist_cache.get((u, v), math.inf)

    def travel(self, u: str, v: str, depart_time: float) -> Tuple[List[str], float]:
        if self.time_dependent_shortest_path is not None:
            key = (u, v, depart_time)
            cached = self._dynamic_travel_cache.get(key)
            if cached is not None:
                return cached
            result = self.time_dependent_shortest_path(u, v, depart_time)
            self._dynamic_travel_cache[key] = result
            return result
        return self.path_cache.get((u, v), []), self.dist(u, v)

    def _corridor_score(self, req: Request) -> float:
        direct = self.dist(req.pickup, req.dropoff)
        if math.isinf(direct):
            return math.inf
        via_req = self.dist(self.source, req.pickup) + direct + self.dist(req.dropoff, self.target)
        if self.base_cost == 0:
            return 1.0
        return via_req / self.base_cost

    def _build_compatibility_mask(self) -> int:
        mask = 0
        for idx, req in enumerate(self.requests):
            if req.name in self.initial_onboard_names:
                mask |= 1 << idx
                continue
            direct = self.dist(req.pickup, req.dropoff)
            if math.isinf(direct):
                continue
            if req.demand > self.capacity:
                self.stats.feasibility_pruned_states += 1
                continue
            earliest_pickup = self.initial_time + self.dist(self.source, req.pickup)
            if math.isinf(earliest_pickup):
                self.stats.feasibility_pruned_states += 1
                continue
            if earliest_pickup - req.request_time > req.wait_limit:
                self.stats.feasibility_pruned_states += 1
                continue
            minimum_complete_cost = (
                self.dist(self.source, req.pickup)
                + direct
                + self.dist(req.dropoff, self.target)
            )
            if minimum_complete_cost > self.time_budget + self.numeric_tolerance:
                self.stats.feasibility_pruned_states += 1
                continue
            if (
                self.enable_pruning
                and self.enable_compatibility_pruning
                and self._corridor_score(req)
                > self.compatibility_ratio * self.compatibility_scale
            ):
                self.stats.heuristic_pruned_states += 1
                self.stats.compatibility_pruned_requests += 1
                continue
            mask |= 1 << idx
        return mask

    def _mask_count(self, mask: int) -> int:
        return mask.bit_count()

    def _seed_coverage_cap(self) -> Optional[int]:
        """Prove that the best warm-start coverage cannot be exceeded.

        The relaxation visits only pickup vertices and the target using
        static lower-bound distances. It ignores every dropoff, vehicle
        capacity, passenger detour, and interactions with onboard requests.
        Consequently, failure to visit one more pickup is a safe certificate
        for the original KMOR search; success merely means no cap is known.
        """
        if (
            not self.enable_seed_coverage_certificate
            or not self.initial_candidates
        ):
            return None

        incumbent = max(
            (candidate.served_count for candidate in self.initial_candidates),
            default=-1,
        )
        onboard_count = sum(
            request.name in self.initial_onboard_names
            for request in self.requests
        )
        required_pickups = incumbent - onboard_count + 1
        if required_pickups <= 0:
            return None

        request_indices = [
            index
            for index, request in enumerate(self.requests)
            if ((self.compatible >> index) & 1)
            and request.name not in self.initial_onboard_names
        ]
        if len(request_indices) < required_pickups:
            return incumbent

        tolerance = self.numeric_tolerance
        frontier: Dict[Tuple[int, int], float] = {}
        for local_index, request_index in enumerate(request_indices):
            request = self.requests[request_index]
            distance = self.dist(self.source, request.pickup)
            if math.isinf(distance):
                continue
            arrival = max(self.initial_time + distance, request.request_time)
            if arrival > request.request_time + request.wait_limit + tolerance:
                continue
            frontier[(1 << local_index, local_index)] = arrival

        for visited_count in range(1, required_pickups + 1):
            if not frontier:
                break
            next_frontier: Dict[Tuple[int, int], float] = {}
            for (visited, last_local), arrival in frontier.items():
                last_request = self.requests[request_indices[last_local]]
                if visited_count == required_pickups:
                    if (
                        arrival + self.dist(last_request.pickup, self.target)
                        <= self.time_budget + tolerance
                    ):
                        return None
                    continue

                for next_local, next_request_index in enumerate(request_indices):
                    bit = 1 << next_local
                    if visited & bit:
                        continue
                    next_request = self.requests[next_request_index]
                    leg = self.dist(last_request.pickup, next_request.pickup)
                    if math.isinf(leg):
                        continue
                    next_arrival = max(
                        arrival + leg,
                        next_request.request_time,
                    )
                    if (
                        next_arrival
                        > next_request.request_time
                        + next_request.wait_limit
                        + tolerance
                    ):
                        continue
                    key = (visited | bit, next_local)
                    if next_arrival < next_frontier.get(key, math.inf):
                        next_frontier[key] = next_arrival
            frontier = next_frontier

        return incumbent

    def _build_pair_pickup_conflicts(self) -> None:
        """Precompute request pairs that cannot meet both pickup deadlines.

        The test ignores dropoffs, capacity, detour, and every other request.
        It is therefore only a necessary-condition filter: a marked pair can
        never coexist in a feasible continuation, while an unmarked pair may
        still turn out to be infeasible later.
        """
        count = len(self.requests)
        self._pair_conflict_masks = [0] * count
        if not self.enable_pair_coverage_bound:
            return

        tolerance = self.numeric_tolerance
        for left in range(count):
            if ((self.compatible >> left) & 1) == 0:
                continue
            first = self.requests[left]
            if first.name in self.initial_onboard_names:
                continue
            for right in range(left + 1, count):
                if ((self.compatible >> right) & 1) == 0:
                    continue
                second = self.requests[right]
                if second.name in self.initial_onboard_names:
                    continue

                first_pickup = max(
                    self.initial_time + self.dist(self.source, first.pickup),
                    first.request_time,
                )
                second_after_first = max(
                    first_pickup + self.dist(first.pickup, second.pickup),
                    second.request_time,
                )
                first_then_second = (
                    first_pickup
                    <= first.request_time + first.wait_limit + tolerance
                    and second_after_first
                    <= second.request_time + second.wait_limit + tolerance
                )

                second_pickup = max(
                    self.initial_time + self.dist(self.source, second.pickup),
                    second.request_time,
                )
                first_after_second = max(
                    second_pickup + self.dist(second.pickup, first.pickup),
                    first.request_time,
                )
                second_then_first = (
                    second_pickup
                    <= second.request_time + second.wait_limit + tolerance
                    and first_after_second
                    <= first.request_time + first.wait_limit + tolerance
                )
                if first_then_second or second_then_first:
                    continue
                self._pair_conflict_masks[left] |= 1 << right
                self._pair_conflict_masks[right] |= 1 << left

    def _max_pairwise_compatible(self, request_mask: int) -> int:
        """Exact MIS cardinality of the precomputed pickup-conflict graph."""
        cached = self._pair_compatible_count_cache.get(request_mask)
        if cached is not None:
            return cached

        conflicting = 0
        remaining = request_mask
        best_vertex = -1
        best_degree = -1
        while remaining:
            bit = remaining & -remaining
            vertex = bit.bit_length() - 1
            remaining ^= bit
            neighbors = self._pair_conflict_masks[vertex] & request_mask
            conflicting |= neighbors
            degree = neighbors.bit_count()
            if degree > best_degree:
                best_degree = degree
                best_vertex = vertex

        if conflicting == 0:
            result = request_mask.bit_count()
        else:
            vertex_bit = 1 << best_vertex
            without_vertex = request_mask & ~vertex_bit
            exclude = self._max_pairwise_compatible(without_vertex)
            include = 1 + self._max_pairwise_compatible(
                without_vertex & ~self._pair_conflict_masks[best_vertex]
            )
            result = max(exclude, include)
        self._pair_compatible_count_cache[request_mask] = result
        return result

    def _expand_full_path(self, key_path: List[str]) -> List[str]:
        if not key_path:
            return []
        full = [key_path[0]]
        for i in range(len(key_path) - 1):
            segment = self.path_cache.get((key_path[i], key_path[i + 1]), [])
            if not segment:
                return []
            full.extend(segment[1:])
        return full

    def _expand_time_dependent_path(
        self,
        key_path: List[str],
        route_events: Tuple[Tuple[str, str], ...],
    ) -> List[str]:
        if not key_path:
            return []
        requests_by_name = {request.name: request for request in self.requests}
        full_path = [key_path[0]]
        current_time = self.initial_time
        for idx, (u, v) in enumerate(zip(key_path, key_path[1:])):
            path, travel_time = self.travel(u, v, current_time)
            if not path or math.isinf(travel_time):
                return []
            full_path.extend(path[1:])
            current_time += travel_time
            if idx < len(route_events):
                action, name = route_events[idx]
                if action == "pickup" and name in requests_by_name:
                    current_time = max(current_time, requests_by_name[name].request_time)
        return full_path

    def _served_names(self, served_mask: int) -> List[str]:
        names = []
        for idx, req in enumerate(self.requests):
            if (served_mask >> idx) & 1:
                names.append(req.name)
        return names

    def _served_pickup_info(
        self, served_mask: int, pickup_times: Tuple[float, ...]
    ) -> List[Tuple[str, float, float, float]]:
        info: List[Tuple[str, float, float, float]] = []
        for idx, req in enumerate(self.requests):
            if (served_mask >> idx) & 1:
                pickup_time = pickup_times[idx]
                wait_time = pickup_time - req.request_time
                info.append((req.name, req.request_time, pickup_time, wait_time))
        info.sort(key=lambda item: item[2])
        return info

    def _can_pick(self, req_idx: int) -> bool:
        return ((self.compatible >> req_idx) & 1) == 1

    def _onboard_mask(self, state: SearchState) -> int:
        return state.picked_mask & ~state.served_mask

    def _ride_time(self, state: SearchState, req_idx: int, at_time: float) -> float:
        pickup_time = state.pickup_times[req_idx]
        if pickup_time < 0:
            return math.inf
        return max(0.0, at_time - pickup_time)

    def _ride_time_limit(self, req_idx: int) -> float:
        direct = self.dist(
            self.requests[req_idx].pickup,
            self.requests[req_idx].dropoff,
        )
        if math.isinf(direct):
            return -math.inf
        return self.detour_ratio * direct

    def _onboard_detour_feasible(self, state: SearchState, at_time: float) -> bool:
        if not self.enforce_passenger_detour:
            return True
        onboard_mask = self._onboard_mask(state)
        for idx in range(len(self.requests)):
            if ((onboard_mask >> idx) & 1) == 0:
                continue
            if (
                self._ride_time(state, idx, at_time)
                > self._ride_time_limit(idx) + self.numeric_tolerance
            ):
                return False
        return True

    def _transition_pickup(self, state: SearchState, req_idx: int) -> Optional[SearchState]:
        self.stats.feasibility_checks += 1
        req = self.requests[req_idx]
        if not self._can_pick(req_idx):
            return None
        if (state.picked_mask >> req_idx) & 1:
            return None
        if state.load + req.demand > self.capacity:
            self.stats.feasibility_pruned_states += 1
            return None

        _, travel_time = self.travel(state.node, req.pickup, state.time_used)
        if math.isinf(travel_time):
            self.stats.feasibility_pruned_states += 1
            return None

        arrival_time = state.time_used + travel_time
        if arrival_time < req.request_time:
            arrival_time = req.request_time
        if not self._onboard_detour_feasible(state, arrival_time):
            self.stats.feasibility_pruned_states += 1
            return None
        if arrival_time - req.request_time > req.wait_limit:
            self.stats.feasibility_pruned_states += 1
            return None
        if arrival_time + self.dist(req.pickup, self.target) > self.time_budget:
            self.stats.feasibility_pruned_states += 1
            return None

        pickup_times = list(state.pickup_times)
        pickup_times[req_idx] = arrival_time
        return SearchState(
            node=req.pickup,
            picked_mask=state.picked_mask | (1 << req_idx),
            served_mask=state.served_mask,
            load=state.load + req.demand,
            time_used=arrival_time,
            key_path=state.key_path + (req.pickup,),
            pickup_times=tuple(pickup_times),
            route_events=state.route_events + (("pickup", req.name),),
        )

    def _transition_dropoff(self, state: SearchState, req_idx: int) -> Optional[SearchState]:
        self.stats.feasibility_checks += 1
        if ((state.picked_mask >> req_idx) & 1) == 0:
            return None
        if ((state.served_mask >> req_idx) & 1) == 1:
            return None

        req = self.requests[req_idx]
        _, travel_time = self.travel(state.node, req.dropoff, state.time_used)
        if math.isinf(travel_time):
            self.stats.feasibility_pruned_states += 1
            return None

        arrival_time = state.time_used + travel_time
        if not self._onboard_detour_feasible(state, arrival_time):
            self.stats.feasibility_pruned_states += 1
            return None
        if arrival_time + self.dist(req.dropoff, self.target) > self.time_budget:
            self.stats.feasibility_pruned_states += 1
            return None

        return SearchState(
            node=req.dropoff,
            picked_mask=state.picked_mask,
            served_mask=state.served_mask | (1 << req_idx),
            load=state.load - req.demand,
            time_used=arrival_time,
            key_path=state.key_path + (req.dropoff,),
            pickup_times=state.pickup_times,
            route_events=state.route_events + (("dropoff", req.name),),
        )

    def _transition_target(self, state: SearchState) -> Optional[SearchState]:
        self.stats.feasibility_checks += 1
        if state.load != 0 or self._onboard_mask(state) != 0:
            self.stats.feasibility_pruned_states += 1
            return None
        _, travel_time = self.travel(state.node, self.target, state.time_used)
        if math.isinf(travel_time):
            self.stats.feasibility_pruned_states += 1
            return None
        arrival_time = state.time_used + travel_time
        if arrival_time > self.time_budget:
            self.stats.feasibility_pruned_states += 1
            return None
        return SearchState(
            node=self.target,
            picked_mask=state.picked_mask,
            served_mask=state.served_mask,
            load=0,
            time_used=arrival_time,
            key_path=state.key_path + (self.target,),
            pickup_times=state.pickup_times,
            route_events=state.route_events + (("target", ""),),
        )

    def _best_possible_served(self, state: SearchState) -> int:
        cache_key = (
            state.node,
            state.picked_mask,
            state.served_mask,
            state.time_used,
        )
        if self.coverage_ub_cache_limit > 0:
            cached = self._coverage_ub_cache.get(cache_key)
            if cached is not None:
                return cached

        possible = self._mask_count(state.served_mask)
        possible += self._mask_count(self._onboard_mask(state))
        used = state.picked_mask | state.served_mask
        reachable_mask = 0
        for idx in range(len(self.requests)):
            if (used >> idx) & 1:
                continue
            if self._request_still_individually_reachable(state, idx):
                reachable_mask |= 1 << idx

        if self.enable_pair_coverage_bound:
            possible += self._max_pairwise_compatible(reachable_mask)
        else:
            possible += self._mask_count(reachable_mask)

        remaining_events = max(0, self.max_events - len(state.key_path))
        mandatory_dropoffs = self._mask_count(self._onboard_mask(state))
        event_limited_pickups = max(
            0,
            (remaining_events - mandatory_dropoffs - 1) // 2,
        )
        possible = min(
            possible,
            self._mask_count(state.served_mask)
            + mandatory_dropoffs
            + event_limited_pickups,
        )
        if self._certified_coverage_cap is not None:
            possible = min(possible, self._certified_coverage_cap)

        if self.coverage_ub_cache_limit > 0:
            if len(self._coverage_ub_cache) >= self.coverage_ub_cache_limit:
                self._coverage_ub_cache.clear()
            self._coverage_ub_cache[cache_key] = possible
        return possible

    def _request_still_individually_reachable(
        self,
        state: SearchState,
        req_idx: int,
    ) -> bool:
        """Return a safe optimistic test for one remaining request.

        Other requests are ignored, so a request is rejected only when it
        cannot finish even by itself. This tightens the served-count upper
        bound without removing any feasible route.
        """
        if not self._can_pick(req_idx):
            return False
        req = self.requests[req_idx]
        pickup_leg = self.dist(state.node, req.pickup)
        direct_leg = self.dist(req.pickup, req.dropoff)
        target_leg = self.drop_target_costs[req_idx]
        if any(math.isinf(value) for value in (pickup_leg, direct_leg, target_leg)):
            return False

        pickup_time = max(state.time_used + pickup_leg, req.request_time)
        if pickup_time - req.request_time > req.wait_limit + self.numeric_tolerance:
            return False
        return (
            pickup_time + direct_leg + target_leg
            <= self.time_budget + self.numeric_tolerance
        )

    def _priority(self, state: SearchState) -> float:
        served = self._mask_count(state.served_mask)
        optimistic = self._best_possible_served(state)
        return -(served * 100000 + optimistic * 1000) + state.time_used

    def _state_signature(self, state: SearchState) -> Tuple[str, int, int, int]:
        return (state.node, state.picked_mask, state.served_mask, state.load)

    def _cheap_lower_bound(self, state: SearchState) -> float:
        onboard_mask = state.picked_mask & ~state.served_mask
        cache_key = (state.node, onboard_mask)
        cached = self._mandatory_dropoff_lb_cache.get(cache_key)
        if cached is not None:
            return cached

        direct = self.dist(state.node, self.target)
        if onboard_mask == 0 or math.isinf(direct):
            self._mandatory_dropoff_lb_cache[cache_key] = direct
            return direct

        # Capacity is small, so an exact Held-Karp path over mandatory
        # dropoffs is inexpensive. It ignores release/detour constraints and
        # all unpicked requests, making it an admissible completion bound.
        dropoffs = sorted(
            {
                self.requests[idx].dropoff
                for idx in range(len(self.requests))
                if (onboard_mask >> idx) & 1
            }
        )
        count = len(dropoffs)
        full_mask = (1 << count) - 1
        dp: Dict[Tuple[int, int], float] = {}
        for index, dropoff in enumerate(dropoffs):
            distance = self.dist(state.node, dropoff)
            if not math.isinf(distance):
                dp[(1 << index, index)] = distance

        for visited in range(1, full_mask + 1):
            for last in range(count):
                cost = dp.get((visited, last))
                if cost is None:
                    continue
                for nxt in range(count):
                    bit = 1 << nxt
                    if visited & bit:
                        continue
                    leg = self.dist(dropoffs[last], dropoffs[nxt])
                    if math.isinf(leg):
                        continue
                    key = (visited | bit, nxt)
                    candidate = cost + leg
                    if candidate < dp.get(key, math.inf):
                        dp[key] = candidate

        mandatory = min(
            (
                dp.get((full_mask, last), math.inf)
                + self.dist(dropoffs[last], self.target)
                for last in range(count)
            ),
            default=math.inf,
        )
        lower = max(direct, mandatory)
        self._mandatory_dropoff_lb_cache[cache_key] = lower
        return lower

    def _pickup_request_bound(self, state: SearchState, req_idx: int) -> float:
        req = self.requests[req_idx]
        drop_leg = self.dist(state.node, req.dropoff)
        tail_leg = self.drop_target_costs[req_idx]
        if math.isinf(drop_leg) or math.isinf(tail_leg):
            return math.inf
        return drop_leg + tail_leg

    def _expensive_lower_bound(self, state: SearchState) -> float:
        signature = self._state_signature(state)
        cached = self.state_lb_cache.get(signature)
        if cached is not None:
            return cached

        lower = self._cheap_lower_bound(state)
        if math.isinf(lower):
            return math.inf

        onboard_mask = state.picked_mask & ~state.served_mask
        for idx in range(len(self.requests)):
            if ((onboard_mask >> idx) & 1) == 0:
                continue
            req = self.requests[idx]
            drop_leg = self.dist(state.node, req.dropoff)
            tail_leg = self.drop_target_costs[idx]
            if math.isinf(drop_leg) or math.isinf(tail_leg):
                return math.inf
            lower = max(lower, drop_leg + tail_leg)
        self.state_lb_cache[signature] = lower
        return lower

    def _should_run_expensive_bound(
        self,
        state: SearchState,
        complete: List[Candidate],
        cheap_lower_bound: float,
    ) -> bool:
        if math.isinf(cheap_lower_bound):
            return False

        onboard_mask = state.picked_mask & ~state.served_mask
        onboard_count = self._mask_count(onboard_mask)
        if onboard_count == 0:
            return False

        slack = self.time_budget - (state.time_used + cheap_lower_bound)
        threshold = max(1.0, self.base_cost * self.expensive_bound_slack_ratio)

        # Only pay for the stronger bound when the route is already tight, or
        # when we already have enough complete candidates and need a sharper cutoff.
        if len(complete) < self.k:
            return False
        if onboard_count < self.expensive_bound_min_onboard:
            return False
        return slack <= threshold

    def _build_candidate(self, state: SearchState) -> Optional[Candidate]:
        if state.node != self.target or state.load != 0 or self._onboard_mask(state) != 0:
            return None
        key_path = list(state.key_path)
        if self.time_dependent_shortest_path is None:
            full_path = self._expand_full_path(key_path)
        else:
            full_path = self._expand_time_dependent_path(key_path, state.route_events)
        if not full_path:
            return None
        served = self._mask_count(state.served_mask)
        total_cost = state.time_used
        return Candidate(
            key_path=key_path,
            full_path=full_path,
            served_requests=self._served_names(state.served_mask),
            pickup_info=self._served_pickup_info(state.served_mask, state.pickup_times),
            total_cost=total_cost,
            total_time=total_cost,
            hops=max(0, len(full_path) - 1),
            served_count=served,
            score=served * 100000 - total_cost,
            route_events=list(state.route_events),
        )

    def _pareto_top_k(self, candidates: List[Candidate]) -> List[Candidate]:
        return select_top_k_candidates(
            candidates,
            self.k,
            target_served_count=self.target_served_count,
            pareto_only=self.pareto_only,
            numeric_tolerance=self.numeric_tolerance,
        )

    def _state_resources_equal(self, left: SearchState, right: SearchState) -> bool:
        if abs(left.time_used - right.time_used) > self.numeric_tolerance:
            return False
        onboard_mask = self._onboard_mask(left)
        while onboard_mask:
            bit = onboard_mask & -onboard_mask
            idx = bit.bit_length() - 1
            onboard_mask ^= bit
            if (
                abs(left.pickup_times[idx] - right.pickup_times[idx])
                > self.numeric_tolerance
            ):
                return False
        return True

    def _state_dominates(self, left: SearchState, right: SearchState) -> bool:
        """Return whether left safely and strictly dominates right."""
        if not self._state_weakly_dominates(left, right):
            return False

        strict = left.time_used < right.time_used - self.numeric_tolerance
        onboard_mask = self._onboard_mask(left)
        while onboard_mask:
            bit = onboard_mask & -onboard_mask
            idx = bit.bit_length() - 1
            onboard_mask ^= bit
            # A later pickup time means an equal or later absolute detour
            # deadline. This remains safe even if a future pickup causes both
            # labels to wait until the same request release time.
            left_pickup = left.pickup_times[idx]
            right_pickup = right.pickup_times[idx]
            if left_pickup < right_pickup - self.numeric_tolerance:
                return False
            if left_pickup > right_pickup + self.numeric_tolerance:
                strict = True
        return strict

    def _state_weakly_dominates(
        self,
        left: SearchState,
        right: SearchState,
    ) -> bool:
        if self._state_signature(left) != self._state_signature(right):
            return False
        if left.time_used > right.time_used + self.numeric_tolerance:
            return False
        onboard_mask = self._onboard_mask(left)
        while onboard_mask:
            bit = onboard_mask & -onboard_mask
            idx = bit.bit_length() - 1
            onboard_mask ^= bit
            if (
                left.pickup_times[idx]
                < right.pickup_times[idx] - self.numeric_tolerance
            ):
                return False
        return True

    def _future_wait_can_equalize(self, state: SearchState) -> bool:
        """Return whether a later release can erase an arrival-time advantage."""
        if self.time_dependent_shortest_path is not None:
            return True
        max_release = self._max_unpicked_release_cache.get(state.picked_mask)
        if max_release is None:
            max_release = -math.inf
            unpicked_mask = ((1 << len(self.requests)) - 1) & ~state.picked_mask
            while unpicked_mask:
                bit = unpicked_mask & -unpicked_mask
                idx = bit.bit_length() - 1
                unpicked_mask ^= bit
                max_release = max(max_release, self.requests[idx].request_time)
            self._max_unpicked_release_cache[state.picked_mask] = max_release
        return max_release > state.time_used + self.numeric_tolerance

    def _persistent_cost_dominates(
        self,
        left: SearchState,
        right: SearchState,
    ) -> bool:
        """Return whether left stays strictly cheaper under every common suffix."""
        if not self._state_weakly_dominates(left, right):
            return False
        if left.time_used >= right.time_used - self.numeric_tolerance:
            return False
        return not self._future_wait_can_equalize(left)

    def _prefix_hops(self, state: SearchState) -> int:
        cached = self._prefix_hop_cache.get(state.key_path)
        if cached is not None:
            return cached
        hops = 0
        for source, target in zip(state.key_path, state.key_path[1:]):
            segment = self.path_cache.get((source, target), ())
            if not segment:
                return 2**31 - 1
            hops += max(0, len(segment) - 1)
        self._prefix_hop_cache[state.key_path] = hops
        return hops

    def _dp_label_rank(self, state: SearchState) -> Tuple[Any, ...]:
        cached = self._dp_rank_cache.get(id(state))
        if cached is not None:
            return cached
        onboard_mask = self._onboard_mask(state)
        pickup_rank = tuple(
            -state.pickup_times[idx]
            for idx in range(len(self.requests))
            if (onboard_mask >> idx) & 1
        )
        rank = (
            self._prefix_hops(state),
            state.route_events,
            state.key_path,
            state.time_used,
            pickup_rank,
        )
        self._dp_rank_cache[id(state)] = rank
        return rank

    def _candidate_dominates(self, left: Candidate, right: Candidate) -> bool:
        no_worse = (
            left.served_count >= right.served_count
            and left.total_cost <= right.total_cost + self.numeric_tolerance
        )
        strictly_better = (
            left.served_count > right.served_count
            or left.total_cost < right.total_cost - self.numeric_tolerance
        )
        return no_worse and strictly_better

    def _update_complete_frontier(
        self,
        frontier: List[Candidate],
        candidate: Candidate,
    ) -> List[Candidate]:
        if any(self._candidate_dominates(old, candidate) for old in frontier):
            return frontier
        kept = [
            old
            for old in frontier
            if not self._candidate_dominates(candidate, old)
        ]
        kept.append(candidate)
        return kept

    def _pareto_bound_prunes(
        self,
        state: SearchState,
        frontier: List[Candidate],
    ) -> bool:
        if not frontier:
            return False
        remaining_lower_bound = self._cheap_lower_bound(state)
        if math.isinf(remaining_lower_bound):
            return True
        lower_cost = state.time_used + remaining_lower_bound
        upper_served = self._best_possible_served(state)
        return any(
            candidate.served_count >= upper_served
            and candidate.total_cost <= lower_cost + self.numeric_tolerance
            and (
                candidate.served_count > upper_served
                or candidate.total_cost < lower_cost - self.numeric_tolerance
            )
            for candidate in frontier
        )

    def _record_state_bound_prune(self) -> None:
        self.stats.state_bound_pruned_states += 1
        self.stats.heuristic_pruned_states += 1

    def _record_pareto_bound_prune(self) -> None:
        self.stats.pareto_bound_pruned_states += 1
        self.stats.heuristic_pruned_states += 1

    def _should_enqueue(
        self,
        state: SearchState,
        dp_frontier: Dict[Tuple[str, int, int, int], List[SearchState]],
    ) -> bool:
        if not self.enable_dp:
            return True
        signature = self._state_signature(state)
        labels = dp_frontier.setdefault(signature, [])
        if any(old == state for old in labels):
            self.stats.dp_pruned_states += 1
            return False

        if any(self._persistent_cost_dominates(old, state) for old in labels):
            self.stats.dp_pruned_states += 1
            return False

        # A single better label preserves the Pareto objective front, but it
        # can erase a distinct route that later ties after a release-time wait.
        # Keep k better-ranked route histories and prune only after all k can
        # replace this prefix under every common suffix.
        rank = self._dp_label_rank(state)
        ranked_dominators = [
            old
            for old in labels
            if self._state_weakly_dominates(old, state)
            and self._dp_label_rank(old) <= rank
        ]
        if len(ranked_dominators) >= max(1, self.k):
            self.stats.dp_pruned_states += 1
            return False

        survivors: List[SearchState] = []
        for old in labels:
            if self._persistent_cost_dominates(state, old):
                continue
            if (
                rank < self._dp_label_rank(old)
                and self._state_weakly_dominates(state, old)
            ):
                old_id = id(old)
                self._dp_domination_counts[old_id] = (
                    self._dp_domination_counts.get(old_id, 0) + 1
                )
                if self._dp_domination_counts[old_id] >= max(1, self.k):
                    continue
            survivors.append(old)

        self._dp_domination_counts[id(state)] = len(ranked_dominators)
        survivors.append(state)
        dp_frontier[signature] = survivors
        return True

    def _is_active_dp_label(
        self,
        state: SearchState,
        dp_frontier: Dict[Tuple[str, int, int, int], List[SearchState]],
    ) -> bool:
        if not self.enable_dp:
            return True
        return any(
            label is state
            for label in dp_frontier.get(self._state_signature(state), [])
        )

    def _push_state(
        self,
        pq: List[Tuple[float, int, SearchState]],
        counter: int,
        state: SearchState,
        dp_frontier: Dict[Tuple[str, int, int, int], List[SearchState]],
    ) -> int:
        if not self._should_enqueue(state, dp_frontier):
            return counter
        heapq.heappush(pq, (self._priority(state), counter, state))
        self.stats.generated_states += 1
        self.stats.peak_open_size = max(self.stats.peak_open_size, len(pq))
        return counter + 1

    def _history_key(
        self,
        history_id: int,
    ) -> int:
        return history_id

    def _history_rank(self, history_id: int) -> Tuple[Any, ...]:
        cached = self._history_rank_cache.get(history_id)
        if cached is not None:
            return cached
        if history_id == 0:
            prefix_hops = self._history_base_prefix_hops
        else:
            prefix_hops = self._history_records[history_id][5]
        key_path, _, route_events = self._materialize_history(history_id)
        # Equal-resource histories receive the same future suffix. Ranking by
        # prefix hops, events, and key path is therefore suffix-compatible and
        # matches the deterministic concrete-plan order used by verification.
        rank = (prefix_hops, route_events, key_path, history_id)
        self._history_rank_cache[history_id] = rank
        return rank

    def _history_prefix_hops(self, history_id: int) -> int:
        if history_id == 0:
            return self._history_base_prefix_hops
        return self._history_records[history_id][5]

    def _bundle_resource_key(self, state: SearchState) -> Tuple[Any, ...]:
        """Exact lookup key for the common equal-resource merge case.

        A tolerance-aware fallback remains in ``_enqueue_dp_bundle`` so this
        index cannot change label equivalence or correctness near a bucket
        boundary.
        """
        onboard_mask = self._onboard_mask(state)
        pickup_resources: List[float] = []
        remaining = onboard_mask
        while remaining:
            bit = remaining & -remaining
            idx = bit.bit_length() - 1
            remaining ^= bit
            pickup_resources.append(state.pickup_times[idx])
        return (
            state.node,
            state.picked_mask,
            state.served_mask,
            state.load,
            state.time_used,
            tuple(pickup_resources),
        )

    def _history_from_state(self, state: SearchState) -> int:
        self._history_base_key_path = state.key_path
        self._history_base_pickup_times = state.pickup_times
        self._history_base_route_events = state.route_events
        self._history_base_prefix_hops = self._prefix_hops(state)
        self._history_records.clear()
        self._history_materialized_cache.clear()
        self._history_intern.clear()
        self._history_rank_cache.clear()
        self._next_history_id = 1
        return 0

    def _materialize_history(
        self,
        history_id: int,
    ) -> Tuple[Tuple[str, ...], Tuple[float, ...], Tuple[Tuple[str, str], ...]]:
        cached = self._history_materialized_cache.get(history_id)
        if cached is not None:
            return cached

        if history_id == 0:
            value = (
                self._history_base_key_path,
                self._history_base_pickup_times,
                self._history_base_route_events,
            )
        else:
            parent_id, node, event, request_idx, pickup_time, _ = (
                self._history_records[history_id]
            )
            parent_value = self._materialize_history(parent_id or 0)
            key_path = parent_value[0] + (node,)
            route_events = parent_value[2] + (event,)
            if request_idx is None:
                pickup_times = parent_value[1]
            else:
                values = list(parent_value[1])
                values[request_idx] = pickup_time
                pickup_times = tuple(values)
            value = (key_path, pickup_times, route_events)

        if len(self._history_materialized_cache) >= 100000:
            self._history_materialized_cache.clear()
        self._history_materialized_cache[history_id] = value
        return value

    def _state_with_history(
        self,
        resource_state: SearchState,
        history_id: int,
    ) -> SearchState:
        key_path, pickup_times, route_events = self._materialize_history(history_id)
        return SearchState(
            node=resource_state.node,
            picked_mask=resource_state.picked_mask,
            served_mask=resource_state.served_mask,
            load=resource_state.load,
            time_used=resource_state.time_used,
            key_path=key_path,
            pickup_times=pickup_times,
            route_events=route_events,
        )

    def _merge_dp_histories(
        self,
        node: DPNode,
        histories: Iterable[int],
    ) -> bool:
        history_values = list(histories)
        if len(history_values) == 1 and len(node.histories) < max(1, self.k):
            key = history_values[0]
            if key not in node.histories:
                node.histories[key] = None
                return True

        old_histories = set(node.histories)
        combined = set(node.histories)
        combined.update(history_values)
        retained = sorted(combined, key=self._history_rank)[: max(1, self.k)]
        representative = retained[0]
        if representative != node.representative_history_id:
            node.representative_history_id = representative
        node.histories = {history_id: None for history_id in retained}
        return any(
            history_id not in old_histories
            and history_id not in node.propagated_history_keys
            for history_id in retained
        )

    def _queue_dp_node(
        self,
        pq: List[Tuple[float, int, DPNode]],
        counter: int,
        node: DPNode,
    ) -> int:
        if not node.active or node.queued:
            return counter
        if not any(
            key not in node.propagated_history_keys
            for key in node.histories
        ):
            return counter
        heapq.heappush(pq, (node.priority, counter, node))
        node.queued = True
        self.stats.peak_open_size = max(self.stats.peak_open_size, len(pq))
        return counter + 1

    def _segment_hops_for_transition(
        self,
        state: SearchState,
        successor: SearchState,
    ) -> int:
        if self.time_dependent_shortest_path is None:
            segment = self.path_cache.get((state.node, successor.node), ())
        else:
            segment, _ = self.travel(state.node, successor.node, state.time_used)
        return max(0, len(segment) - 1) if segment else 2**31 - 1

    def _extend_history(
        self,
        history_id: int,
        transition: DPTransition,
    ) -> int:
        successor = transition.successor
        event = successor.route_events[-1]
        request_idx = transition.request_idx if event[0] == "pickup" else None
        pickup_time = successor.time_used if request_idx is not None else None
        # Keep timing in the key: equal event/node suffixes can still belong
        # to different resource labels when release times are involved.
        intern_key = (history_id, event, successor.node, request_idx, pickup_time)
        route_id = self._history_intern.get(intern_key)
        if route_id is None:
            route_id = self._next_history_id
            self._next_history_id += 1
            self._history_intern[intern_key] = route_id
            # Reading the hop counter must not materialize the complete
            # history. This operation is on every bundled transition.
            parent_hops = self._history_prefix_hops(history_id)
            self._history_records[route_id] = (
                history_id,
                successor.node,
                event,
                request_idx,
                pickup_time,
                parent_hops + transition.segment_hops,
            )
        return route_id

    def _enqueue_dp_bundle(
        self,
        pq: List[Tuple[float, int, DPNode]],
        counter: int,
        dp_frontier: Dict[Tuple[str, int, int, int], List[DPNode]],
        state: SearchState,
        histories: Iterable[int],
    ) -> int:
        history_values = list(histories)
        if not history_values:
            return counter

        signature = self._state_signature(state)
        labels = [node for node in dp_frontier.setdefault(signature, []) if node.active]
        dp_frontier[signature] = labels

        resource_key = self._bundle_resource_key(state)
        indexed = self._bundle_resource_index.get(resource_key)
        if (
            indexed is not None
            and indexed.active
            and self._state_resources_equal(indexed.state, state)
        ):
            if self._merge_dp_histories(indexed, history_values):
                counter = self._queue_dp_node(pq, counter, indexed)
            self.stats.dp_pruned_states += len(history_values)
            return counter

        for node in labels:
            if self._state_resources_equal(node.state, state):
                self._bundle_resource_index[resource_key] = node
                if self._merge_dp_histories(node, history_values):
                    counter = self._queue_dp_node(pq, counter, node)
                self.stats.dp_pruned_states += len(history_values)
                return counter

        dominated_nodes: List[DPNode] = []
        for node in labels:
            if self._persistent_cost_dominates(node.state, state):
                self.stats.dp_pruned_states += len(history_values)
                return counter
            if self._persistent_cost_dominates(state, node.state):
                dominated_nodes.append(node)

        if dominated_nodes:
            dominated_ids = {id(node) for node in dominated_nodes}
            survivors = [
                node for node in labels if id(node) not in dominated_ids
            ]
            for node in dominated_nodes:
                node.active = False
                self.stats.dp_pruned_states += max(1, len(node.histories))
        else:
            survivors = labels

        unique_histories = set(history_values)
        if len(history_values) == 1:
            retained = history_values
        else:
            retained = sorted(unique_histories, key=self._history_rank)[: max(1, self.k)]
        first = retained[0]
        node = DPNode(
            # ``state`` is already materialized from the representative
            # history by the caller. Rebuilding it here duplicated the most
            # expensive tuple work for every newly created DP node.
            state=state,
            histories={history_id: None for history_id in retained},
            priority=self._priority(state),
            representative_history_id=first,
        )
        survivors.append(node)
        dp_frontier[signature] = survivors
        self._bundle_resource_index[resource_key] = node
        self.stats.generated_states += 1
        self.stats.dp_pruned_states += max(0, len(history_values) - 1)
        return self._queue_dp_node(pq, counter, node)

    def _add_dag_incoming(
        self,
        node: DPNode,
        parent: Optional[DPNode],
        transition: Optional[DPTransition],
    ) -> None:
        if parent is None or transition is None:
            return
        event = transition.successor.route_events[-1]
        key = (id(parent), event)
        if key in node.incoming_keys:
            return
        node.incoming_keys.add(key)
        node.incoming.append((parent, transition))
        self.stats.dag_edges_created += 1

    def _record_new_dag_node(self) -> None:
        self.stats.dag_nodes_created += 1
        self.stats.active_dp_labels += 1
        self.stats.peak_active_dp_labels = max(
            self.stats.peak_active_dp_labels,
            self.stats.active_dp_labels,
        )

    def _deactivate_dag_node(self, node: DPNode) -> None:
        if not node.active:
            return
        node.active = False
        self.stats.active_dp_labels = max(
            0,
            self.stats.active_dp_labels - 1,
        )

    def _queue_dp_dag_node(
        self,
        pq: List[Tuple[float, int, DPNode]],
        counter: int,
        node: DPNode,
    ) -> int:
        if not node.active or node.queued:
            return counter
        heapq.heappush(pq, (node.priority, counter, node))
        node.queued = True
        self.stats.peak_open_size = max(self.stats.peak_open_size, len(pq))
        return counter + 1

    def _enqueue_dp_dag(
        self,
        pq: List[Tuple[float, int, DPNode]],
        counter: int,
        dp_frontier: Dict[Tuple[str, int, int, int], List[DPNode]],
        state: SearchState,
        parent: Optional[DPNode],
        transition: Optional[DPTransition],
    ) -> int:
        """Insert one resource label while retaining concrete-plan ancestry.

        Equal-resource histories become incoming DAG edges. The resource node
        is not re-expanded because every such history has exactly the same
        feasible suffix set and transition costs.
        """
        signature = self._state_signature(state)
        labels = [
            node for node in dp_frontier.setdefault(signature, []) if node.active
        ]
        dp_frontier[signature] = labels

        resource_key = self._bundle_resource_key(state)
        indexed = self._bundle_resource_index.get(resource_key)
        if (
            indexed is not None
            and indexed.active
            and self._state_resources_equal(indexed.state, state)
        ):
            self._add_dag_incoming(indexed, parent, transition)
            self.stats.dp_pruned_states += 1
            return counter

        for node in labels:
            if self._state_resources_equal(node.state, state):
                self._bundle_resource_index[resource_key] = node
                self._add_dag_incoming(node, parent, transition)
                self.stats.dp_pruned_states += 1
                return counter

        dominated_nodes: List[DPNode] = []
        for node in labels:
            if self._persistent_cost_dominates(node.state, state):
                self.stats.dp_pruned_states += 1
                return counter
            if self._persistent_cost_dominates(state, node.state):
                dominated_nodes.append(node)

        if dominated_nodes:
            dominated_ids = {id(node) for node in dominated_nodes}
            survivors = [
                node for node in labels if id(node) not in dominated_ids
            ]
            for node in dominated_nodes:
                self._deactivate_dag_node(node)
                self.stats.dp_pruned_states += 1
        else:
            survivors = labels

        node = DPNode(
            state=state,
            histories={},
            priority=self._priority(state),
        )
        self._add_dag_incoming(node, parent, transition)
        self._record_new_dag_node()
        survivors.append(node)
        dp_frontier[signature] = survivors
        self._bundle_resource_index[resource_key] = node
        self.stats.generated_states += 1
        return self._queue_dp_dag_node(pq, counter, node)

    def _dag_top_histories(
        self,
        node: DPNode,
        initial_node: DPNode,
        memo: Dict[int, Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        """Recover at most k suffix-compatible histories for one DAG node."""
        node_id = id(node)
        cached = memo.get(node_id)
        if cached is not None:
            return cached
        if node is initial_node:
            memo[node_id] = (0,)
            return memo[node_id]

        candidates: Set[int] = set()
        for parent, transition in node.incoming:
            for history_id in self._dag_top_histories(
                parent,
                initial_node,
                memo,
            ):
                candidates.add(self._extend_history(history_id, transition))

        retained = tuple(
            sorted(candidates, key=self._history_rank)[: max(1, self.k)]
        )
        memo[node_id] = retained
        return retained

    def _build_dp_transitions(
        self,
        state: SearchState,
        complete: List[Candidate],
    ) -> Tuple[DPTransition, ...]:
        transitions: List[DPTransition] = []
        onboard_mask = self._onboard_mask(state)
        if onboard_mask == 0 and state.load == 0:
            target_state = self._transition_target(state)
        else:
            target_state = None
        if target_state is not None:
            transitions.append(
                DPTransition(
                    successor=target_state,
                    request_idx=None,
                    segment_hops=self._segment_hops_for_transition(state, target_state),
                )
            )

        remaining_onboard = onboard_mask
        while remaining_onboard:
            bit = remaining_onboard & -remaining_onboard
            idx = bit.bit_length() - 1
            remaining_onboard ^= bit
            nxt = self._transition_dropoff(state, idx)
            if nxt is None:
                continue
            if self.enable_pruning and self.enable_state_bounds:
                lower_bound = self._cheap_lower_bound(nxt)
                if math.isinf(lower_bound) or nxt.time_used + lower_bound > self.time_budget:
                    self._record_state_bound_prune()
                    continue
            transitions.append(
                DPTransition(
                    successor=nxt,
                    request_idx=idx,
                    segment_hops=self._segment_hops_for_transition(state, nxt),
                )
            )

        pickup_mask = self.compatible & ~state.picked_mask
        while pickup_mask:
            bit = pickup_mask & -pickup_mask
            idx = bit.bit_length() - 1
            pickup_mask ^= bit
            nxt = self._transition_pickup(state, idx)
            if nxt is None:
                continue
            if self.enable_pruning and self.enable_state_bounds:
                lower_bound = self._cheap_lower_bound(nxt)
                if math.isinf(lower_bound) or nxt.time_used + lower_bound > self.time_budget:
                    self._record_state_bound_prune()
                    continue
                request_bound = self._pickup_request_bound(nxt, idx)
                lower_bound = max(lower_bound, request_bound)
                if math.isinf(lower_bound) or nxt.time_used + lower_bound > self.time_budget:
                    self._record_state_bound_prune()
                    continue
                if self._should_run_expensive_bound(nxt, complete, lower_bound):
                    expensive_bound = self._expensive_lower_bound(nxt)
                    if math.isinf(expensive_bound) or nxt.time_used + expensive_bound > self.time_budget:
                        self._record_state_bound_prune()
                        continue
            transitions.append(
                DPTransition(
                    successor=nxt,
                    request_idx=idx,
                    segment_hops=self._segment_hops_for_transition(state, nxt),
                )
            )
        return tuple(transitions)

    def _solve_bundled_eager_dp(self) -> List[Candidate]:
        """Expand each resource label once while propagating up to k histories."""
        search_started = time.monotonic()
        self._bundle_resource_index.clear()
        initial_picked_mask = 0
        initial_load = 0
        pickup_times = [-1.0] * len(self.requests)
        for idx, req in enumerate(self.requests):
            if req.name not in self.initial_onboard_names:
                continue
            initial_picked_mask |= 1 << idx
            initial_load += req.demand
            pickup_times[idx] = self.initial_pickup_times.get(req.name, self.initial_time)
        if initial_load > self.capacity:
            self._build_search_certificate([], [])
            return []

        initial = SearchState(
            node=self.source,
            picked_mask=initial_picked_mask,
            served_mask=0,
            load=initial_load,
            time_used=self.initial_time,
            key_path=(self.source,),
            pickup_times=tuple(pickup_times),
            route_events=(),
        )

        pq: List[Tuple[float, int, DPNode]] = []
        dp_frontier: Dict[Tuple[str, int, int, int], List[DPNode]] = {}
        counter = self._enqueue_dp_bundle(
            pq,
            0,
            dp_frontier,
            initial,
            [self._history_from_state(initial)],
        )
        complete: List[Candidate] = []
        complete_frontier: List[Candidate] = []
        expansions = 0

        while pq and expansions < self.max_states:
            if time.monotonic() - search_started > self.search_timeout_seconds:
                self.stats.timed_out = True
                break
            _, _, node = heapq.heappop(pq)
            node.queued = False
            self.stats.popped_states += 1
            if not node.active:
                continue

            pending = [
                history_id
                for history_id in node.histories
                if history_id not in node.propagated_history_keys
            ]
            if not pending:
                continue

            first_expansion = node.transitions is None
            if (
                first_expansion
                and self.enable_pruning
                and self._pareto_bound_prunes(node.state, complete_frontier)
            ):
                node.active = False
                self._record_pareto_bound_prune()
                continue
            if first_expansion:
                expansions += 1
                self.stats.expanded_states += 1

            node.propagated_history_keys.update(
                pending
            )
            onboard_mask = self._onboard_mask(node.state)
            if (
                node.state.node == self.target
                and onboard_mask == 0
                and node.state.load == 0
            ):
                node.transitions = ()
                for history_id in pending:
                    candidate = self._build_candidate(
                        self._state_with_history(node.state, history_id)
                    )
                    if candidate is None:
                        continue
                    complete.append(candidate)
                    complete_frontier = self._update_complete_frontier(
                        complete_frontier,
                        candidate,
                    )
                    self.stats.complete_candidates += 1
                continue

            if len(node.state.key_path) >= self.max_events:
                node.transitions = ()
                continue

            if first_expansion:
                node.transitions = self._build_dp_transitions(node.state, complete)

            for transition in node.transitions or ():
                next_histories = [
                    self._extend_history(history_id, transition)
                    for history_id in pending
                ]
                counter = self._enqueue_dp_bundle(
                    pq,
                    counter,
                    dp_frontier,
                    # Feasibility resources are identical for every history
                    # in the bundle. Concrete path tuples are reconstructed
                    # only for terminal plans and deterministic history rank.
                    transition.successor,
                    next_histories,
                )

        if pq and expansions >= self.max_states:
            self.stats.hit_max_states = True
        self.last_complete_frontier = select_top_k_candidates(
            complete_frontier,
            max(1, len(complete_frontier)),
            pareto_only=True,
            numeric_tolerance=self.numeric_tolerance,
        )
        self._build_search_certificate(
            (
                node.state
                for _, _, node in pq
                if node.active
            ),
            self.last_complete_frontier,
        )
        return self._pareto_top_k(complete)

    def _solve_bundled_dp(self) -> List[Candidate]:
        """Plan-level DP with resource search and lazy history reconstruction."""
        search_started = time.monotonic()
        self._bundle_resource_index.clear()
        initial_picked_mask = 0
        initial_load = 0
        pickup_times = [-1.0] * len(self.requests)
        for idx, req in enumerate(self.requests):
            if req.name not in self.initial_onboard_names:
                continue
            initial_picked_mask |= 1 << idx
            initial_load += req.demand
            pickup_times[idx] = self.initial_pickup_times.get(
                req.name,
                self.initial_time,
            )
        if initial_load > self.capacity:
            self._build_search_certificate([], [])
            return []

        initial = SearchState(
            node=self.source,
            picked_mask=initial_picked_mask,
            served_mask=0,
            load=initial_load,
            time_used=self.initial_time,
            key_path=(self.source,),
            pickup_times=tuple(pickup_times),
            route_events=(),
        )
        self._history_from_state(initial)

        pq: List[Tuple[float, int, DPNode]] = []
        dp_frontier: Dict[Tuple[str, int, int, int], List[DPNode]] = {}
        counter = self._enqueue_dp_dag(
            pq,
            0,
            dp_frontier,
            initial,
            None,
            None,
        )
        initial_node = self._bundle_resource_index[
            self._bundle_resource_key(initial)
        ]
        seed_candidates = select_top_k_candidates(
            self.initial_candidates,
            max(1, self.k),
            pareto_only=False,
            numeric_tolerance=self.numeric_tolerance,
        )
        terminal_records: List[Tuple[DPNode, Candidate]] = []
        complete_frontier: List[Candidate] = []
        for candidate in seed_candidates:
            complete_frontier = self._update_complete_frontier(
                complete_frontier,
                candidate,
            )
        expansions = 0

        while pq and expansions < self.max_states:
            if time.monotonic() - search_started > self.search_timeout_seconds:
                self.stats.timed_out = True
                break
            _, _, node = heapq.heappop(pq)
            node.queued = False
            self.stats.popped_states += 1
            if not node.active:
                continue

            if (
                self.enable_pruning
                and self._pareto_bound_prunes(node.state, complete_frontier)
            ):
                self._deactivate_dag_node(node)
                self._record_pareto_bound_prune()
                continue

            expansions += 1
            self.stats.expanded_states += 1
            onboard_mask = self._onboard_mask(node.state)
            if (
                node.state.node == self.target
                and onboard_mask == 0
                and node.state.load == 0
            ):
                node.transitions = ()
                candidate = self._build_candidate(node.state)
                if candidate is not None:
                    terminal_records.append((node, candidate))
                    complete_frontier = self._update_complete_frontier(
                        complete_frontier,
                        candidate,
                    )
                    self.stats.complete_candidates += 1
                continue

            if len(node.state.key_path) >= self.max_events:
                node.transitions = ()
                continue

            node.transitions = self._build_dp_transitions(
                node.state,
                complete_frontier,
            )
            for transition in node.transitions:
                counter = self._enqueue_dp_dag(
                    pq,
                    counter,
                    dp_frontier,
                    transition.successor,
                    node,
                    transition,
                )

        if pq and expansions >= self.max_states:
            self.stats.hit_max_states = True

        pareto_terminals = [
            (node, candidate)
            for node, candidate in terminal_records
            if not any(
                self._candidate_dominates(other, candidate)
                for other in complete_frontier
            )
        ]
        history_memo: Dict[int, Tuple[int, ...]] = {}
        # Warm starts tighten only safe objective bounds during certified local
        # recovery.  They must not bypass the canonical history enumeration,
        # otherwise an old objective-equivalent plan can displace the plan that
        # a from-scratch solve selects under the deterministic Top-k order.
        complete: List[Candidate] = (
            list(seed_candidates)
            if self.include_initial_candidates_in_output
            else []
        )
        for node, _ in pareto_terminals:
            for history_id in self._dag_top_histories(
                node,
                initial_node,
                history_memo,
            ):
                candidate = self._build_candidate(
                    self._state_with_history(node.state, history_id)
                )
                if candidate is not None:
                    complete.append(candidate)

        # A truncated run may deactivate a predecessor before its replacement
        # reaches a terminal. Preserve already observed feasible plans as a
        # conservative partial-result fallback.
        if not complete and terminal_records:
            complete = [candidate for _, candidate in terminal_records]

        self.last_complete_frontier = select_top_k_candidates(
            complete_frontier,
            max(1, len(complete_frontier)),
            pareto_only=True,
            numeric_tolerance=self.numeric_tolerance,
        )
        self._build_search_certificate(
            (
                node.state
                for _, _, node in pq
                if node.active
            ),
            self.last_complete_frontier,
        )
        return self._pareto_top_k(complete)

    def _compact_fast_state(self, state: SearchState) -> SearchState:
        """Drop route history while retaining every feasibility resource."""
        return SearchState(
            node=state.node,
            picked_mask=state.picked_mask,
            served_mask=state.served_mask,
            load=state.load,
            time_used=state.time_used,
            key_path=(state.node,),
            pickup_times=state.pickup_times,
            route_events=(),
        )

    def _materialize_fast_label(self, label: FastDPLabel) -> SearchState:
        chain: List[FastDPLabel] = []
        current: Optional[FastDPLabel] = label
        while current is not None:
            chain.append(current)
            current = current.parent
        chain.reverse()

        key_path = [chain[0].state.node]
        route_events: List[Tuple[str, str]] = []
        for item in chain[1:]:
            key_path.append(item.state.node)
            if item.event is not None:
                route_events.append(item.event)
        return SearchState(
            node=label.state.node,
            picked_mask=label.state.picked_mask,
            served_mask=label.state.served_mask,
            load=label.state.load,
            time_used=label.state.time_used,
            key_path=tuple(key_path),
            pickup_times=label.state.pickup_times,
            route_events=tuple(route_events),
        )

    def _fast_onboard_resources(self, state: SearchState) -> Tuple[float, ...]:
        onboard_mask = self._onboard_mask(state)
        return tuple(
            state.pickup_times[idx]
            for idx in range(len(self.requests))
            if (onboard_mask >> idx) & 1
        )

    def _make_fast_label(
        self,
        state: SearchState,
        parent: Optional[FastDPLabel],
        event: Optional[Tuple[str, str]],
        depth: int,
    ) -> FastDPLabel:
        compact = self._compact_fast_state(state)
        return FastDPLabel(
            state=compact,
            parent=parent,
            event=event,
            depth=depth,
            onboard_pickup_times=self._fast_onboard_resources(compact),
            # These expensive values are filled only after the lightweight
            # resource-dominance test accepts the label.
            optimistic_served=-1,
            lower_cost=math.nan,
            priority=0.0,
        )

    def _finalize_fast_label(self, label: FastDPLabel) -> None:
        """Compute queue/bound values only for a surviving DP label."""
        state = label.state
        optimistic_served = self._best_possible_served(state)
        remaining = self._cheap_lower_bound(state)
        label.optimistic_served = optimistic_served
        label.lower_cost = (
            math.inf
            if math.isinf(remaining)
            else state.time_used + remaining
        )
        served = self._mask_count(state.served_mask)
        label.priority = (
            -(served * 100000 + optimistic_served * 1000)
            + state.time_used
        )

    def _fast_pareto_bound_prunes(
        self,
        label: FastDPLabel,
        frontier: List[Candidate],
    ) -> bool:
        if not frontier or math.isinf(label.lower_cost):
            return math.isinf(label.lower_cost)
        return any(
            candidate.served_count >= label.optimistic_served
            and candidate.total_cost <= label.lower_cost + self.numeric_tolerance
            and (
                candidate.served_count > label.optimistic_served
                or candidate.total_cost
                < label.lower_cost - self.numeric_tolerance
            )
            for candidate in frontier
        )

    def _fast_label_dominates(
        self,
        left: FastDPLabel,
        right: FastDPLabel,
    ) -> bool:
        """Safe strict dominance using only active onboard resources."""
        if left.state.time_used > right.state.time_used + self.numeric_tolerance:
            return False
        strict = (
            left.state.time_used
            < right.state.time_used - self.numeric_tolerance
        )
        for left_pickup, right_pickup in zip(
            left.onboard_pickup_times,
            right.onboard_pickup_times,
        ):
            if left_pickup < right_pickup - self.numeric_tolerance:
                return False
            if left_pickup > right_pickup + self.numeric_tolerance:
                strict = True
        return strict

    def _enqueue_fast_label(
        self,
        pq: List[Tuple[float, int, FastDPLabel]],
        counter: int,
        frontier: Dict[Tuple[str, int, int, int], List[FastDPLabel]],
        label: FastDPLabel,
    ) -> int:
        signature = self._state_signature(label.state)
        labels = [old for old in frontier.setdefault(signature, []) if old.active]

        for old in labels:
            if (
                abs(old.state.time_used - label.state.time_used)
                <= self.numeric_tolerance
                and all(
                    abs(left - right) <= self.numeric_tolerance
                    for left, right in zip(
                        old.onboard_pickup_times,
                        label.onboard_pickup_times,
                    )
                )
            ):
                self.stats.dp_pruned_states += 1
                return counter
            if self._fast_label_dominates(old, label):
                self.stats.dp_pruned_states += 1
                return counter

        survivors: List[FastDPLabel] = []
        for old in labels:
            if self._fast_label_dominates(label, old):
                old.active = False
                self.stats.dp_pruned_states += 1
            else:
                survivors.append(old)
        self._finalize_fast_label(label)
        survivors.append(label)
        frontier[signature] = survivors

        heapq.heappush(pq, (label.priority, counter, label))
        self.stats.generated_states += 1
        self.stats.peak_open_size = max(self.stats.peak_open_size, len(pq))
        return counter + 1

    def _solve_exact_fast_dp(self) -> List[Candidate]:
        """Exact static resource-label DP without propagating route histories."""
        if self.time_dependent_shortest_path is not None or self.initial_onboard_names:
            # Dynamic continuation needs multiple physical route histories.
            return self._solve_bundled_dp()

        search_started = time.monotonic()
        initial_state = SearchState(
            node=self.source,
            picked_mask=0,
            served_mask=0,
            load=0,
            time_used=self.initial_time,
            key_path=(self.source,),
            pickup_times=tuple([-1.0] * len(self.requests)),
            route_events=(),
        )
        initial = self._make_fast_label(initial_state, None, None, 0)
        pq: List[Tuple[float, int, FastDPLabel]] = []
        frontier: Dict[Tuple[str, int, int, int], List[FastDPLabel]] = {}
        counter = self._enqueue_fast_label(pq, 0, frontier, initial)
        complete: List[Candidate] = []
        complete_frontier: List[Candidate] = []
        expansions = 0

        while pq and expansions < self.max_states:
            if time.monotonic() - search_started > self.search_timeout_seconds:
                self.stats.timed_out = True
                break
            _, _, label = heapq.heappop(pq)
            self.stats.popped_states += 1
            if not label.active:
                continue
            state = label.state

            if self.enable_pruning and self._fast_pareto_bound_prunes(
                label,
                complete_frontier,
            ):
                label.active = False
                self._record_pareto_bound_prune()
                continue

            expansions += 1
            self.stats.expanded_states += 1
            onboard_mask = self._onboard_mask(state)
            if state.node == self.target and onboard_mask == 0 and state.load == 0:
                candidate = self._build_candidate(self._materialize_fast_label(label))
                if candidate is not None:
                    complete.append(candidate)
                    complete_frontier = self._update_complete_frontier(
                        complete_frontier,
                        candidate,
                    )
                    self.stats.complete_candidates += 1
                continue

            if label.depth >= self.max_events:
                continue

            for transition in self._build_dp_transitions(state, complete):
                successor = transition.successor
                event = successor.route_events[-1] if successor.route_events else None
                next_label = self._make_fast_label(
                    successor,
                    label,
                    event,
                    label.depth + 1,
                )
                counter = self._enqueue_fast_label(
                    pq,
                    counter,
                    frontier,
                    next_label,
                )

        if pq and expansions >= self.max_states:
            self.stats.hit_max_states = True
        self.last_complete_frontier = select_top_k_candidates(
            complete_frontier,
            max(1, len(complete_frontier)),
            pareto_only=True,
            numeric_tolerance=self.numeric_tolerance,
        )
        self._build_search_certificate(
            (
                label.state
                for _, _, label in pq
                if label.active
            ),
            self.last_complete_frontier,
        )
        return self._pareto_top_k(complete)

    def _initial_insertion_state(self) -> Optional[SearchState]:
        """Create the empty route ``source -> target`` for static KMOR."""
        pickup_times = tuple([-1.0] * len(self.requests))
        initial = SearchState(
            node=self.source,
            picked_mask=0,
            served_mask=0,
            load=0,
            time_used=self.initial_time,
            key_path=(self.source, self.target),
            pickup_times=pickup_times,
            route_events=(("target", ""),),
        )
        return self._replay_insertion_route(initial.key_path, initial.route_events)

    def _replay_insertion_route(
        self,
        key_path: Tuple[str, ...],
        route_events: Tuple[Tuple[str, str], ...],
    ) -> Optional[SearchState]:
        """Replay an inserted route through the canonical constraint checks."""
        cache_key = (key_path, route_events)
        if cache_key in self._insertion_replay_cache:
            return self._insertion_replay_cache[cache_key]

        def reject() -> Optional[SearchState]:
            self._insertion_replay_cache[cache_key] = None
            return None

        if not key_path or len(key_path) != len(route_events) + 1:
            return reject()

        pickup_times = [-1.0] * len(self.requests)
        picked_mask = 0
        for idx, req in enumerate(self.requests):
            if req.name in self.initial_onboard_names:
                picked_mask |= 1 << idx
                pickup_times[idx] = self.initial_pickup_times.get(
                    req.name, self.initial_time
                )
        load = sum(
            req.demand
            for idx, req in enumerate(self.requests)
            if (picked_mask >> idx) & 1
        )
        state = SearchState(
            node=key_path[0],
            picked_mask=picked_mask,
            served_mask=0,
            load=load,
            time_used=self.initial_time,
            key_path=(key_path[0],),
            pickup_times=tuple(pickup_times),
            route_events=(),
        )
        request_index = {req.name: idx for idx, req in enumerate(self.requests)}

        for node, event in zip(key_path[1:], route_events):
            kind, name = event
            if kind == "pickup":
                req_idx = request_index.get(name)
                if req_idx is None or node != self.requests[req_idx].pickup:
                    return reject()
                state = self._transition_pickup(state, req_idx)
            elif kind == "dropoff":
                req_idx = request_index.get(name)
                if req_idx is None or node != self.requests[req_idx].dropoff:
                    return reject()
                state = self._transition_dropoff(state, req_idx)
            elif kind == "target":
                if node != self.target:
                    return reject()
                state = self._transition_target(state)
            else:
                return reject()
            if state is None:
                return reject()
        self._insertion_replay_cache[cache_key] = state
        return state

    def _naive_route_cost(self, nodes: List[str]) -> float:
        total = 0.0
        for left, right in zip(nodes, nodes[1:]):
            distance = self.dist(left, right)
            if math.isinf(distance):
                return math.inf
            total += distance
        return total

    def _dp_insertion_options(
        self,
        state: SearchState,
        req_idx: int,
    ) -> Tuple[Tuple[int, int, float], ...]:
        """Evaluate all pair-insertion positions with prefix/suffix reuse.

        For a route ``v_0 ... v_m``, the first insertion table stores the
        route cost after inserting the pickup at every gap.  The second table
        adds the dropoff by replacing only one edge.  Thus each position is
        evaluated from cached partial costs instead of rebuilding and
        rescanning the complete route.
        """
        cache_key = (tuple(state.key_path), req_idx)
        cached = self._insertion_cost_cache.get(cache_key)
        if cached is not None:
            self.stats.dp_insertion_cache_hits += 1
            return cached[1]

        nodes = list(state.key_path)
        req = self.requests[req_idx]
        self.stats.dp_insertion_evaluations += 1
        if len(nodes) < 2:
            return ()

        edge_costs = []
        prefix = [0.0]
        for left, right in zip(nodes, nodes[1:]):
            distance = self.dist(left, right)
            if math.isinf(distance):
                return ()
            edge_costs.append(distance)
            prefix.append(prefix[-1] + distance)
        base_cost = prefix[-1]
        options: List[Tuple[int, int, float]] = []

        # The target remains the final stop. p and q are positions in the
        # route after pickup insertion; q is the dropoff position.
        for p in range(1, len(nodes)):
            before = nodes[p - 1]
            after = nodes[p]
            pickup_cost = (
                base_cost
                - edge_costs[p - 1]
                + self.dist(before, req.pickup)
                + self.dist(req.pickup, after)
            )
            if math.isinf(pickup_cost):
                continue
            after_pickup = nodes[:p] + [req.pickup] + nodes[p:]
            for q in range(p + 1, len(after_pickup)):
                self.stats.dp_insertion_position_checks += 1
                left = after_pickup[q - 1]
                right = after_pickup[q]
                drop_cost = (
                    pickup_cost
                    - self.dist(left, right)
                    + self.dist(left, req.dropoff)
                    + self.dist(req.dropoff, right)
                )
                if not math.isinf(drop_cost):
                    options.append((p, q, drop_cost))

        self._insertion_cost_cache[cache_key] = (tuple(edge_costs), tuple(options))
        return tuple(options)

    def _naive_insertion_options(
        self,
        state: SearchState,
        req_idx: int,
    ) -> Iterable[Tuple[int, int, float]]:
        """Reference evaluator: rebuild and rescan each inserted route."""
        nodes = list(state.key_path)
        req = self.requests[req_idx]
        self.stats.naive_insertion_evaluations += 1
        for p in range(1, len(nodes)):
            after_pickup = nodes[:p] + [req.pickup] + nodes[p:]
            for q in range(p + 1, len(after_pickup)):
                self.stats.naive_insertion_position_checks += 1
                new_nodes = after_pickup[:q] + [req.dropoff] + after_pickup[q:]
                cost = self._naive_route_cost(new_nodes)
                if not math.isinf(cost):
                    yield p, q, cost

    def _inserted_route(
        self,
        state: SearchState,
        req_idx: int,
        pickup_position: int,
        dropoff_position: int,
    ) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, str], ...]]:
        req = self.requests[req_idx]
        nodes = list(state.key_path)
        events = list(state.route_events)
        nodes = nodes[:pickup_position] + [req.pickup] + nodes[pickup_position:]
        event_position = pickup_position - 1
        events = (
            events[:event_position]
            + [("pickup", req.name)]
            + events[event_position:]
        )
        nodes = (
            nodes[:dropoff_position]
            + [req.dropoff]
            + nodes[dropoff_position:]
        )
        event_position = dropoff_position - 1
        events = (
            events[:event_position]
            + [("dropoff", req.name)]
            + events[event_position:]
        )
        return tuple(nodes), tuple(events)

    def _insertion_compatible_upper_bound(self, state: SearchState) -> int:
        upper = self._mask_count(state.served_mask)
        used = state.picked_mask | state.served_mask
        for req_idx in range(len(self.requests)):
            if (used >> req_idx) & 1:
                continue
            if self._can_pick(req_idx):
                upper += 1
        if self.target_served_count is not None:
            upper = min(upper, self.target_served_count)
        return upper

    def _insertion_served_upper_bound(self, state: SearchState) -> int:
        """Safe, insertion-aware coverage upper bound.

        A remaining request is counted only if at least one pickup/dropoff
        insertion can be replayed without violating the canonical constraints.
        This is still optimistic because requests are tested independently.
        """
        cache_key = (state.key_path, state.route_events)
        cached = self._insertion_upper_cache.get(cache_key)
        if cached is not None:
            return cached
        upper = self._mask_count(state.served_mask)
        used = state.picked_mask | state.served_mask
        for req_idx in range(len(self.requests)):
            if (used >> req_idx) & 1 or not self._can_pick(req_idx):
                continue
            options = (
                self._dp_insertion_options(state, req_idx)
                if self.enable_dp
                else self._naive_insertion_options(state, req_idx)
            )
            for p, q, route_cost in options:
                if route_cost > self.time_budget + self.numeric_tolerance:
                    continue
                key_path, route_events = self._inserted_route(
                    state, req_idx, p, q
                )
                if self._replay_insertion_route(key_path, route_events) is not None:
                    upper += 1
                    break
        if self.target_served_count is not None:
            upper = min(upper, self.target_served_count)
        self._insertion_upper_cache[cache_key] = upper
        return upper

    def _insertion_priority(self, state: SearchState) -> float:
        served = self._mask_count(state.served_mask)
        optimistic = self._insertion_compatible_upper_bound(state)
        return -(served * 100000 + optimistic * 1000) + state.time_used

    def _insertion_pareto_bound_prunes(
        self,
        state: SearchState,
        frontier: List[Candidate],
    ) -> bool:
        if not frontier:
            return False
        lower_cost = state.time_used
        upper_served = self._insertion_served_upper_bound(state)
        return any(
            candidate.served_count >= upper_served
            and candidate.total_cost <= lower_cost + self.numeric_tolerance
            and (
                candidate.served_count > upper_served
                or candidate.total_cost < lower_cost - self.numeric_tolerance
            )
            for candidate in frontier
        )

    def _insertion_successors(self, state: SearchState) -> Iterable[SearchState]:
        used = state.picked_mask | state.served_mask
        for req_idx, req in enumerate(self.requests):
            if (used >> req_idx) & 1:
                continue
            if not self._can_pick(req_idx):
                continue
            if self.enable_dp:
                options = self._dp_insertion_options(state, req_idx)
            else:
                options = self._naive_insertion_options(state, req_idx)
            for p, q, route_cost in options:
                if route_cost > self.time_budget + self.numeric_tolerance:
                    continue
                key_path, route_events = self._inserted_route(state, req_idx, p, q)
                successor = self._replay_insertion_route(key_path, route_events)
                if successor is not None:
                    yield successor

    def _solve_insertion_routes(self) -> List[Candidate]:
        """Static KMOR search over pair insertions.

        The search frontier is identical for Full and NoPruning; only the
        Pareto/bound checks differ. ``enable_dp`` selects the insertion cost
        evaluator, not a second resource-state search algorithm.
        """
        started = time.monotonic()
        initial = self._initial_insertion_state()
        if initial is None:
            self._build_search_certificate([], [])
            return []

        pq: List[Tuple[float, int, SearchState]] = []
        heapq.heappush(pq, (self._insertion_priority(initial), 0, initial))
        self.stats.generated_states += 1
        self.stats.peak_open_size = 1
        counter = 1
        visited: Set[Tuple[Tuple[str, ...], Tuple[Tuple[str, str], ...]]] = set()
        complete: List[Candidate] = []
        complete_frontier: List[Candidate] = []
        expansions = 0

        while pq and expansions < self.max_states:
            if time.monotonic() - started > self.search_timeout_seconds:
                self.stats.timed_out = True
                break
            _, _, state = heapq.heappop(pq)
            self.stats.popped_states += 1
            identity = (state.key_path, state.route_events)
            if identity in visited:
                continue
            visited.add(identity)

            candidate = self._build_candidate(state)

            # Once the optimistic served upper bound equals the current
            # route, this route cannot improve its coverage by later
            # insertions. Only then is it safe to discard a route already
            # dominated by the current Pareto frontier.
            if (
                candidate is not None
                and self.enable_pruning
                and (
                    len(complete_frontier) >= max(1, self.k)
                    or self.target_served_count is not None
                )
            ):
                upper_served = self._insertion_served_upper_bound(state)
                if (
                    self.target_served_count is not None
                    and upper_served < self.target_served_count
                ):
                    self._record_state_bound_prune()
                    continue
                if (
                    upper_served <= candidate.served_count
                    and self._insertion_pareto_bound_prunes(
                        state, complete_frontier
                    )
                ):
                    self._record_pareto_bound_prune()
                    continue

            # Count a state as expanded only after all safe pruning checks
            # have passed.  This keeps examined_partial_routes consistent
            # with the actual search work performed by the variant.
            expansions += 1
            self.stats.expanded_states += 1

            if candidate is not None:
                complete.append(candidate)
                complete_frontier = self._update_complete_frontier(
                    complete_frontier, candidate
                )
                self.stats.complete_candidates += 1

            if len(state.key_path) >= self.max_events:
                continue
            for successor in self._insertion_successors(state):
                heapq.heappush(
                    pq,
                    (self._insertion_priority(successor), counter, successor),
                )
                self.stats.generated_states += 1
                counter += 1
                self.stats.peak_open_size = max(self.stats.peak_open_size, len(pq))

        if pq and expansions >= self.max_states:
            self.stats.hit_max_states = True
        self.last_complete_frontier = select_top_k_candidates(
            complete_frontier,
            max(1, len(complete_frontier)),
            pareto_only=True,
            numeric_tolerance=self.numeric_tolerance,
        )
        # Insertion states use a different route-level coverage bound.  Until
        # that bound is exposed as a certificate primitive, only queue
        # exhaustion can certify this optional experimental solver.
        self._build_search_certificate(
            (state for _, _, state in pq),
            self.last_complete_frontier,
            allow_resource_bounds=False,
        )
        return self._pareto_top_k(complete)

    def _solve_label_search(self) -> List[Candidate]:
        """State-DP solver shared by static and dynamic/time-dependent KMOR."""
        search_started = time.monotonic()
        self._dp_domination_counts.clear()
        self._dp_rank_cache.clear()
        initial_picked_mask = 0
        initial_load = 0
        pickup_times = [-1.0] * len(self.requests)
        for idx, req in enumerate(self.requests):
            if req.name not in self.initial_onboard_names:
                continue
            initial_picked_mask |= 1 << idx
            initial_load += req.demand
            pickup_times[idx] = self.initial_pickup_times.get(req.name, self.initial_time)
        if initial_load > self.capacity:
            self._build_search_certificate([], [])
            return []

        initial = SearchState(
            node=self.source,
            picked_mask=initial_picked_mask,
            served_mask=0,
            load=initial_load,
            time_used=self.initial_time,
            key_path=(self.source,),
            pickup_times=tuple(pickup_times),
            route_events=(),
        )

        pq: List[Tuple[float, int, SearchState]] = []
        counter = 0
        dp_frontier: Dict[Tuple[str, int, int, int], List[SearchState]] = {}
        counter = self._push_state(pq, counter, initial, dp_frontier)
        complete: List[Candidate] = []
        complete_frontier: List[Candidate] = []
        expansions = 0

        while pq and expansions < self.max_states:
            if time.monotonic() - search_started > self.search_timeout_seconds:
                self.stats.timed_out = True
                break
            _, _, state = heapq.heappop(pq)
            self.stats.popped_states += 1

            if not self._is_active_dp_label(state, dp_frontier):
                self.stats.dp_pruned_states += 1
                continue

            if self.enable_pruning and self._pareto_bound_prunes(
                state,
                complete_frontier,
            ):
                self._record_pareto_bound_prune()
                continue

            expansions += 1
            self.stats.expanded_states += 1

            onboard_mask = self._onboard_mask(state)
            if state.node == self.target and onboard_mask == 0 and state.load == 0:
                candidate = self._build_candidate(state)
                if candidate is not None:
                    complete.append(candidate)
                    complete_frontier = self._update_complete_frontier(
                        complete_frontier,
                        candidate,
                    )
                    self.stats.complete_candidates += 1
                continue

            if len(state.key_path) >= self.max_events:
                continue

            target_state = self._transition_target(state)
            if target_state is not None:
                counter = self._push_state(pq, counter, target_state, dp_frontier)

            for idx in range(len(self.requests)):
                if (onboard_mask >> idx) & 1:
                    nxt = self._transition_dropoff(state, idx)
                    if nxt is not None:
                        if self.enable_pruning and self.enable_state_bounds:
                            lower_bound = self._cheap_lower_bound(nxt)
                            if math.isinf(lower_bound) or nxt.time_used + lower_bound > self.time_budget:
                                self._record_state_bound_prune()
                                continue
                        counter = self._push_state(pq, counter, nxt, dp_frontier)

            for idx in range(len(self.requests)):
                nxt = self._transition_pickup(state, idx)
                if nxt is None:
                    continue
                if self.enable_pruning and self.enable_state_bounds:
                    lower_bound = self._cheap_lower_bound(nxt)
                    if math.isinf(lower_bound) or nxt.time_used + lower_bound > self.time_budget:
                        self._record_state_bound_prune()
                        continue
                    request_bound = self._pickup_request_bound(nxt, idx)
                    lower_bound = max(lower_bound, request_bound)
                    if math.isinf(lower_bound) or nxt.time_used + lower_bound > self.time_budget:
                        self._record_state_bound_prune()
                        continue
                    if self._should_run_expensive_bound(nxt, complete, lower_bound):
                        expensive_bound = self._expensive_lower_bound(nxt)
                        if math.isinf(expensive_bound) or nxt.time_used + expensive_bound > self.time_budget:
                            self._record_state_bound_prune()
                            continue
                counter = self._push_state(pq, counter, nxt, dp_frontier)

        if pq and expansions >= self.max_states:
            self.stats.hit_max_states = True
        self.last_complete_frontier = select_top_k_candidates(
            complete_frontier,
            max(1, len(complete_frontier)),
            pareto_only=True,
            numeric_tolerance=self.numeric_tolerance,
        )
        self._build_search_certificate(
            (
                state
                for _, _, state in pq
                if self._is_active_dp_label(state, dp_frontier)
            ),
            self.last_complete_frontier,
        )
        return self._pareto_top_k(complete)

    def solve(self) -> List[Candidate]:
        self.search_certificate = SearchCertificate()
        if self.use_insertion_solver:
            return self._solve_insertion_routes()
        if self.enable_dp:
            if self.dp_mode == "exact_fast":
                return self._solve_exact_fast_dp()
            return self._solve_bundled_dp()
        return self._solve_label_search()


def dpkmor_algorithm(
    graph: Graph,
    source: str,
    target: str,
    requests: List[Request],
    **params: Any,
) -> List[Tuple[List[str], float, int, float, float, int, List[str], List[str], List[Tuple[str, float, float, float]]]]:
    planner = RideSharePlanner(graph, source, target, requests, **params)
    plans = planner.solve()
    output = []
    for item in plans:
        output.append(
            (
                item.full_path,
                item.score,
                item.served_count,
                item.total_cost,
                item.total_time,
                item.hops,
                item.served_requests,
                item.key_path,
                item.pickup_info,
            )
        )
    return output


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="DPKMOR - 拼车 top-k 动态规划实验")
    ap.add_argument("--input", required=True, help="边文件路径")
    ap.add_argument("--delimiter", default="space", choices=["space", "tab", "comma"])
    ap.add_argument("--ignore-first-col", action="store_true", help="边文件首列为 edge id")
    ap.add_argument("--directed", action="store_true", help="是否有向图")
    ap.add_argument("--source", required=True, help="车辆起点")
    ap.add_argument("--target", required=True, help="车辆终点")
    ap.add_argument("--requests-file", required=True, help="乘客请求文件")
    ap.add_argument(
        "--param",
        nargs="*",
        default=[],
        help="算法参数，例如 k=5 capacity=4 time_budget=100 detour_ratio=1.8",
    )
    args = ap.parse_args(argv)

    print("读取路网...")
    graph = read_edge_list(
        args.input,
        delimiter=args.delimiter,
        directed=args.directed,
        ignore_first_col=args.ignore_first_col,
    )
    print(f"节点数: {len(list(graph.nodes()))}")

    print("读取乘客请求...")
    requests = parse_requests_file(args.requests_file)
    print(f"请求数: {len(requests)}")
    for req in requests[:5]:
        print(
            f"  {req.name}: {req.pickup} -> {req.dropoff}, demand={req.demand}, request_time={req.request_time}, wait_limit={req.wait_limit}"
        )
    if len(requests) > 5:
        print(f"  ... 其余 {len(requests) - 5} 个请求省略")

    params = parse_kv_list(args.param)

    print("开始搜索 top-k 拼车方案...")
    start = time.time()
    results = dpkmor_algorithm(graph, args.source, args.target, requests, **params)
    elapsed = time.time() - start

    if not results:
        print("未找到满足约束的拼车方案。")
        return 1

    print(f"\n运行时间: {elapsed:.3f}s")
    print(f"输出方案数: {len(results)}")
    print("=" * 72)

    for i, (path, score, served, cost, time_val, hops, served_names, key_path, pickup_info) in enumerate(results, start=1):
        print(f"Top-{i}")
        print(f"  服务乘客数: {served}")
        print(f"  总成本/时间: {cost:.2f}")
        print(f"  路网跳数: {hops}")
        print(f"  服务乘客: {', '.join(served_names) if served_names else '无'}")
        if pickup_info:
            print("  接驾明细:")
            for name, request_time, pickup_time, wait_time in pickup_info:
                print(
                    f"    {name}: request_time={request_time:.2f}, pickup_time={pickup_time:.2f}, wait_time={wait_time:.2f}"
                )
        print(f"  关键事件路径: {' -> '.join(key_path)}")
        print(f"  展开路网路径: {' -> '.join(path)}")
        print(f"  综合评分: {score:.2f}")
        print("-" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
