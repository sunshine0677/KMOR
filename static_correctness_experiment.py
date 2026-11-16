#!/usr/bin/env python3
"""Correctness verification for static KMOR.

The experiment compares an independent exhaustive event-order enumerator with
KPrun-Search (event-state search without DP reuse) and DPKMOR (the same search
with resource-safe state-DP reuse).
It writes reproducible query instances and both per-query and aggregate CSVs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from Dpkmor import (
    BUNDLED_IMPLEMENTATION_VERSION,
    Candidate,
    Graph,
    Request,
    RideSharePlanner,
    read_edge_list,
)
from prepare_paper_datasets import generate_requests_for_path


ALGORITHM_VERSION = "static-correctness-physical-path-v6.1"
COMPLETE = "COMPLETE"
TIMEOUT = "TIMEOUT"
STATE_LIMIT = "STATE_LIMIT"
OOM = "OOM"


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class QueryInstance:
    query_id: str
    dataset: str
    request_size: int
    seed: int
    source: str
    target: str
    time_budget: float
    requests: Tuple[Request, ...]


@dataclass(frozen=True)
class ExactState:
    node: str
    picked_mask: int
    served_mask: int
    load: int
    time_used: float
    pickup_times: Tuple[float, ...]
    key_path: Tuple[str, ...]
    route_events: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class RouteRecord:
    served_count: int
    cost: float
    served_requests: Tuple[str, ...]
    route_events: Tuple[Tuple[str, str], ...]
    key_path: Tuple[str, ...]
    hops: int
    full_path: Optional[Tuple[str, ...]] = None


@dataclass
class SolverResult:
    method: str
    status: str
    frontier: List[RouteRecord]
    routes: List[RouteRecord]
    preprocess_time: float
    search_time: float
    expanded_states: int
    generated_states: int
    feasibility_checks: int
    feasibility_pruned: int
    heuristic_pruned: int
    dp_pruned: int
    peak_open_size: int
    hit_max_states: bool
    timed_out: bool


METHOD_FIELDS = [
    "query_id",
    "query_fingerprint",
    "run_fingerprint",
    "dataset",
    "request_size",
    "seed",
    "method",
    "status",
    "k_max",
    "frontier_size",
    "returned_routes",
    "best_served",
    "best_cost",
    "preprocess_time",
    "search_time",
    "expanded_states",
    "generated_states",
    "feasibility_checks",
    "feasibility_pruned",
    "heuristic_pruned",
    "dp_pruned",
    "peak_open_size",
    "hit_max_states",
    "timed_out",
    "frontier_json",
    "routes_json",
]

COMPARISON_FIELDS = [
    "query_id",
    "query_fingerprint",
    "dataset",
    "request_size",
    "seed",
    "k",
    "comparison",
    "reference_method",
    "method",
    "reference_status",
    "method_status",
    "comparable",
    "frontier_match",
    "topk_objective_match",
    "coverage_match",
    "cost_match",
    "service_set_match",
    "route_match",
    "key_path_match",
    "physical_path_match",
    "physical_path_source",
    "concrete_topk_match",
]

SUMMARY_FIELDS = [
    "dataset",
    "request_size",
    "k",
    "comparison",
    "total_queries",
    "completed_pairs",
    "unresolved_pairs",
    "timeout_pairs",
    "oom_pairs",
    "state_limit_pairs",
    "frontier_match_rate",
    "topk_objective_match_rate",
    "coverage_match_rate",
    "cost_match_rate",
    "service_set_match_rate",
    "route_match_rate",
    "key_path_match_rate",
    "physical_path_compared_pairs",
    "physical_path_recorded_pairs",
    "physical_path_match_rate",
    "concrete_topk_match_rate",
]


def parse_int_list(raw: str) -> List[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("integer lists must contain positive values")
    return sorted(set(values))


def parse_name_list(raw: str) -> List[str]:
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def load_configs(path: Path) -> List[DatasetConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [DatasetConfig(**item) for item in payload["datasets"]]


def request_to_dict(request: Request) -> Dict[str, object]:
    return {
        "name": request.name,
        "pickup": request.pickup,
        "dropoff": request.dropoff,
        "demand": request.demand,
        "request_time": request.request_time,
        "wait_limit": request.wait_limit,
    }


def query_to_dict(query: QueryInstance) -> Dict[str, object]:
    return {
        "query_id": query.query_id,
        "dataset": query.dataset,
        "request_size": query.request_size,
        "seed": query.seed,
        "source": query.source,
        "target": query.target,
        "time_budget": query.time_budget,
        "requests": [request_to_dict(request) for request in query.requests],
    }


def query_from_dict(payload: Dict[str, object]) -> QueryInstance:
    requests = tuple(Request(**item) for item in payload["requests"])
    return QueryInstance(
        query_id=str(payload["query_id"]),
        dataset=str(payload["dataset"]),
        request_size=int(payload["request_size"]),
        seed=int(payload["seed"]),
        source=str(payload["source"]),
        target=str(payload["target"]),
        time_budget=float(payload["time_budget"]),
        requests=requests,
    )


def query_fingerprint(query: QueryInstance) -> str:
    encoded = json.dumps(
        query_to_dict(query),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def run_fingerprint(
    query: QueryInstance,
    capacity: int,
    detour_ratio: float,
    k_max: int,
    max_states: int,
    exact_max_states: int,
    timeout_seconds: float,
    cost_tolerance: float,
    dp_mode: str = "exact_fast",
) -> str:
    payload = {
        "algorithm_version": ALGORITHM_VERSION,
        "query": query_fingerprint(query),
        "capacity": capacity,
        "detour_ratio": detour_ratio,
        "k_max": k_max,
        "max_states": max_states,
        "exact_max_states": exact_max_states,
        "timeout_seconds": timeout_seconds,
        "cost_tolerance": cost_tolerance,
    }
    # Preserve fingerprints produced by the original exact_fast experiment,
    # while ensuring bundled runs can never resume exact_fast result rows.
    if dp_mode != "exact_fast":
        payload["dp_mode"] = dp_mode
        payload["dp_implementation"] = BUNDLED_IMPLEMENTATION_VERSION
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def write_query_bank(path: Path, queries: Sequence[QueryInstance]) -> None:
    text = "\n".join(
        json.dumps(query_to_dict(query), sort_keys=True)
        for query in queries
    )
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def read_query_bank(path: Path) -> List[QueryInstance]:
    queries = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            queries.append(query_from_dict(json.loads(line)))
    return queries


def tuple_request_to_request(
    item: Tuple[str, str, str, int, float, float],
    name: str,
) -> Request:
    _, pickup, dropoff, demand, request_time, wait_limit = item
    return Request(
        name=name,
        pickup=str(pickup),
        dropoff=str(dropoff),
        demand=int(demand),
        request_time=float(request_time),
        wait_limit=float(wait_limit),
    )


def generate_query_bank(
    workdir: Path,
    configs: Sequence[DatasetConfig],
    request_sizes: Sequence[int],
    queries_per_size: int,
    seed: int,
) -> List[QueryInstance]:
    queries: List[QueryInstance] = []
    max_size = max(request_sizes)
    master_pool_size = max(40, max_size * 3)

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

        pool_rng = random.Random(seed + dataset_index * 1_000_003)
        master_pool = generate_requests_for_path(
            graph,
            reference_path,
            float(shortest_cost),
            master_pool_size,
            pool_rng,
            profile="effectiveness",
        )

        for request_size in request_sizes:
            for query_number in range(1, queries_per_size + 1):
                query_seed = (
                    seed
                    + dataset_index * 1_000_003
                    + request_size * 10_007
                    + query_number * 101
                )
                rng = random.Random(query_seed)
                selected = rng.sample(master_pool, request_size)
                requests = tuple(
                    tuple_request_to_request(
                        item,
                        f"q{query_number}_r{request_index}",
                    )
                    for request_index, item in enumerate(selected, start=1)
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


def route_rank(
    route: RouteRecord,
    tolerance: float = 1e-6,
) -> Tuple[object, ...]:
    cost_rank: object
    if tolerance > 0:
        cost_rank = round(route.cost / tolerance)
    else:
        cost_rank = route.cost
    return (
        -route.served_count,
        cost_rank,
        route.hops,
        route.route_events,
        route.key_path,
    )


def route_identity(route: RouteRecord) -> Tuple[object, ...]:
    return (route.route_events, route.key_path, route.served_requests, route.full_path)


def dominates(left: RouteRecord, right: RouteRecord, tolerance: float) -> bool:
    return (
        left.served_count >= right.served_count
        and left.cost <= right.cost + tolerance
        and (
            left.served_count > right.served_count
            or left.cost < right.cost - tolerance
        )
    )


def update_exact_frontier(
    frontier: Sequence[RouteRecord],
    candidate: RouteRecord,
    tolerance: float,
    max_ties: int,
) -> List[RouteRecord]:
    if any(dominates(old, candidate, tolerance) for old in frontier):
        return list(frontier)

    kept = [old for old in frontier if not dominates(candidate, old, tolerance)]
    if any(route_identity(old) == route_identity(candidate) for old in kept):
        return kept
    kept.append(candidate)

    grouped: Dict[Tuple[int, int], List[RouteRecord]] = {}
    scale = max(tolerance, 1e-12)
    for route in kept:
        objective = (route.served_count, round(route.cost / scale))
        grouped.setdefault(objective, []).append(route)

    limited = []
    for routes in grouped.values():
        limited.extend(
            sorted(routes, key=lambda route: route_rank(route, tolerance))[:max_ties]
        )
    return sorted(limited, key=lambda route: route_rank(route, tolerance))


def candidate_to_record(candidate: Candidate) -> RouteRecord:
    return RouteRecord(
        served_count=candidate.served_count,
        cost=float(candidate.total_cost),
        served_requests=tuple(sorted(candidate.served_requests)),
        route_events=tuple(candidate.route_events),
        key_path=tuple(candidate.key_path),
        hops=int(candidate.hops),
        full_path=tuple(candidate.full_path),
    )


class ExhaustiveKMORSolver:
    """Independent exhaustive enumerator over pickup/dropoff event orders."""

    def __init__(
        self,
        source: str,
        target: str,
        requests: Sequence[Request],
        path_cache: Dict[Tuple[str, str], Tuple[List[str], float]],
        capacity: int,
        time_budget: float,
        detour_ratio: float,
        max_states: int,
        timeout_seconds: float,
        max_ties: int,
        numeric_tolerance: float,
    ):
        self.source = source
        self.target = target
        self.requests = list(requests)
        self.path_cache = path_cache
        self.capacity = capacity
        self.time_budget = time_budget
        self.detour_ratio = detour_ratio
        self.max_states = max_states
        self.timeout_seconds = timeout_seconds
        self.max_ties = max_ties
        self.tolerance = numeric_tolerance
        self.expanded = 0
        self.generated = 0
        self.feasibility_checks = 0
        self.feasibility_pruned = 0
        self.peak_open = 0
        self.hit_max_states = False
        self.timed_out = False
        self.compatible_mask = self._build_compatible_mask()

    def dist(self, source: str, target: str) -> float:
        item = self.path_cache.get((source, target))
        return float(item[1]) if item is not None else math.inf

    def _build_compatible_mask(self) -> int:
        mask = 0
        for index, request in enumerate(self.requests):
            direct = self.dist(request.pickup, request.dropoff)
            earliest_pickup = self.dist(self.source, request.pickup)
            minimum_complete = (
                earliest_pickup
                + direct
                + self.dist(request.dropoff, self.target)
            )
            if (
                request.demand <= self.capacity
                and not math.isinf(direct)
                and not math.isinf(earliest_pickup)
                and earliest_pickup - request.request_time
                <= request.wait_limit + self.tolerance
                and minimum_complete <= self.time_budget + self.tolerance
            ):
                mask |= 1 << index
        return mask

    def _onboard_mask(self, state: ExactState) -> int:
        return state.picked_mask & ~state.served_mask

    def _detour_feasible(self, state: ExactState, at_time: float) -> bool:
        onboard = self._onboard_mask(state)
        for index, request in enumerate(self.requests):
            if ((onboard >> index) & 1) == 0:
                continue
            pickup_time = state.pickup_times[index]
            direct = self.dist(request.pickup, request.dropoff)
            if (
                pickup_time < 0
                or at_time - pickup_time
                > self.detour_ratio * direct + self.tolerance
            ):
                return False
        return True

    def _full_hops(self, key_path: Sequence[str]) -> int:
        hops = 0
        for source, target in zip(key_path, key_path[1:]):
            item = self.path_cache.get((source, target))
            if item is None or not item[0]:
                return 10**18
            hops += max(0, len(item[0]) - 1)
        return hops

    def _full_path(self, key_path: Sequence[str]) -> Optional[Tuple[str, ...]]:
        if not key_path:
            return None
        path = [key_path[0]]
        for source, target in zip(key_path, key_path[1:]):
            segment = self.path_cache.get((source, target))
            if segment is None or not segment[0]:
                return None
            path.extend(segment[0][1:])
        return tuple(path)

    def _terminal_candidate(self, state: ExactState) -> Optional[RouteRecord]:
        if self._onboard_mask(state) != 0 or state.load != 0:
            return None
        travel_time = self.dist(state.node, self.target)
        total_cost = state.time_used + travel_time
        if math.isinf(travel_time) or total_cost > self.time_budget + self.tolerance:
            return None
        if state.node == self.target:
            key_path = state.key_path
            events = state.route_events
        else:
            key_path = state.key_path + (self.target,)
            events = state.route_events + (("target", ""),)
        full_path = self._full_path(key_path)
        if full_path is None:
            return None
        served = tuple(
            sorted(
                request.name
                for index, request in enumerate(self.requests)
                if (state.served_mask >> index) & 1
            )
        )
        return RouteRecord(
            served_count=len(served),
            cost=total_cost,
            served_requests=served,
            route_events=events,
            key_path=key_path,
            hops=self._full_hops(key_path),
            full_path=full_path,
        )

    def solve(self) -> SolverResult:
        timeout_started = time.monotonic()
        measurement_started = time.perf_counter()
        initial = ExactState(
            node=self.source,
            picked_mask=0,
            served_mask=0,
            load=0,
            time_used=0.0,
            pickup_times=tuple(-1.0 for _ in self.requests),
            key_path=(self.source,),
            route_events=(),
        )
        stack = [initial]
        self.generated = 1
        self.peak_open = 1
        frontier: List[RouteRecord] = []

        while stack and self.expanded < self.max_states:
            if time.monotonic() - timeout_started > self.timeout_seconds:
                self.timed_out = True
                break

            state = stack.pop()
            self.expanded += 1
            onboard = self._onboard_mask(state)

            if onboard == 0 and state.load == 0:
                candidate = self._terminal_candidate(state)
                if candidate is not None:
                    frontier = update_exact_frontier(
                        frontier,
                        candidate,
                        self.tolerance,
                        self.max_ties,
                    )
                if state.node == self.target:
                    continue

            if state.picked_mask.bit_count() + state.served_mask.bit_count() >= 2 * len(self.requests):
                continue

            for index, request in enumerate(self.requests):
                if ((onboard >> index) & 1) == 0:
                    continue
                self.feasibility_checks += 1
                travel_time = self.dist(state.node, request.dropoff)
                arrival = state.time_used + travel_time
                if (
                    math.isinf(travel_time)
                    or not self._detour_feasible(state, arrival)
                    or arrival + self.dist(request.dropoff, self.target)
                    > self.time_budget + self.tolerance
                ):
                    self.feasibility_pruned += 1
                    continue
                stack.append(
                    ExactState(
                        node=request.dropoff,
                        picked_mask=state.picked_mask,
                        served_mask=state.served_mask | (1 << index),
                        load=state.load - request.demand,
                        time_used=arrival,
                        pickup_times=state.pickup_times,
                        key_path=state.key_path + (request.dropoff,),
                        route_events=state.route_events + (("dropoff", request.name),),
                    )
                )
                self.generated += 1

            for index, request in enumerate(self.requests):
                if ((self.compatible_mask >> index) & 1) == 0:
                    continue
                if (state.picked_mask >> index) & 1:
                    continue
                self.feasibility_checks += 1
                if state.load + request.demand > self.capacity:
                    self.feasibility_pruned += 1
                    continue
                travel_time = self.dist(state.node, request.pickup)
                arrival = max(
                    state.time_used + travel_time,
                    request.request_time,
                )
                if (
                    math.isinf(travel_time)
                    or not self._detour_feasible(state, arrival)
                    or arrival - request.request_time
                    > request.wait_limit + self.tolerance
                    or arrival + self.dist(request.pickup, self.target)
                    > self.time_budget + self.tolerance
                ):
                    self.feasibility_pruned += 1
                    continue
                pickup_times = list(state.pickup_times)
                pickup_times[index] = arrival
                stack.append(
                    ExactState(
                        node=request.pickup,
                        picked_mask=state.picked_mask | (1 << index),
                        served_mask=state.served_mask,
                        load=state.load + request.demand,
                        time_used=arrival,
                        pickup_times=tuple(pickup_times),
                        key_path=state.key_path + (request.pickup,),
                        route_events=state.route_events + (("pickup", request.name),),
                    )
                )
                self.generated += 1

            self.peak_open = max(self.peak_open, len(stack))

        if stack and self.expanded >= self.max_states:
            self.hit_max_states = True
        status = TIMEOUT if self.timed_out else STATE_LIMIT if self.hit_max_states else COMPLETE
        frontier.sort(key=lambda route: route_rank(route, self.tolerance))
        return SolverResult(
            method="Exhaustive",
            status=status,
            frontier=frontier,
            routes=frontier[: self.max_ties],
            preprocess_time=0.0,
            search_time=time.perf_counter() - measurement_started,
            expanded_states=self.expanded,
            generated_states=self.generated,
            feasibility_checks=self.feasibility_checks,
            feasibility_pruned=self.feasibility_pruned,
            heuristic_pruned=0,
            dp_pruned=0,
            peak_open_size=self.peak_open,
            hit_max_states=self.hit_max_states,
            timed_out=self.timed_out,
        )


def planner_result(
    method: str,
    planner: RideSharePlanner,
    preprocess_time: float,
) -> SolverResult:
    started = time.perf_counter()
    routes = planner.solve()
    search_time = time.perf_counter() - started
    status = (
        TIMEOUT
        if planner.stats.timed_out
        else STATE_LIMIT
        if planner.stats.hit_max_states
        else COMPLETE
    )
    return SolverResult(
        method=method,
        status=status,
        frontier=sorted(
            (candidate_to_record(candidate) for candidate in planner.last_complete_frontier),
            key=lambda route: route_rank(route, planner.numeric_tolerance),
        ),
        routes=sorted(
            (candidate_to_record(candidate) for candidate in routes),
            key=lambda route: route_rank(route, planner.numeric_tolerance),
        ),
        preprocess_time=preprocess_time,
        search_time=search_time,
        expanded_states=planner.stats.expanded_states,
        generated_states=planner.stats.generated_states,
        feasibility_checks=planner.stats.feasibility_checks,
        feasibility_pruned=planner.stats.feasibility_pruned_states,
        heuristic_pruned=planner.stats.heuristic_pruned_states,
        dp_pruned=planner.stats.dp_pruned_states,
        peak_open_size=planner.stats.peak_open_size,
        hit_max_states=planner.stats.hit_max_states,
        timed_out=planner.stats.timed_out,
    )


def run_query(
    graph: Graph,
    query: QueryInstance,
    shortest_path_cache: Dict[Tuple[str, str], Tuple[List[str], float]],
    capacity: int,
    detour_ratio: float,
    k_max: int,
    max_states: int,
    exact_max_states: int,
    timeout_seconds: float,
    numeric_tolerance: float,
    dp_mode: str = "exact_fast",
) -> List[SolverResult]:
    common = {
        "capacity": capacity,
        "k": k_max,
        "time_budget": query.time_budget,
        "detour_ratio": detour_ratio,
        "max_states": max_states,
        "search_timeout_seconds": timeout_seconds,
        "pareto_only": True,
        "enable_compatibility_pruning": False,
        "enable_state_bounds": True,
        "aggressive_served_pruning": False,
        "numeric_tolerance": numeric_tolerance,
        "shortest_path_cache": shortest_path_cache,
        "dp_mode": dp_mode,
    }

    try:
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
        kprun_preprocess = time.perf_counter() - started

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
        dpkmor_preprocess = time.perf_counter() - started

        exact = ExhaustiveKMORSolver(
            query.source,
            query.target,
            query.requests,
            {
                key: (list(path), float(kprun.dist_cache[key]))
                for key, path in kprun.path_cache.items()
            },
            capacity=capacity,
            time_budget=query.time_budget,
            detour_ratio=detour_ratio,
            max_states=exact_max_states,
            timeout_seconds=timeout_seconds,
            max_ties=k_max,
            numeric_tolerance=numeric_tolerance,
        ).solve()

        return [
            exact,
            planner_result("KPrun-Search", kprun, kprun_preprocess),
            planner_result("DPKMOR", dpkmor, dpkmor_preprocess),
        ]
    except MemoryError:
        return [oom_solver_result(method) for method in ["Exhaustive", "KPrun-Search", "DPKMOR"]]


def oom_solver_result(method: str) -> SolverResult:
    return SolverResult(
        method=method,
        status=OOM,
        frontier=[],
        routes=[],
        preprocess_time=math.inf,
        search_time=math.inf,
        expanded_states=0,
        generated_states=0,
        feasibility_checks=0,
        feasibility_pruned=0,
        heuristic_pruned=0,
        dp_pruned=0,
        peak_open_size=0,
        hit_max_states=False,
        timed_out=False,
    )


def record_to_dict(route: RouteRecord) -> Dict[str, object]:
    payload = {
        "served_count": route.served_count,
        "cost": route.cost,
        "served_requests": list(route.served_requests),
        "route_events": [list(event) for event in route.route_events],
        "key_path": list(route.key_path),
        "hops": route.hops,
    }
    if route.full_path is not None:
        payload["full_path"] = list(route.full_path)
    return payload


def record_from_dict(payload: Dict[str, object]) -> RouteRecord:
    return RouteRecord(
        served_count=int(payload["served_count"]),
        cost=float(payload["cost"]),
        served_requests=tuple(str(item) for item in payload["served_requests"]),
        route_events=tuple(
            (str(event[0]), str(event[1])) for event in payload["route_events"]
        ),
        key_path=tuple(str(item) for item in payload["key_path"]),
        hops=int(payload["hops"]),
        full_path=(
            tuple(str(item) for item in payload["full_path"])
            if payload.get("full_path") is not None
            else None
        ),
    )


def routes_to_json(routes: Sequence[RouteRecord]) -> str:
    return json.dumps(
        [record_to_dict(route) for route in routes],
        separators=(",", ":"),
    )


def routes_from_json(raw: str) -> List[RouteRecord]:
    return [record_from_dict(item) for item in json.loads(raw or "[]")]


def solver_result_to_row(
    query: QueryInstance,
    result: SolverResult,
    run_id: str,
    k_max: int,
) -> Dict[str, object]:
    best = result.routes[0] if result.routes else None
    return {
        "query_id": query.query_id,
        "query_fingerprint": query_fingerprint(query),
        "run_fingerprint": run_id,
        "dataset": query.dataset,
        "request_size": query.request_size,
        "seed": query.seed,
        "method": result.method,
        "status": result.status,
        "k_max": k_max,
        "frontier_size": len(result.frontier),
        "returned_routes": len(result.routes),
        "best_served": best.served_count if best else 0,
        "best_cost": best.cost if best else math.inf,
        "preprocess_time": result.preprocess_time,
        "search_time": result.search_time,
        "expanded_states": result.expanded_states,
        "generated_states": result.generated_states,
        "feasibility_checks": result.feasibility_checks,
        "feasibility_pruned": result.feasibility_pruned,
        "heuristic_pruned": result.heuristic_pruned,
        "dp_pruned": result.dp_pruned,
        "peak_open_size": result.peak_open_size,
        "hit_max_states": result.hit_max_states,
        "timed_out": result.timed_out,
        "frontier_json": routes_to_json(result.frontier),
        "routes_json": routes_to_json(result.routes),
    }


def row_to_solver_result(row: Dict[str, str]) -> SolverResult:
    return SolverResult(
        method=row["method"],
        status=row["status"],
        frontier=routes_from_json(row["frontier_json"]),
        routes=routes_from_json(row["routes_json"]),
        preprocess_time=float(row["preprocess_time"]),
        search_time=float(row["search_time"]),
        expanded_states=int(row["expanded_states"]),
        generated_states=int(row["generated_states"]),
        feasibility_checks=int(row["feasibility_checks"]),
        feasibility_pruned=int(row["feasibility_pruned"]),
        heuristic_pruned=int(row["heuristic_pruned"]),
        dp_pruned=int(row["dp_pruned"]),
        peak_open_size=int(row["peak_open_size"]),
        hit_max_states=str(row["hit_max_states"]).lower() == "true",
        timed_out=str(row["timed_out"]).lower() == "true",
    )


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def quantized_cost(value: float, tolerance: float) -> int:
    return round(value / max(tolerance, 1e-12))


def objective_sequence(
    routes: Sequence[RouteRecord],
    k: int,
    tolerance: float,
) -> List[Tuple[int, int]]:
    return [
        (route.served_count, quantized_cost(route.cost, tolerance))
        for route in sorted(routes, key=lambda route: route_rank(route, tolerance))[:k]
    ]


def compare_results(
    query: QueryInstance,
    reference: SolverResult,
    method: SolverResult,
    k: int,
    tolerance: float,
) -> Dict[str, object]:
    comparable = reference.status == COMPLETE and method.status == COMPLETE
    reference_routes = sorted(
        reference.routes,
        key=lambda route: route_rank(route, tolerance),
    )[:k]
    method_routes = sorted(
        method.routes,
        key=lambda route: route_rank(route, tolerance),
    )[:k]

    if comparable:
        reference_frontier = {
            (route.served_count, quantized_cost(route.cost, tolerance))
            for route in reference.frontier
        }
        method_frontier = {
            (route.served_count, quantized_cost(route.cost, tolerance))
            for route in method.frontier
        }
        frontier_match = reference_frontier == method_frontier
        topk_objective_match = objective_sequence(
            reference_routes, k, tolerance
        ) == objective_sequence(method_routes, k, tolerance)
        coverage_match = [route.served_count for route in reference_routes] == [
            route.served_count for route in method_routes
        ]
        cost_match = (
            len(reference_routes) == len(method_routes)
            and all(
                abs(left.cost - right.cost) <= tolerance
                for left, right in zip(reference_routes, method_routes)
            )
        )
        service_set_match = [
            route.served_requests for route in reference_routes
        ] == [route.served_requests for route in method_routes]
        route_match = [route.route_events for route in reference_routes] == [
            route.route_events for route in method_routes
        ]
        key_path_match = [route.key_path for route in reference_routes] == [
            route.key_path for route in method_routes
        ]
        paths_available = all(
            route.full_path is not None
            for route in (*reference_routes, *method_routes)
        )
        physical_path_match = (
            [route.full_path for route in reference_routes]
            == [route.full_path for route in method_routes]
            if paths_available
            else ""
        )
        physical_path_source = "recorded" if paths_available else "missing"
        concrete_topk_match = (
            cost_match
            and service_set_match
            and route_match
            and key_path_match
            and physical_path_match
            if paths_available
            else ""
        )
    else:
        frontier_match = ""
        topk_objective_match = ""
        coverage_match = ""
        cost_match = ""
        service_set_match = ""
        route_match = ""
        key_path_match = ""
        physical_path_match = ""
        physical_path_source = "missing"
        concrete_topk_match = ""

    return {
        "query_id": query.query_id,
        "query_fingerprint": query_fingerprint(query),
        "dataset": query.dataset,
        "request_size": query.request_size,
        "seed": query.seed,
        "k": k,
        "comparison": f"{method.method}_vs_{reference.method}",
        "reference_method": reference.method,
        "method": method.method,
        "reference_status": reference.status,
        "method_status": method.status,
        "comparable": comparable,
        "frontier_match": frontier_match,
        "topk_objective_match": topk_objective_match,
        "coverage_match": coverage_match,
        "cost_match": cost_match,
        "service_set_match": service_set_match,
        "route_match": route_match,
        "key_path_match": key_path_match,
        "physical_path_match": physical_path_match,
        "physical_path_source": physical_path_source,
        "concrete_topk_match": concrete_topk_match,
    }


def build_comparison_rows(
    queries: Sequence[QueryInstance],
    result_map: Dict[Tuple[str, str], SolverResult],
    k_values: Sequence[int],
    tolerance: float,
) -> List[Dict[str, object]]:
    rows = []
    for query in queries:
        exact = result_map.get((query.query_id, "Exhaustive"))
        kprun = result_map.get((query.query_id, "KPrun-Search"))
        dpkmor = result_map.get((query.query_id, "DPKMOR"))
        if exact is None or kprun is None or dpkmor is None:
            continue
        for k in k_values:
            rows.append(compare_results(query, exact, kprun, k, tolerance))
            rows.append(compare_results(query, exact, dpkmor, k, tolerance))
            rows.append(compare_results(query, kprun, dpkmor, k, tolerance))
    return rows


def bool_value(value: object) -> bool:
    return str(value).strip().lower() == "true"


def summarize_group(
    rows: Sequence[Dict[str, object]],
    dataset: object,
    request_size: object,
    k: object,
    comparison: str,
) -> Dict[str, object]:
    comparable = [row for row in rows if bool_value(row["comparable"])]

    def rate(field: str) -> float:
        verified = [
            row for row in comparable
            if str(row.get(field, "")).strip().lower() in {"true", "false"}
        ]
        if not verified:
            return math.nan
        return sum(bool_value(row[field]) for row in verified) / len(verified)

    return {
        "dataset": dataset,
        "request_size": request_size,
        "k": k,
        "comparison": comparison,
        "total_queries": len(rows),
        "completed_pairs": len(comparable),
        "unresolved_pairs": len(rows) - len(comparable),
        "timeout_pairs": sum(
            TIMEOUT in {str(row["reference_status"]), str(row["method_status"])}
            for row in rows
        ),
        "oom_pairs": sum(
            OOM in {str(row["reference_status"]), str(row["method_status"])}
            for row in rows
        ),
        "state_limit_pairs": sum(
            STATE_LIMIT in {str(row["reference_status"]), str(row["method_status"])}
            for row in rows
        ),
        "frontier_match_rate": rate("frontier_match"),
        "topk_objective_match_rate": rate("topk_objective_match"),
        "coverage_match_rate": rate("coverage_match"),
        "cost_match_rate": rate("cost_match"),
        "service_set_match_rate": rate("service_set_match"),
        "route_match_rate": rate("route_match"),
        "key_path_match_rate": rate("key_path_match"),
        "physical_path_compared_pairs": sum(
            str(row.get("physical_path_match", "")).strip().lower() in {"true", "false"}
            for row in comparable
        ),
        "physical_path_recorded_pairs": sum(
            str(row.get("physical_path_source", "")) == "recorded"
            for row in comparable
        ),
        "physical_path_match_rate": rate("physical_path_match"),
        "concrete_topk_match_rate": rate("concrete_topk_match"),
    }


def summarize_comparisons(
    rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int, int, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            int(row["request_size"]),
            int(row["k"]),
            str(row["comparison"]),
        )
        grouped.setdefault(key, []).append(row)

    summary = [
        summarize_group(items, dataset, request_size, k, comparison)
        for (dataset, request_size, k, comparison), items in sorted(grouped.items())
    ]

    comparisons = sorted({str(row["comparison"]) for row in rows})
    for comparison in comparisons:
        items = [row for row in rows if row["comparison"] == comparison]
        summary.append(summarize_group(items, "ALL", "ALL", "ALL", comparison))
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify static KMOR correctness against exhaustive search"
    )
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--config", default="paper_datasets.json")
    parser.add_argument("--output-dir", default="paper_results_correctness")
    parser.add_argument(
        "--query-bank",
        default="",
        help="Optional existing query-bank JSONL; defaults to the output directory",
    )
    parser.add_argument(
        "--query-ids",
        default="",
        help="Optional comma-separated query IDs for targeted verification",
    )
    parser.add_argument("--datasets", default="OL,TG")
    parser.add_argument("--request-sizes", default="8,10,12")
    parser.add_argument("--k-values", default="1,3,5,10")
    parser.add_argument("--queries-per-size", type=int, default=100)
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--detour-ratio", type=float, default=1.4)
    parser.add_argument("--max-states", type=int, default=2_000_000)
    parser.add_argument("--exact-max-states", type=int, default=5_000_000)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--cost-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--dp-mode",
        choices=("exact_fast", "bundled"),
        default="exact_fast",
        help=(
            "DPKMOR implementation used for correctness verification. "
            "The default preserves prior experiments; bundled verifies "
            "concrete plan-level Top-k histories."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--reuse-query-bank", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = make_parser().parse_args(argv)
    workdir = Path(args.workdir).resolve()
    output_dir = (workdir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    query_bank_path = (
        (workdir / args.query_bank).resolve()
        if args.query_bank
        else output_dir / "correctness_queries.jsonl"
    )
    method_path = output_dir / "correctness_method_results.csv"
    comparison_path = output_dir / "correctness_comparison_results.csv"
    summary_path = output_dir / "correctness_summary.csv"
    run_config_path = output_dir / "correctness_run_config.json"

    run_config_path.write_text(
        json.dumps(
            {
                "algorithm_version": ALGORITHM_VERSION,
                "dp_mode": args.dp_mode,
                "config": args.config,
                "datasets": args.datasets,
                "request_sizes": args.request_sizes,
                "k_values": args.k_values,
                "queries_per_size": args.queries_per_size,
                "capacity": args.capacity,
                "detour_ratio": args.detour_ratio,
                "max_states": args.max_states,
                "exact_max_states": args.exact_max_states,
                "timeout_seconds": args.timeout,
                "cost_tolerance": args.cost_tolerance,
                "query_bank": args.query_bank,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    k_values = parse_int_list(args.k_values)
    request_sizes = parse_int_list(args.request_sizes)
    selected_names = set(parse_name_list(args.datasets))
    configs = [
        config
        for config in load_configs(workdir / args.config)
        if config.name.upper() in selected_names
    ]
    missing = selected_names - {config.name.upper() for config in configs}
    if missing:
        raise ValueError(f"datasets not found in config: {sorted(missing)}")

    if args.summarize_only:
        comparison_rows = read_csv(comparison_path)
        summary_rows = summarize_comparisons(comparison_rows)
        write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
        print(f"Saved {summary_path}")
        return 0

    if (args.reuse_query_bank or args.query_bank) and query_bank_path.exists():
        queries = read_query_bank(query_bank_path)
        queries = [
            query
            for query in queries
            if query.dataset.upper() in selected_names
            and query.request_size in request_sizes
        ]
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
    print(f"Prepared {len(queries)} queries in {query_bank_path}")
    if args.generate_only:
        return 0

    existing_rows = [] if args.no_resume else read_csv(method_path)
    method_rows: List[Dict[str, object]] = list(existing_rows)
    reusable: Dict[Tuple[str, str, str], Dict[str, str]] = {
        (row["query_id"], row["method"], row["run_fingerprint"]): row
        for row in existing_rows
    }
    result_map: Dict[Tuple[str, str], SolverResult] = {}
    rows_by_key: Dict[Tuple[str, str, str], Dict[str, object]] = {
        (str(row["query_id"]), str(row["method"]), str(row["run_fingerprint"])): row
        for row in method_rows
    }

    configs_by_name = {config.name: config for config in configs}
    queries_by_dataset: Dict[str, List[QueryInstance]] = {}
    for query in queries:
        queries_by_dataset.setdefault(query.dataset, []).append(query)

    for dataset, dataset_queries in queries_by_dataset.items():
        config = configs_by_name[dataset]
        print(f"Loading {dataset} graph...")
        graph = read_edge_list(
            str(workdir / config.edge_file),
            directed=config.directed,
            ignore_first_col=config.ignore_first_col,
        )
        shortest_path_cache: Dict[Tuple[str, str], Tuple[List[str], float]] = {}

        for index, query in enumerate(dataset_queries, start=1):
            run_id = run_fingerprint(
                query,
                args.capacity,
                args.detour_ratio,
                max(k_values),
                args.max_states,
                args.exact_max_states,
                args.timeout,
                args.cost_tolerance,
                args.dp_mode,
            )
            cached = {
                method: reusable.get((query.query_id, method, run_id))
                for method in ["Exhaustive", "KPrun-Search", "DPKMOR"]
            }
            if all(row is not None for row in cached.values()):
                results = [row_to_solver_result(cached[method]) for method in cached]
                label = "resumed"
            else:
                results = run_query(
                    graph,
                    query,
                    shortest_path_cache,
                    capacity=args.capacity,
                    detour_ratio=args.detour_ratio,
                    k_max=max(k_values),
                    max_states=args.max_states,
                    exact_max_states=args.exact_max_states,
                    timeout_seconds=args.timeout,
                    numeric_tolerance=args.cost_tolerance,
                    dp_mode=args.dp_mode,
                )
                label = "completed"
                for result in results:
                    stale_keys = [
                        key
                        for key in rows_by_key
                        if key[0] == query.query_id
                        and key[1] == result.method
                        and key[2] != run_id
                    ]
                    for stale_key in stale_keys:
                        del rows_by_key[stale_key]
                    row = solver_result_to_row(
                        query,
                        result,
                        run_id,
                        max(k_values),
                    )
                    rows_by_key[(query.query_id, result.method, run_id)] = row
                method_rows = list(rows_by_key.values())
                method_rows.sort(
                    key=lambda row: (
                        str(row["dataset"]),
                        int(row["request_size"]),
                        str(row["query_id"]),
                        str(row["method"]),
                    )
                )
                write_csv(method_path, method_rows, METHOD_FIELDS)

            for result in results:
                result_map[(query.query_id, result.method)] = result
            status_text = ", ".join(
                f"{result.method}:{result.status}" for result in results
            )
            print(
                f"[{dataset} {index}/{len(dataset_queries)}] "
                f"{query.query_id} {label} - {status_text}"
            )

    comparison_rows = build_comparison_rows(
        queries,
        result_map,
        k_values,
        args.cost_tolerance,
    )
    summary_rows = summarize_comparisons(comparison_rows)
    write_csv(comparison_path, comparison_rows, COMPARISON_FIELDS)
    write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    print(f"Saved {method_path}")
    print(f"Saved {comparison_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {run_config_path} (dp_mode={args.dp_mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
