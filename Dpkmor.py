#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DPKMOR:
在真实路网上做乘客拼车实验，支持：
1. 乘客 pickup/dropoff 顺序约束
2. 车辆容量约束
3. 乘客等待时间约束（request_time 之后 wait_limit 内必须接到）
4. Top-k 多方案输出
5. 基于状态压缩 DP + 剪枝的搜索
"""

from __future__ import annotations

import argparse
import heapq
import math
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


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
        if source == target:
            return [source], 0.0

        pq: List[Tuple[float, str]] = [(0.0, source)]
        dist: Dict[str, float] = {source: 0.0}
        prev: Dict[str, Optional[str]] = {source: None}
        visited = set()

        while pq:
            curr_dist, node = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)

            if node == target:
                break

            for nxt, weight in self.neighbors(node):
                nd = curr_dist + weight
                if nd < dist.get(nxt, math.inf):
                    dist[nxt] = nd
                    prev[nxt] = node
                    heapq.heappush(pq, (nd, nxt))

        if target not in dist:
            return [], math.inf

        path = []
        cur: Optional[str] = target
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()
        return path, dist[target]


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


@dataclass
class PlannerStats:
    generated_states: int = 0
    popped_states: int = 0
    expanded_states: int = 0
    dp_pruned_states: int = 0
    feasibility_pruned_states: int = 0
    heuristic_pruned_states: int = 0
    compatibility_pruned_requests: int = 0
    complete_candidates: int = 0


@dataclass(frozen=True)
class SearchState:
    node: str
    picked_mask: int
    served_mask: int
    load: int
    time_used: float
    key_path: Tuple[str, ...]
    pickup_times: Tuple[float, ...]


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
        self.time_budget = float(params.get("time_budget", math.inf))
        self.detour_ratio = float(params.get("detour_ratio", 1.8))
        self.compatibility_scale = float(params.get("compatibility_scale", 1.0))
        self.max_candidates = int(params.get("max_candidates", self.k * 8))
        self.expensive_bound_slack_ratio = float(params.get("expensive_bound_slack_ratio", 0.05))
        self.expensive_bound_min_onboard = int(params.get("expensive_bound_min_onboard", 2))
        self.enable_state_bounds = bool(params.get("enable_state_bounds", False))
        self.aggressive_served_pruning = bool(params.get("aggressive_served_pruning", False))
        target_served = params.get("target_served_count")
        self.target_served_count = int(target_served) if target_served is not None else None
        self.enable_pruning = bool(params.get("enable_pruning", True))
        self.enable_dp = bool(params.get("enable_dp", True))
        self.shortest_path_cache = params.get("shortest_path_cache")
        self.stats = PlannerStats()

        self.key_nodes = self._build_key_nodes()
        self.dist_cache: Dict[Tuple[str, str], float] = {}
        self.path_cache: Dict[Tuple[str, str], List[str]] = {}
        self.state_lb_cache: Dict[Tuple[str, int, int, int], float] = {}
        self._precompute_shortest_paths()
        self.base_route, self.base_cost = self.graph.shortest_path(self.source, self.target)
        self.drop_target_costs = [self.dist(req.dropoff, self.target) for req in self.requests]
        self.compatible = self._build_compatibility_mask()

    def _build_key_nodes(self) -> List[str]:
        nodes = {self.source, self.target}
        for req in self.requests:
            nodes.add(req.pickup)
            nodes.add(req.dropoff)
        return sorted(nodes)

    def _precompute_shortest_paths(self) -> None:
        for u in self.key_nodes:
            for v in self.key_nodes:
                if u == v:
                    self.dist_cache[(u, v)] = 0.0
                    self.path_cache[(u, v)] = [u]
                    continue
                cache_key = (u, v)
                if self.shortest_path_cache is not None and cache_key in self.shortest_path_cache:
                    path, dist = self.shortest_path_cache[cache_key]
                else:
                    path, dist = self.graph.shortest_path(u, v)
                    if self.shortest_path_cache is not None:
                        self.shortest_path_cache[cache_key] = (path, dist)
                self.dist_cache[(u, v)] = dist
                self.path_cache[(u, v)] = path

    def dist(self, u: str, v: str) -> float:
        return self.dist_cache.get((u, v), math.inf)

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
            direct = self.dist(req.pickup, req.dropoff)
            if math.isinf(direct):
                continue
            if req.demand > self.capacity:
                self.stats.feasibility_pruned_states += 1
                continue
            earliest_pickup = self.dist(self.source, req.pickup)
            if math.isinf(earliest_pickup):
                self.stats.feasibility_pruned_states += 1
                continue
            if earliest_pickup - req.request_time > req.wait_limit:
                self.stats.feasibility_pruned_states += 1
                continue
            if self.enable_pruning and self._corridor_score(req) > self.detour_ratio * self.compatibility_scale:
                self.stats.heuristic_pruned_states += 1
                self.stats.compatibility_pruned_requests += 1
                continue
            mask |= 1 << idx
        return mask

    def _mask_count(self, mask: int) -> int:
        return mask.bit_count()

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

    def _transition_pickup(self, state: SearchState, req_idx: int) -> Optional[SearchState]:
        req = self.requests[req_idx]
        if not self._can_pick(req_idx):
            return None
        if (state.picked_mask >> req_idx) & 1:
            return None
        if state.load + req.demand > self.capacity:
            self.stats.feasibility_pruned_states += 1
            return None

        travel_time = self.dist(state.node, req.pickup)
        if math.isinf(travel_time):
            self.stats.feasibility_pruned_states += 1
            return None

        arrival_time = state.time_used + travel_time
        if arrival_time < req.request_time:
            arrival_time = req.request_time
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
        )

    def _transition_dropoff(self, state: SearchState, req_idx: int) -> Optional[SearchState]:
        if ((state.picked_mask >> req_idx) & 1) == 0:
            return None
        if ((state.served_mask >> req_idx) & 1) == 1:
            return None

        req = self.requests[req_idx]
        travel_time = self.dist(state.node, req.dropoff)
        if math.isinf(travel_time):
            self.stats.feasibility_pruned_states += 1
            return None

        arrival_time = state.time_used + travel_time
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
        )

    def _transition_target(self, state: SearchState) -> Optional[SearchState]:
        if state.load != 0:
            self.stats.feasibility_pruned_states += 1
            return None
        travel_time = self.dist(state.node, self.target)
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
        )

    def _best_possible_served(self, state: SearchState) -> int:
        possible = self._mask_count(state.served_mask)
        used = state.picked_mask | state.served_mask
        for idx in range(len(self.requests)):
            if (used >> idx) & 1:
                continue
            if self._can_pick(idx):
                possible += 1
        return possible

    def _priority(self, state: SearchState) -> float:
        served = self._mask_count(state.served_mask)
        optimistic = self._best_possible_served(state)
        return -(served * 100000 + optimistic * 1000) + state.time_used

    def _state_signature(self, state: SearchState) -> Tuple[str, int, int, int]:
        return (state.node, state.picked_mask, state.served_mask, state.load)

    def _cheap_lower_bound(self, state: SearchState) -> float:
        lower = self.dist(state.node, self.target)
        if math.isinf(lower):
            return math.inf
        onboard_mask = state.picked_mask & ~state.served_mask
        for idx in range(len(self.requests)):
            if ((onboard_mask >> idx) & 1) == 0:
                continue
            lower = max(lower, self.drop_target_costs[idx])
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
        key_path = list(state.key_path)
        full_path = self._expand_full_path(key_path)
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
        )

    def _pareto_top_k(self, candidates: List[Candidate]) -> List[Candidate]:
        if self.target_served_count is not None:
            filtered = [cand for cand in candidates if cand.served_count == self.target_served_count]
            if filtered:
                candidates = filtered

        unique: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], Candidate] = {}
        for cand in candidates:
            key = (tuple(cand.key_path), tuple(cand.served_requests))
            old = unique.get(key)
            if old is None or cand.total_cost < old.total_cost:
                unique[key] = cand

        values = list(unique.values())
        pareto: List[Candidate] = []
        for cand in values:
            dominated = False
            for other in values:
                if other is cand:
                    continue
                if other.served_count >= cand.served_count and other.total_cost <= cand.total_cost:
                    if other.served_count > cand.served_count or other.total_cost < cand.total_cost:
                        dominated = True
                        break
            if not dominated:
                pareto.append(cand)

        pareto.sort(key=lambda x: (-x.served_count, x.total_cost, x.hops))
        return pareto[: self.k]

    def _should_enqueue(
        self,
        state: SearchState,
        best_cost: Dict[Tuple[str, int, int, int], float],
    ) -> bool:
        if not self.enable_dp:
            return True
        signature = self._state_signature(state)
        old = best_cost.get(signature)
        if old is not None and old <= state.time_used:
            self.stats.dp_pruned_states += 1
            return False
        best_cost[signature] = state.time_used
        return True

    def _push_state(
        self,
        pq: List[Tuple[float, int, SearchState]],
        counter: int,
        state: SearchState,
        best_cost: Dict[Tuple[str, int, int, int], float],
    ) -> int:
        if not self._should_enqueue(state, best_cost):
            return counter
        heapq.heappush(pq, (self._priority(state), counter, state))
        self.stats.generated_states += 1
        return counter + 1

    def solve(self) -> List[Candidate]:
        initial = SearchState(
            node=self.source,
            picked_mask=0,
            served_mask=0,
            load=0,
            time_used=0.0,
            key_path=(self.source,),
            pickup_times=tuple([-1.0] * len(self.requests)),
        )

        pq: List[Tuple[float, int, SearchState]] = []
        counter = 0
        best_cost: Dict[Tuple[str, int, int, int], float] = {}
        counter = self._push_state(pq, counter, initial, best_cost)
        complete: List[Candidate] = []
        top_k_frontier: List[Candidate] = []
        expansions = 0
        cutoff_served = -1
        cutoff_cost = math.inf
        best_complete_served = -1
        best_cost_for_best_served = math.inf

        while pq and expansions < self.max_states:
            _, _, state = heapq.heappop(pq)
            self.stats.popped_states += 1

            signature = self._state_signature(state)
            if self.enable_dp:
                old = best_cost.get(signature)
                if old is not None and old < state.time_used:
                    self.stats.dp_pruned_states += 1
                    continue

            if self.enable_pruning and len(top_k_frontier) >= self.k:
                optimistic_served = self._best_possible_served(state)
                if optimistic_served < cutoff_served:
                    self.stats.heuristic_pruned_states += 1
                    continue
                if optimistic_served == cutoff_served and state.time_used >= cutoff_cost:
                    self.stats.heuristic_pruned_states += 1
                    continue
            elif self.enable_pruning and self.aggressive_served_pruning and best_complete_served >= 0:
                optimistic_served = self._best_possible_served(state)
                if optimistic_served < best_complete_served:
                    self.stats.heuristic_pruned_states += 1
                    continue
                if optimistic_served == best_complete_served and state.time_used >= best_cost_for_best_served:
                    self.stats.heuristic_pruned_states += 1
                    continue

            expansions += 1
            self.stats.expanded_states += 1

            if state.node == self.target:
                candidate = self._build_candidate(state)
                if candidate is not None:
                    complete.append(candidate)
                    if candidate.served_count > best_complete_served:
                        best_complete_served = candidate.served_count
                        best_cost_for_best_served = candidate.total_cost
                    elif candidate.served_count == best_complete_served:
                        best_cost_for_best_served = min(best_cost_for_best_served, candidate.total_cost)
                    if self.enable_pruning:
                        top_k_frontier.append(candidate)
                        top_k_frontier.sort(key=lambda item: (-item.served_count, item.total_cost, item.hops))
                        del top_k_frontier[self.k :]
                        if len(top_k_frontier) >= self.k:
                            boundary = top_k_frontier[-1]
                            cutoff_served = boundary.served_count
                            cutoff_cost = boundary.total_cost
                    self.stats.complete_candidates += 1
                continue

            if len(state.key_path) >= self.max_events:
                continue

            target_state = self._transition_target(state)
            if target_state is not None:
                counter = self._push_state(pq, counter, target_state, best_cost)

            onboard_mask = state.picked_mask & ~state.served_mask
            for idx in range(len(self.requests)):
                if (onboard_mask >> idx) & 1:
                    nxt = self._transition_dropoff(state, idx)
                    if nxt is not None:
                        if self.enable_pruning and self.enable_state_bounds:
                            lower_bound = self._cheap_lower_bound(nxt)
                            if math.isinf(lower_bound) or nxt.time_used + lower_bound > self.time_budget:
                                self.stats.heuristic_pruned_states += 1
                                continue
                        counter = self._push_state(pq, counter, nxt, best_cost)

            for idx in range(len(self.requests)):
                nxt = self._transition_pickup(state, idx)
                if nxt is None:
                    continue
                if self.enable_pruning and self.enable_state_bounds:
                    lower_bound = self._cheap_lower_bound(nxt)
                    if math.isinf(lower_bound) or nxt.time_used + lower_bound > self.time_budget:
                        self.stats.heuristic_pruned_states += 1
                        continue
                    request_bound = self._pickup_request_bound(nxt, idx)
                    lower_bound = max(lower_bound, request_bound)
                    if math.isinf(lower_bound) or nxt.time_used + lower_bound > self.time_budget:
                        self.stats.heuristic_pruned_states += 1
                        continue
                    if self._should_run_expensive_bound(nxt, complete, lower_bound):
                        expensive_bound = self._expensive_lower_bound(nxt)
                        if math.isinf(expensive_bound) or nxt.time_used + expensive_bound > self.time_budget:
                            self.stats.heuristic_pruned_states += 1
                            continue
                counter = self._push_state(pq, counter, nxt, best_cost)

        return self._pareto_top_k(complete)


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
