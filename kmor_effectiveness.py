#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KMOR effectiveness comparison.

Compare:
1. KMOR
2. Greedy insertion
3. Nearest insertion
4. Best insertion (Solomon-style)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from Dpkmor import Graph, RideSharePlanner, parse_requests_file, read_edge_list


@dataclass
class EffectivenessResult:
    method: str
    served_count: int
    total_cost: float
    total_time: float
    avg_wait: float
    max_wait: float
    preprocess_time: float
    solve_time: float
    runtime: float
    served_requests: List[str]
    feasible: bool = True


@dataclass
class ScaleEffectivenessResult:
    scale: str
    method: str
    served_count: int
    total_cost: float
    avg_wait: float
    max_wait: float
    preprocess_time: float
    solve_time: float
    runtime: float
    feasible: bool = True


Token = Tuple[str, Optional[int]]


def token_node(planner: RideSharePlanner, token: Token) -> str:
    kind, req_idx = token
    if kind == "start":
        return planner.source
    if kind == "end":
        return planner.target
    if req_idx is None:
        raise ValueError("request index missing")
    req = planner.requests[req_idx]
    return req.pickup if kind == "pickup" else req.dropoff


def resolve_endpoints(graph: Graph, source: str, target: str) -> Tuple[str, str]:
    nodes = list(graph.nodes())
    if not nodes:
        raise ValueError("graph has no nodes")
    try:
        numeric_nodes = sorted((int(node), str(node)) for node in nodes)
        default_source = numeric_nodes[0][1]
        default_target = numeric_nodes[-1][1]
    except ValueError:
        sorted_nodes = sorted(str(node) for node in nodes)
        default_source = sorted_nodes[0]
        default_target = sorted_nodes[-1]

    resolved_source = default_source if source.lower() == "auto" else source
    resolved_target = default_target if target.lower() == "auto" else target
    return resolved_source, resolved_target


def expand_event_route(planner: RideSharePlanner, route: Sequence[Token]) -> List[str]:
    full: List[str] = []
    for i, token in enumerate(route):
        node = token_node(planner, token)
        if i == 0:
            full.append(node)
            continue
        prev_node = token_node(planner, route[i - 1])
        segment = planner.path_cache.get((prev_node, node), [])
        if not segment:
            return []
        full.extend(segment[1:])
    return full


def evaluate_event_route(planner: RideSharePlanner, route: Sequence[Token]) -> Optional[Dict[str, object]]:
    if not route or route[0][0] != "start" or route[-1][0] != "end":
        return None

    current_node = planner.source
    current_time = 0.0
    load = 0
    picked = set()
    served = set()
    pickup_info: List[Tuple[str, float, float, float]] = []

    for token in route[1:]:
        next_node = token_node(planner, token)
        travel_time = planner.dist(current_node, next_node)
        if travel_time == float("inf"):
            return None
        current_time += travel_time

        kind, req_idx = token
        if kind == "pickup":
            if req_idx is None:
                return None
            req = planner.requests[req_idx]
            if current_time < req.request_time:
                current_time = req.request_time
            wait_time = current_time - req.request_time
            if wait_time > req.wait_limit:
                return None
            if load + req.demand > planner.capacity:
                return None
            load += req.demand
            picked.add(req_idx)
            pickup_info.append((req.name, req.request_time, current_time, wait_time))
        elif kind == "dropoff":
            if req_idx is None or req_idx not in picked or req_idx in served:
                return None
            req = planner.requests[req_idx]
            load -= req.demand
            served.add(req_idx)
        elif kind == "end":
            if load != 0:
                return None

        if current_time > planner.time_budget:
            return None
        current_node = next_node

    full_path = expand_event_route(planner, route)
    if not full_path:
        return None

    waits = [item[3] for item in pickup_info]
    served_requests = [planner.requests[idx].name for idx in sorted(served)]
    return {
        "served_count": len(served),
        "served_requests": served_requests,
        "total_cost": current_time,
        "total_time": current_time,
        "avg_wait": sum(waits) / len(waits) if waits else 0.0,
        "max_wait": max(waits) if waits else 0.0,
    }


def inf_result(method: str) -> EffectivenessResult:
    return EffectivenessResult(
        method=method,
        served_count=0,
        total_cost=float("inf"),
        total_time=float("inf"),
        avg_wait=float("inf"),
        max_wait=float("inf"),
        preprocess_time=float("inf"),
        solve_time=float("inf"),
        runtime=float("inf"),
        served_requests=[],
        feasible=False,
    )


def build_result_from_eval(method: str, info: Optional[Dict[str, object]], runtime: float) -> EffectivenessResult:
    if info is None:
        result = inf_result(method)
        result.preprocess_time = 0.0
        result.solve_time = runtime
        result.runtime = runtime
        return result
    return EffectivenessResult(
        method=method,
        served_count=int(info["served_count"]),
        total_cost=float(info["total_cost"]),
        total_time=float(info["total_time"]),
        avg_wait=float(info["avg_wait"]),
        max_wait=float(info["max_wait"]),
        preprocess_time=0.0,
        solve_time=runtime,
        runtime=runtime,
        served_requests=list(info["served_requests"]),
        feasible=True,
    )


def run_greedy_baseline(planner: RideSharePlanner) -> EffectivenessResult:
    start = time.perf_counter()
    route: List[Token] = [("start", None), ("end", None)]
    chosen = set()
    best_feasible_route: List[Token] = route.copy()
    best_feasible_eval = evaluate_event_route(planner, route)

    while True:
        current_eval = evaluate_event_route(planner, route)
        if current_eval is None:
            break
        best_feasible_route = route.copy()
        best_feasible_eval = current_eval
        current_cost = float(current_eval["total_cost"])

        best_route = None
        best_eval = None
        best_key = None

        for req_idx, req in enumerate(planner.requests):
            if req_idx in chosen:
                continue
            if req.demand > planner.capacity:
                continue
            for i in range(1, len(route)):
                for j in range(i + 1, len(route) + 1):
                    trial = list(route)
                    trial.insert(i, ("pickup", req_idx))
                    trial.insert(j, ("dropoff", req_idx))
                    trial_eval = evaluate_event_route(planner, trial)
                    if trial_eval is None:
                        continue
                    score_key = (
                        -int(trial_eval["served_count"]),
                        float(trial_eval["total_cost"]) - current_cost,
                        float(trial_eval["avg_wait"]),
                    )
                    if best_key is None or score_key < best_key:
                        best_key = score_key
                        best_route = trial
                        best_eval = trial_eval

        if best_route is None or best_eval is None:
            break

        route = best_route
        chosen = {
            req_idx
            for kind, req_idx in route
            if kind in {"pickup", "dropoff"} and req_idx is not None
        }

    final_eval = evaluate_event_route(planner, route)
    runtime = time.perf_counter() - start
    if final_eval is None:
        final_eval = best_feasible_eval
        route = best_feasible_route
    if final_eval is None:
        result = inf_result("GreedyInsertion")
        result.solve_time = runtime
        result.runtime = runtime
        result.preprocess_time = 0.0
        return result
    return EffectivenessResult(
        method="GreedyInsertion",
        served_count=int(final_eval["served_count"]),
        total_cost=float(final_eval["total_cost"]),
        total_time=float(final_eval["total_time"]),
        avg_wait=float(final_eval["avg_wait"]),
        max_wait=float(final_eval["max_wait"]),
        preprocess_time=0.0,
        solve_time=runtime,
        runtime=runtime,
        served_requests=list(final_eval["served_requests"]),
        feasible=True,
    )


def route_request_set(route: Sequence[Token]) -> set:
    return {
        req_idx
        for kind, req_idx in route
        if kind in {"pickup", "dropoff"} and req_idx is not None
    }


def best_insertion_for_request(
    planner: RideSharePlanner,
    route: Sequence[Token],
    req_idx: int,
) -> Optional[Tuple[List[Token], Dict[str, object], float]]:
    current_eval = evaluate_event_route(planner, route)
    if current_eval is None:
        return None
    current_cost = float(current_eval["total_cost"])
    best: Optional[Tuple[List[Token], Dict[str, object], float]] = None

    for i in range(1, len(route)):
        for j in range(i + 1, len(route) + 1):
            trial = list(route)
            trial.insert(i, ("pickup", req_idx))
            trial.insert(j, ("dropoff", req_idx))
            trial_eval = evaluate_event_route(planner, trial)
            if trial_eval is None:
                continue
            marginal_cost = float(trial_eval["total_cost"]) - current_cost
            if best is None or marginal_cost < best[2]:
                best = (trial, trial_eval, marginal_cost)
    return best


def run_nearest_insertion_baseline(planner: RideSharePlanner) -> EffectivenessResult:
    start = time.perf_counter()
    route: List[Token] = [("start", None), ("end", None)]
    best_feasible_eval = evaluate_event_route(planner, route)

    while True:
        chosen = route_request_set(route)
        candidates = []
        for req_idx, req in enumerate(planner.requests):
            if req_idx in chosen or req.demand > planner.capacity:
                continue
            insertion = best_insertion_for_request(planner, route, req_idx)
            if insertion is None:
                continue
            trial_route, trial_eval, marginal_cost = insertion
            nearest_distance = min(planner.dist(token_node(planner, token), req.pickup) for token in route)
            candidates.append(
                (
                    nearest_distance,
                    marginal_cost,
                    float(trial_eval["avg_wait"]),
                    trial_route,
                    trial_eval,
                )
            )

        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        route = candidates[0][3]
        best_feasible_eval = candidates[0][4]

    runtime = time.perf_counter() - start
    return build_result_from_eval("NearestInsertion", best_feasible_eval, runtime)


def run_solomon_insertion_baseline(planner: RideSharePlanner) -> EffectivenessResult:
    start = time.perf_counter()
    route: List[Token] = [("start", None), ("end", None)]
    best_feasible_eval = evaluate_event_route(planner, route)

    while True:
        chosen = route_request_set(route)
        current_eval = evaluate_event_route(planner, route)
        if current_eval is None:
            break
        candidates = []

        for req_idx, req in enumerate(planner.requests):
            if req_idx in chosen or req.demand > planner.capacity:
                continue
            insertion = best_insertion_for_request(planner, route, req_idx)
            if insertion is None:
                continue
            trial_route, trial_eval, marginal_cost = insertion
            direct_cost = planner.dist(req.pickup, req.dropoff)
            wait_penalty = float(trial_eval["avg_wait"])
            regret_like_saving = direct_cost - marginal_cost
            score = (
                -int(trial_eval["served_count"]),
                -regret_like_saving,
                marginal_cost,
                wait_penalty,
            )
            candidates.append((score, trial_route, trial_eval))

        if not candidates:
            break
        candidates.sort(key=lambda item: item[0])
        route = candidates[0][1]
        best_feasible_eval = candidates[0][2]

    runtime = time.perf_counter() - start
    return build_result_from_eval("BestInsertionSolomon", best_feasible_eval, runtime)


def run_kmor(planner: RideSharePlanner) -> EffectivenessResult:
    start = time.perf_counter()
    plans = planner.solve()
    runtime = time.perf_counter() - start
    best = plans[0] if plans else None
    if best is None:
        result = inf_result("KMOR")
        result.avg_wait = 0.0
        result.max_wait = 0.0
        result.solve_time = runtime
        result.runtime = runtime
        result.preprocess_time = 0.0
        return result

    waits = [item[3] for item in best.pickup_info]
    return EffectivenessResult(
        method="KMOR",
        served_count=best.served_count,
        total_cost=best.total_cost,
        total_time=best.total_time,
        avg_wait=sum(waits) / len(waits) if waits else 0.0,
        max_wait=max(waits) if waits else 0.0,
        preprocess_time=0.0,
        solve_time=runtime,
        runtime=runtime,
        served_requests=list(best.served_requests),
        feasible=True,
    )


def print_results(results: Sequence[EffectivenessResult]) -> None:
    print("\n=== KMOR vs Greedy ===")
    header = (
        f"{'Method':<18}"
        f"{'Status':>10}"
        f"{'Served':>8}"
        f"{'Cost':>12}"
        f"{'AvgWait':>12}"
        f"{'MaxWait':>12}"
        f"{'Prep(s)':>12}"
        f"{'Solve(s)':>12}"
        f"{'Total(s)':>12}"
    )
    print(header)
    print("-" * len(header))
    for item in results:
        print(
            f"{item.method:<18}"
            f"{('OK' if item.feasible else 'INF'):>10}"
            f"{item.served_count:>8}"
            f"{item.total_cost:>12.2f}"
            f"{item.avg_wait:>12.2f}"
            f"{item.max_wait:>12.2f}"
            f"{item.preprocess_time:>12.6f}"
            f"{item.solve_time:>12.6f}"
            f"{item.runtime:>12.6f}"
        )

    print()
    for item in results:
        served_names = ", ".join(item.served_requests) if item.served_requests else "None"
        print(f"{item.method}: served_requests = {served_names}")


def print_scale_results(results: Sequence[ScaleEffectivenessResult]) -> None:
    print("\n=== KMOR vs Greedy Across Graph Scales ===")
    header = (
        f"{'Scale':<10}"
        f"{'Method':<18}"
        f"{'Status':>10}"
        f"{'Served':>8}"
        f"{'Cost':>12}"
        f"{'AvgWait':>12}"
        f"{'MaxWait':>12}"
        f"{'Prep(s)':>12}"
        f"{'Solve(s)':>12}"
        f"{'Total(s)':>12}"
    )
    print(header)
    print("-" * len(header))
    for item in results:
        print(
            f"{item.scale:<10}"
            f"{item.method:<18}"
            f"{('OK' if item.feasible else 'INF'):>10}"
            f"{item.served_count:>8}"
            f"{item.total_cost:>12.2f}"
            f"{item.avg_wait:>12.2f}"
            f"{item.max_wait:>12.2f}"
            f"{item.preprocess_time:>12.6f}"
            f"{item.solve_time:>12.6f}"
            f"{item.runtime:>12.6f}"
        )


def run_single_comparison(
    input_path: str,
    requests_path: str,
    source: str,
    target: str,
    delimiter: str,
    directed: bool,
    ignore_first_col: bool,
    common_params: Dict[str, object],
) -> List[EffectivenessResult]:
    graph = read_edge_list(
        input_path,
        delimiter=delimiter,
        directed=directed,
        ignore_first_col=ignore_first_col,
    )
    resolved_source, resolved_target = resolve_endpoints(graph, source, target)
    requests = parse_requests_file(requests_path)

    prep_start = time.perf_counter()
    planner = RideSharePlanner(
        graph,
        resolved_source,
        resolved_target,
        requests,
        **common_params,
    )
    prep_time = time.perf_counter() - prep_start

    kmor_result = run_kmor(planner)
    kmor_result.preprocess_time = prep_time
    kmor_result.runtime = kmor_result.preprocess_time + kmor_result.solve_time

    baseline_results = [
        run_greedy_baseline(planner),
        run_nearest_insertion_baseline(planner),
        run_solomon_insertion_baseline(planner),
    ]
    for result in baseline_results:
        result.preprocess_time = prep_time
        result.runtime = result.preprocess_time + result.solve_time

    return [kmor_result, *baseline_results]


def run_single_comparison_with_timeout(
    input_path: str,
    requests_path: str,
    source: str,
    target: str,
    delimiter: str,
    directed: bool,
    ignore_first_col: bool,
    common_params: Dict[str, object],
    timeout_seconds: float,
) -> List[EffectivenessResult]:
    methods = ["KMOR", "GreedyInsertion", "NearestInsertion", "BestInsertionSolomon"]
    fd, json_path = tempfile.mkstemp(prefix="kmor_effectiveness_", suffix=".json", dir=os.getcwd())
    os.close(fd)
    command = [
        sys.executable,
        __file__,
        "--input",
        input_path,
        "--requests-file",
        requests_path,
        "--source",
        source,
        "--target",
        target,
        "--delimiter",
        delimiter,
        "--capacity",
        str(common_params["capacity"]),
        "--k",
        str(common_params["k"]),
        "--time-budget",
        str(common_params["time_budget"]),
        "--detour-ratio",
        str(common_params["detour_ratio"]),
        "--max-states",
        str(common_params["max_states"]),
        "--emit-json",
        json_path,
    ]
    if ignore_first_col:
        command.append("--ignore-first-col")
    if directed:
        command.append("--directed")

    try:
        completed = subprocess.run(
            command,
            cwd=os.getcwd(),
            timeout=timeout_seconds,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        try:
            os.remove(json_path)
        except OSError:
            pass
        return [inf_result(method) for method in methods]

    if completed.returncode != 0:
        try:
            os.remove(json_path)
        except OSError:
            pass
        return [inf_result(method) for method in methods]

    try:
        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    finally:
        try:
            os.remove(json_path)
        except OSError:
            pass

    if payload.get("status") != "ok":
        return [inf_result(method) for method in methods]
    return [EffectivenessResult(**item) for item in payload["results"]]


def parse_scales(raw: str) -> List[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_scale_budget_map(raw: str) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    if not raw:
        return mapping
    for part in raw.split(","):
        item = part.strip()
        if not item or ":" not in item:
            continue
        scale, value = item.split(":", 1)
        mapping[scale.strip()] = float(value.strip())
    return mapping


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="KMOR effectiveness comparison")
    ap.add_argument("--input", help="Edge list file path")
    ap.add_argument("--requests-file", help="Request file path")
    ap.add_argument("--source", required=True, help="Source node or auto")
    ap.add_argument("--target", required=True, help="Target node or auto")
    ap.add_argument("--delimiter", default="space", choices=["space", "tab", "comma"])
    ap.add_argument("--ignore-first-col", action="store_true")
    ap.add_argument("--directed", action="store_true")
    ap.add_argument("--capacity", type=int, default=4)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--time-budget", type=float, default=float("inf"))
    ap.add_argument("--detour-ratio", type=float, default=1.8)
    ap.add_argument("--max-states", type=int, default=200000)
    ap.add_argument("--timeout", type=float, default=3600.0, help="Per-dataset timeout in seconds")
    ap.add_argument("--emit-json", default="", help=argparse.SUPPRESS)
    ap.add_argument("--dataset-dir", default=".", help="Dataset directory in batch mode")
    ap.add_argument("--scales", default="small,medium,large", help="Comma-separated graph scales")
    ap.add_argument("--batch", action="store_true", help="Run small/medium/large batch comparison")
    ap.add_argument(
        "--scale-time-budgets",
        default="",
        help="Per-scale time budgets, e.g. small:150,medium:300,large:1000",
    )
    return ap


def main() -> int:
    args = build_parser().parse_args()

    common_params = {
        "capacity": args.capacity,
        "k": args.k,
        "time_budget": args.time_budget,
        "detour_ratio": args.detour_ratio,
        "max_states": args.max_states,
    }

    if args.batch:
        dataset_dir = Path(args.dataset_dir).resolve()
        scales = parse_scales(args.scales)
        scale_budget_map = parse_scale_budget_map(args.scale_time_budgets)
        aggregated: List[ScaleEffectivenessResult] = []
        for scale in scales:
            input_path = dataset_dir / f"edges_{scale}.txt"
            requests_path = dataset_dir / f"requests_{scale}.txt"
            if not input_path.exists():
                raise FileNotFoundError(f"Missing graph file: {input_path}")
            if not requests_path.exists():
                raise FileNotFoundError(f"Missing request file: {requests_path}")

            scale_params = dict(common_params)
            if scale in scale_budget_map:
                scale_params["time_budget"] = scale_budget_map[scale]

            results = run_single_comparison_with_timeout(
                str(input_path),
                str(requests_path),
                args.source,
                args.target,
                args.delimiter,
                args.directed,
                args.ignore_first_col,
                scale_params,
                args.timeout,
            )
            for item in results:
                aggregated.append(
                    ScaleEffectivenessResult(
                        scale=scale,
                        method=item.method,
                        served_count=item.served_count,
                        total_cost=item.total_cost,
                        avg_wait=item.avg_wait,
                        max_wait=item.max_wait,
                        preprocess_time=item.preprocess_time,
                        solve_time=item.solve_time,
                        runtime=item.runtime,
                        feasible=item.feasible,
                    )
                )
        print_scale_results(aggregated)
        return 0

    if not args.input or not args.requests_file:
        raise ValueError("single-run mode requires --input and --requests-file")

    if args.emit_json:
        try:
            results = run_single_comparison(
                args.input,
                args.requests_file,
                args.source,
                args.target,
                args.delimiter,
                args.directed,
                args.ignore_first_col,
                common_params,
            )
            payload = {"status": "ok", "results": [result.__dict__ for result in results]}
        except MemoryError:
            payload = {"status": "oom"}
        except BaseException as exc:  # pragma: no cover - subprocess safety
            payload = {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}
        Path(args.emit_json).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return 0

    results = run_single_comparison_with_timeout(
        args.input,
        args.requests_file,
        args.source,
        args.target,
        args.delimiter,
        args.directed,
        args.ignore_first_col,
        common_params,
        args.timeout,
    )
    print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
