#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KMOR experiment runner

当前默认模式：
不同请求规模下的消融实验（Full / NoPruning / NoDP）

推荐默认请求规模：
5, 8, 10, 12, 15
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from Dpkmor import Graph, Request, RideSharePlanner, parse_requests_file, read_edge_list


@dataclass
class ExperimentResult:
    method: str
    served_count: int
    total_cost: float
    total_time: float
    avg_wait: float
    max_wait: float
    runtime: float
    preprocess_time: float = 0.0
    search_time: float = 0.0
    generated_states: int = 0
    expanded_states: int = 0
    dp_pruned_states: int = 0
    feasibility_pruned_states: int = 0
    heuristic_pruned_states: int = 0
    complete_candidates: int = 0
    served_requests: Optional[List[str]] = None
    timed_out: bool = False
    hit_max_states: bool = False


@dataclass
class AggregateResult:
    size: int
    method: str
    trials: int
    served_count: float
    total_cost: float
    avg_wait: float
    runtime: float
    preprocess_time: float
    search_time: float
    generated_states: float
    expanded_states: float
    dp_pruned_states: float
    feasibility_pruned_states: float
    heuristic_pruned_states: float
    hit_max_states_rate: float


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
    pickup_times: Dict[int, float] = {}
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
            pickup_times[req_idx] = current_time
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

    served_requests = [planner.requests[idx].name for idx in sorted(served)]
    waits = [item[3] for item in pickup_info]
    full_path = expand_event_route(planner, route)
    if not full_path:
        return None

    return {
        "served_count": len(served),
        "served_requests": served_requests,
        "pickup_info": pickup_info,
        "total_cost": current_time,
        "total_time": current_time,
        "avg_wait": sum(waits) / len(waits) if waits else 0.0,
        "max_wait": max(waits) if waits else 0.0,
        "full_path": full_path,
    }


def run_shortest_baseline(planner: RideSharePlanner) -> ExperimentResult:
    start = time.time()
    route = [("start", None), ("end", None)]
    info = evaluate_event_route(planner, route)
    runtime = time.time() - start
    if info is None:
        raise RuntimeError("shortest baseline route is infeasible")
    return ExperimentResult(
        method="ShortestPath",
        served_count=int(info["served_count"]),
        total_cost=float(info["total_cost"]),
        total_time=float(info["total_time"]),
        avg_wait=float(info["avg_wait"]),
        max_wait=float(info["max_wait"]),
        runtime=runtime,
        served_requests=list(info["served_requests"]),
    )


def run_greedy_baseline(planner: RideSharePlanner) -> ExperimentResult:
    start = time.time()
    route: List[Token] = [("start", None), ("end", None)]
    chosen = set()

    while True:
        current_eval = evaluate_event_route(planner, route)
        if current_eval is None:
            break
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
                    incr_cost = float(trial_eval["total_cost"]) - current_cost
                    score_key = (
                        -int(trial_eval["served_count"]),
                        incr_cost,
                        float(trial_eval["avg_wait"]),
                    )
                    if best_key is None or score_key < best_key:
                        best_key = score_key
                        best_route = trial
                        best_eval = trial_eval

        if best_route is None or best_eval is None:
            break

        route = best_route
        chosen = {req_idx for kind, req_idx in route if kind in {"pickup", "dropoff"} and req_idx is not None}

    final_eval = evaluate_event_route(planner, route)
    runtime = time.time() - start
    if final_eval is None:
        raise RuntimeError("greedy baseline produced infeasible route")
    return ExperimentResult(
        method="GreedyInsertion",
        served_count=int(final_eval["served_count"]),
        total_cost=float(final_eval["total_cost"]),
        total_time=float(final_eval["total_time"]),
        avg_wait=float(final_eval["avg_wait"]),
        max_wait=float(final_eval["max_wait"]),
        runtime=runtime,
        served_requests=list(final_eval["served_requests"]),
    )


def run_kmor_variant(
    graph: Graph,
    source: str,
    target: str,
    requests: List[Request],
    method_name: str,
    **params: object,
) -> ExperimentResult:
    preprocess_start = time.time()
    planner = RideSharePlanner(graph, source, target, requests, **params)
    preprocess_time = time.time() - preprocess_start
    search_start = time.time()
    plans = planner.solve()
    search_time = time.time() - search_start
    runtime = preprocess_time + search_time

    best = plans[0] if plans else None
    avg_wait = 0.0
    max_wait = 0.0
    served_requests: List[str] = []
    if best is not None and best.pickup_info:
        waits = [item[3] for item in best.pickup_info]
        avg_wait = sum(waits) / len(waits)
        max_wait = max(waits)
        served_requests = list(best.served_requests)

    return ExperimentResult(
        method=method_name,
        served_count=best.served_count if best is not None else 0,
        total_cost=best.total_cost if best is not None else float("inf"),
        total_time=best.total_time if best is not None else float("inf"),
        avg_wait=avg_wait,
        max_wait=max_wait,
        runtime=runtime,
        preprocess_time=preprocess_time,
        search_time=search_time,
        generated_states=planner.stats.generated_states,
        expanded_states=planner.stats.expanded_states,
        dp_pruned_states=planner.stats.dp_pruned_states,
        feasibility_pruned_states=planner.stats.feasibility_pruned_states,
        heuristic_pruned_states=planner.stats.heuristic_pruned_states,
        complete_candidates=planner.stats.complete_candidates,
        served_requests=served_requests,
        hit_max_states=planner.stats.expanded_states >= planner.max_states,
    )


def run_ablation_triplet(
    graph: Graph,
    source: str,
    target: str,
    requests: List[Request],
    timeout_seconds: float,
    input_path: Optional[str] = None,
    delimiter: str = "space",
    ignore_first_col: bool = False,
    directed: bool = False,
    align_served_to_full: bool = True,
    **common_params: object,
) -> List[ExperimentResult]:
    full = run_kmor_variant_with_timeout(
        graph,
        source,
        target,
        requests,
        "Full",
        timeout_seconds,
        input_path=input_path,
        delimiter=delimiter,
        ignore_first_col=ignore_first_col,
        directed=directed,
        enable_pruning=True,
        enable_dp=True,
        **common_params,
    )

    target_served_count = None
    if align_served_to_full and not full.timed_out and full.served_count >= 0:
        target_served_count = full.served_count

    variant_results = [full]
    for variant_name, switches in [
        ("NoPruning", {"enable_pruning": False, "enable_dp": True}),
        ("NoDP", {"enable_pruning": True, "enable_dp": False}),
    ]:
        params = dict(common_params)
        if target_served_count is not None:
            params["target_served_count"] = target_served_count
        result = run_kmor_variant_with_timeout(
            graph,
            source,
            target,
            requests,
            variant_name,
            timeout_seconds,
            input_path=input_path,
            delimiter=delimiter,
            ignore_first_col=ignore_first_col,
            directed=directed,
            **switches,
            **params,
        )
        variant_results.append(result)
    return variant_results


def serialize_requests(requests: Sequence[Request]) -> List[Dict[str, object]]:
    return [
        {
            "name": req.name,
            "pickup": req.pickup,
            "dropoff": req.dropoff,
            "demand": req.demand,
            "request_time": req.request_time,
            "wait_limit": req.wait_limit,
        }
        for req in requests
    ]


def deserialize_requests(payload: Sequence[Dict[str, object]]) -> List[Request]:
    return [
        Request(
            name=str(item["name"]),
            pickup=str(item["pickup"]),
            dropoff=str(item["dropoff"]),
            demand=int(item["demand"]),
            request_time=float(item["request_time"]),
            wait_limit=float(item["wait_limit"]),
        )
        for item in payload
    ]


def run_kmor_variant_with_timeout(
    graph: Graph,
    source: str,
    target: str,
    requests: List[Request],
    method_name: str,
    timeout_seconds: float,
    input_path: Optional[str] = None,
    delimiter: str = "space",
    ignore_first_col: bool = False,
    directed: bool = False,
    **params: object,
) -> ExperimentResult:
    if timeout_seconds <= 0:
        return run_kmor_variant(graph, source, target, requests, method_name, **params)
    if not input_path:
        return run_kmor_variant(graph, source, target, requests, method_name, **params)

    payload = {
        "source": source,
        "target": target,
        "method_name": method_name,
        "params": dict(params),
        "requests": serialize_requests(requests),
    }
    command = [
        sys.executable,
        __file__,
        "--input",
        input_path,
        "--requests-file",
        input_path,
        "--source",
        source,
        "--target",
        target,
        "--delimiter",
        delimiter,
        "--worker-json",
        json.dumps(payload, ensure_ascii=True),
    ]
    if ignore_first_col:
        command.append("--ignore-first-col")
    if directed:
        command.append("--directed")

    start = time.time()
    proc = subprocess.Popen(
        command,
        cwd=None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return ExperimentResult(
            method=method_name,
            served_count=-1,
            total_cost=float("inf"),
            total_time=float("inf"),
            avg_wait=float("inf"),
            max_wait=float("inf"),
            runtime=timeout_seconds,
            preprocess_time=float("inf"),
            search_time=float("inf"),
            served_requests=[],
            timed_out=True,
            hit_max_states=False,
        )

    runtime = time.time() - start
    if proc.returncode != 0:
        raise RuntimeError(stderr.strip() or f"worker exited with code {proc.returncode}")
    if not stdout.strip():
        return ExperimentResult(
            method=method_name,
            served_count=-1,
            total_cost=float("inf"),
            total_time=float("inf"),
            avg_wait=float("inf"),
            max_wait=float("inf"),
            runtime=runtime,
            preprocess_time=float("inf"),
            search_time=float("inf"),
            served_requests=[],
            timed_out=True,
            hit_max_states=False,
        )

    worker_result = json.loads(stdout.strip())
    result = ExperimentResult(**worker_result)
    result.runtime = min(result.runtime, runtime)
    return result


def average_results(size: int, method: str, results: Sequence[ExperimentResult]) -> AggregateResult:
    count = len(results)
    return AggregateResult(
        size=size,
        method=method,
        trials=count,
        served_count=sum(item.served_count for item in results) / count,
        total_cost=sum(item.total_cost for item in results) / count,
        avg_wait=sum(item.avg_wait for item in results) / count,
        runtime=sum(item.runtime for item in results) / count,
        preprocess_time=sum(item.preprocess_time for item in results) / count,
        search_time=sum(item.search_time for item in results) / count,
        generated_states=sum(item.generated_states for item in results) / count,
        expanded_states=sum(item.expanded_states for item in results) / count,
        dp_pruned_states=sum(item.dp_pruned_states for item in results) / count,
        feasibility_pruned_states=sum(item.feasibility_pruned_states for item in results) / count,
        heuristic_pruned_states=sum(item.heuristic_pruned_states for item in results) / count,
        hit_max_states_rate=sum(1.0 if item.hit_max_states else 0.0 for item in results) / count,
    )


def print_ablation_table(results: Sequence[ExperimentResult]) -> None:
    print("\n=== Full / 去剪枝 / 去DP 效率消融 ===")
    header = (
        f"{'Variant':<16}"
        f"{'Status':>10}"
        f"{'Served':>8}"
        f"{'Cost':>12}"
        f"{'Runtime(s)':>12}"
        f"{'Generated':>12}"
        f"{'Expanded':>12}"
        f"{'DPPruned':>12}"
        f"{'FeasPruned':>14}"
        f"{'HeurPruned':>14}"
    )
    print(header)
    print("-" * len(header))
    for item in results:
        print(
            f"{item.method:<16}"
            f"{('TIMEOUT' if item.timed_out else 'OK'):>10}"
            f"{item.served_count:>8}"
            f"{item.total_cost:>12.2f}"
            f"{item.runtime:>12.4f}"
            f"{item.generated_states:>12}"
            f"{item.expanded_states:>12}"
            f"{item.dp_pruned_states:>12}"
            f"{item.feasibility_pruned_states:>14}"
            f"{item.heuristic_pruned_states:>14}"
        )


def print_scale_ablation_table(results: Sequence[AggregateResult]) -> None:
    print("\n=== 不同请求规模的消融实验（平均值） ===")
    header = (
        f"{'Size':<8}"
        f"{'Variant':<16}"
        f"{'Served':>10}"
        f"{'Cost':>12}"
        f"{'Runtime(s)':>12}"
        f"{'Generated':>12}"
        f"{'Expanded':>12}"
        f"{'DPPruned':>12}"
        f"{'FeasPruned':>14}"
        f"{'HeurPruned':>14}"
    )
    print(header)
    print("-" * len(header))
    for item in results:
        print(
            f"{item.size:<8}"
            f"{item.method:<16}"
            f"{item.served_count:>10.2f}"
            f"{item.total_cost:>12.2f}"
            f"{item.runtime:>12.4f}"
            f"{item.generated_states:>12.1f}"
            f"{item.expanded_states:>12.1f}"
            f"{item.dp_pruned_states:>12.1f}"
            f"{item.feasibility_pruned_states:>14.1f}"
            f"{item.heuristic_pruned_states:>14.1f}"
        )


def parse_int_list(raw: str) -> List[int]:
    values: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def sample_requests(requests: Sequence[Request], size: int, seed: int) -> List[Request]:
    rng = random.Random(seed)
    if size >= len(requests):
        return list(requests)
    indices = sorted(rng.sample(range(len(requests)), size))
    return [requests[idx] for idx in indices]


def run_scale_ablation(
    graph: Graph,
    source: str,
    target: str,
    all_requests: List[Request],
    input_path: str,
    delimiter: str,
    ignore_first_col: bool,
    directed: bool,
    sizes: Sequence[int],
    trials: int,
    base_seed: int,
    timeout_seconds: float,
    common_params: Dict[str, object],
) -> List[AggregateResult]:
    aggregated: List[AggregateResult] = []
    variants = [
        ("Full", {"enable_pruning": True, "enable_dp": True}),
        ("NoPruning", {"enable_pruning": False, "enable_dp": True}),
        ("NoDP", {"enable_pruning": True, "enable_dp": False}),
    ]

    for size in sizes:
        print(f"[ScaleAblation] request_size={size}, trials={trials}")
        sampled_results: Dict[str, List[ExperimentResult]] = {name: [] for name, _ in variants}
        for trial in range(trials):
            print(f"  - trial {trial + 1}/{trials}")
            sampled_requests = sample_requests(all_requests, size, base_seed + size * 1000 + trial)
            for variant_name, switches in variants:
                print(f"    * {variant_name} running...")
                result = run_kmor_variant_with_timeout(
                    graph,
                    source,
                    target,
                    sampled_requests,
                    variant_name,
                    timeout_seconds,
                    input_path=input_path,
                    delimiter=delimiter,
                    ignore_first_col=ignore_first_col,
                    directed=directed,
                    **common_params,
                    **switches,
                )
                if result.timed_out:
                    print(f"      -> {variant_name} timed out at {timeout_seconds:.1f}s")
                else:
                    print(f"      -> {variant_name} finished in {result.runtime:.4f}s")
                sampled_results[variant_name].append(result)
        for variant_name, _ in variants:
            aggregated.append(average_results(size, variant_name, sampled_results[variant_name]))
    return aggregated


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="KMOR experiments")
    ap.add_argument("--worker-json", help=argparse.SUPPRESS)
    ap.add_argument("--input", required=True, help="边文件路径")
    ap.add_argument("--requests-file", required=True, help="请求文件路径")
    ap.add_argument("--source", required=True, help="车辆起点")
    ap.add_argument("--target", required=True, help="车辆终点")
    ap.add_argument("--delimiter", default="space", choices=["space", "tab", "comma"])
    ap.add_argument("--ignore-first-col", action="store_true")
    ap.add_argument("--directed", action="store_true")
    ap.add_argument("--capacity", type=int, default=4)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--time-budget", type=float, default=float("inf"))
    ap.add_argument("--detour-ratio", type=float, default=1.8)
    ap.add_argument("--max-states", type=int, default=200000)
    ap.add_argument(
        "--experiment",
        choices=["ablation", "scale_ablation"],
        default="scale_ablation",
        help="选择实验类型",
    )
    ap.add_argument(
        "--request-sizes",
        default="5,8,10,12,15",
        help="批量消融的请求规模列表，例如 5,10,15,20",
    )
    ap.add_argument("--trials", type=int, default=5, help="每个请求规模重复抽样次数")
    ap.add_argument("--seed", type=int, default=42, help="随机采样种子")
    ap.add_argument("--timeout", type=float, default=30.0, help="单次变体求解超时秒数，默认 30 秒")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    if args.worker_json:
        payload = json.loads(args.worker_json)
        requests = deserialize_requests(payload["requests"])
        graph = read_edge_list(
            args.input,
            delimiter=args.delimiter,
            directed=args.directed,
            ignore_first_col=args.ignore_first_col,
        )
        result = run_kmor_variant(
            graph,
            str(payload["source"]),
            str(payload["target"]),
            requests,
            str(payload["method_name"]),
            **dict(payload["params"]),
        )
        print(json.dumps(result.__dict__, ensure_ascii=True))
        return 0

    graph = read_edge_list(
        args.input,
        delimiter=args.delimiter,
        directed=args.directed,
        ignore_first_col=args.ignore_first_col,
    )
    requests = parse_requests_file(args.requests_file)

    common_params = {
        "capacity": args.capacity,
        "k": args.k,
        "time_budget": args.time_budget,
        "detour_ratio": args.detour_ratio,
        "max_states": args.max_states,
        "aggressive_served_pruning": args.experiment == "ablation",
    }

    if args.experiment == "ablation":
        ablation_results = run_ablation_triplet(
            graph,
            args.source,
            args.target,
            requests,
            args.timeout,
            input_path=args.input,
            delimiter=args.delimiter,
            ignore_first_col=args.ignore_first_col,
            directed=args.directed,
            **common_params,
        )
        print_ablation_table(ablation_results)

    if args.experiment == "scale_ablation":
        sizes = parse_int_list(args.request_sizes)
        valid_sizes = [size for size in sizes if 1 <= size <= len(requests)]
        if not valid_sizes:
            raise ValueError("request sizes are invalid for the provided request file")
        scale_results = run_scale_ablation(
            graph,
            args.source,
            args.target,
            requests,
            args.input,
            args.delimiter,
            args.ignore_first_col,
            args.directed,
            valid_sizes,
            args.trials,
            args.seed,
            args.timeout,
            common_params,
        )
        print_scale_ablation_table(scale_results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
