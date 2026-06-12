#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare paper datasets for KMOR experiments.

Outputs:
- effectiveness requests/config
- ablation requests/config
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple

from Dpkmor import Graph, read_edge_list


@dataclass
class DatasetConfig:
    name: str
    edge_file: str
    requests_file: str
    source: str
    target: str
    time_budget: float
    ignore_first_col: bool = True
    directed: bool = False
    num_requests: int = 0
    shortest_path_cost: float = 0.0


def resolve_endpoints(graph: Graph) -> Tuple[str, str]:
    nodes = list(graph.nodes())
    if not nodes:
        raise ValueError("graph has no nodes")
    try:
        numeric_nodes = sorted((int(node), str(node)) for node in nodes)
        return numeric_nodes[0][1], numeric_nodes[-1][1]
    except ValueError:
        sorted_nodes = sorted(str(node) for node in nodes)
        return sorted_nodes[0], sorted_nodes[-1]


def edge_weight(graph: Graph, u: str, v: str) -> float:
    for nb, w in graph.neighbors(u):
        if nb == v:
            return float(w)
    return float("inf")


def sample_side_node(
    graph: Graph,
    anchor: str,
    path_nodes: Set[str],
    rng: random.Random,
    max_walk: int = 5,
) -> str:
    current = anchor
    best = anchor
    for _ in range(max_walk):
        neighbors = [str(nb) for nb, _ in graph.neighbors(current)]
        if not neighbors:
            break
        off_path = [nb for nb in neighbors if nb not in path_nodes]
        if off_path:
            current = rng.choice(off_path)
            best = current
        else:
            current = rng.choice(neighbors)
    return best


def path_prefix_costs(graph: Graph, path: Sequence[str]) -> List[float]:
    prefix = [0.0]
    total = 0.0
    for i in range(len(path) - 1):
        total += edge_weight(graph, path[i], path[i + 1])
        prefix.append(total)
    return prefix


def choose_long_route(graph: Graph, rng: random.Random, samples: int = 24) -> Tuple[str, str, List[str], float]:
    nodes = list(graph.nodes())
    if len(nodes) < 2:
        raise ValueError("graph needs at least two nodes")

    best_source = None
    best_target = None
    best_path: List[str] = []
    best_cost = -1.0

    sampled_sources = rng.sample(nodes, min(samples, len(nodes)))
    for source in sampled_sources:
        sampled_targets = rng.sample(nodes, min(samples, len(nodes)))
        for target in sampled_targets:
            if source == target:
                continue
            path, cost = graph.shortest_path(source, target)
            if path and cost != float("inf") and cost > best_cost:
                best_source = str(source)
                best_target = str(target)
                best_path = path
                best_cost = float(cost)

    if best_source is None or best_target is None or not best_path:
        source, target = resolve_endpoints(graph)
        path, cost = graph.shortest_path(source, target)
        if not path or cost == float("inf"):
            raise RuntimeError("failed to find a reachable source-target pair")
        return source, target, path, float(cost)

    return best_source, best_target, best_path, best_cost


def sample_demand(rng: random.Random) -> int:
    x = rng.random()
    if x < 0.5:
        return 1
    if x < 0.85:
        return 2
    return 3


def sample_competitive_demand(rng: random.Random) -> int:
    x = rng.random()
    if x < 0.25:
        return 1
    if x < 0.75:
        return 2
    return 3


def clamp_index(n: int, low_frac: float, high_frac: float, rng: random.Random) -> int:
    low = max(1, min(n - 2, int(n * low_frac)))
    high = max(low, min(n - 2, int(n * high_frac)))
    return rng.randint(low, high)


def scaled_wait(shortest_cost: float, rng: random.Random, tightness: float) -> float:
    base = max(20.0, shortest_cost * tightness)
    return round(base * rng.uniform(0.85, 1.12), 1)


def make_request_with_window(
    idx: int,
    pickup_idx: int,
    dropoff_idx: int,
    path: Sequence[str],
    prefix_costs: Sequence[float],
    demand: int,
    wait_limit: float,
    rng: random.Random,
) -> Tuple[str, str, str, int, float, float]:
    pickup_eta = prefix_costs[pickup_idx]
    lead_time = wait_limit * rng.uniform(0.35, 0.75)
    request_time = round(max(0.0, pickup_eta - lead_time), 1)
    return (
        f"r{idx}",
        path[pickup_idx],
        path[dropoff_idx],
        demand,
        request_time,
        round(wait_limit, 1),
    )


def make_core_request(
    idx: int,
    path: Sequence[str],
    prefix_costs: Sequence[float],
    shortest_cost: float,
    rng: random.Random,
    wait_scale: float = 1.0,
) -> Tuple[str, str, str, int, float, float]:
    n = len(path)
    pickup_idx = clamp_index(n, 0.16, 0.34, rng)
    dropoff_idx = clamp_index(n, 0.46, 0.68, rng)
    if dropoff_idx <= pickup_idx:
        dropoff_idx = min(n - 2, pickup_idx + max(6, n // 8))
    wait_limit = scaled_wait(shortest_cost, rng, 0.035 * wait_scale)
    demand = 1 if rng.random() < 0.65 else 2
    return make_request_with_window(idx, pickup_idx, dropoff_idx, path, prefix_costs, demand, wait_limit, rng)


def make_competitive_request(
    idx: int,
    path: Sequence[str],
    prefix_costs: Sequence[float],
    shortest_cost: float,
    rng: random.Random,
    wait_scale: float = 1.0,
) -> Tuple[str, str, str, int, float, float]:
    n = len(path)
    pickup_idx = clamp_index(n, 0.24, 0.45, rng)
    dropoff_idx = clamp_index(n, 0.50, 0.76, rng)
    if dropoff_idx <= pickup_idx:
        dropoff_idx = min(n - 2, pickup_idx + max(8, n // 7))
    wait_limit = scaled_wait(shortest_cost, rng, 0.024 * wait_scale)
    demand = sample_competitive_demand(rng)
    return make_request_with_window(idx, pickup_idx, dropoff_idx, path, prefix_costs, demand, wait_limit, rng)


def make_reverse_request(
    idx: int,
    path: Sequence[str],
    prefix_costs: Sequence[float],
    shortest_cost: float,
    rng: random.Random,
    wait_scale: float = 1.0,
) -> Tuple[str, str, str, int, float, float]:
    n = len(path)
    pickup_idx = clamp_index(n, 0.46, 0.72, rng)
    dropoff_idx = clamp_index(n, 0.22, 0.55, rng)
    if dropoff_idx >= pickup_idx:
        dropoff_idx = max(1, pickup_idx - max(6, n // 9))
    wait_limit = scaled_wait(shortest_cost, rng, 0.045 * wait_scale)
    demand = 1 if rng.random() < 0.7 else 2
    return make_request_with_window(idx, pickup_idx, dropoff_idx, path, prefix_costs, demand, wait_limit, rng)


def make_ablation_decoy_request(
    idx: int,
    path: Sequence[str],
    prefix_costs: Sequence[float],
    shortest_cost: float,
    rng: random.Random,
) -> Tuple[str, str, str, int, float, float]:
    n = len(path)
    pickup_idx = clamp_index(n, 0.68, 0.88, rng)
    dropoff_idx = clamp_index(n, 0.08, 0.32, rng)
    if dropoff_idx >= pickup_idx:
        dropoff_idx = max(1, pickup_idx - max(10, n // 4))
    wait_limit = scaled_wait(shortest_cost, rng, 0.060)
    demand = 1 if rng.random() < 0.75 else 2
    return make_request_with_window(idx, pickup_idx, dropoff_idx, path, prefix_costs, demand, wait_limit, rng)


def generate_requests_for_path(
    graph: Graph,
    path: Sequence[str],
    shortest_cost: float,
    num_requests: int,
    rng: random.Random,
    profile: str = "effectiveness",
) -> List[Tuple[str, str, str, int, float, float]]:
    prefix_costs = path_prefix_costs(graph, path)
    path_nodes = set(path)
    if profile == "ablation":
        num_core = max(4, round(num_requests * 0.35))
        num_competitive = round(num_requests * 0.25)
        num_reverse = round(num_requests * 0.15)
        num_decoy = num_requests - num_core - num_competitive - num_reverse
        wait_scale = 0.90
    else:
        num_core = round(num_requests * 0.35)
        num_competitive = round(num_requests * 0.45)
        num_reverse = num_requests - num_core - num_competitive
        num_decoy = 0
        wait_scale = 1.0

    requests: List[Tuple[str, str, str, int, float, float]] = []
    idx = 1
    for _ in range(num_core):
        requests.append(make_core_request(idx, path, prefix_costs, shortest_cost, rng, wait_scale=wait_scale))
        idx += 1
    for _ in range(num_competitive):
        req = make_competitive_request(idx, path, prefix_costs, shortest_cost, rng, wait_scale=wait_scale)
        if profile == "ablation" and rng.random() < 0.55:
            name, pickup, dropoff, demand, request_time, wait_limit = req
            pickup = sample_side_node(graph, pickup, path_nodes, rng, max_walk=6)
            if rng.random() < 0.45:
                dropoff = sample_side_node(graph, dropoff, path_nodes, rng, max_walk=6)
            req = (name, pickup, dropoff, demand, request_time, wait_limit)
        requests.append(req)
        idx += 1
    for _ in range(num_reverse):
        req = make_reverse_request(idx, path, prefix_costs, shortest_cost, rng, wait_scale=wait_scale)
        if profile == "ablation":
            name, pickup, dropoff, demand, request_time, wait_limit = req
            pickup = sample_side_node(graph, pickup, path_nodes, rng, max_walk=8)
            dropoff = sample_side_node(graph, dropoff, path_nodes, rng, max_walk=8)
            req = (name, pickup, dropoff, demand, request_time, wait_limit)
        requests.append(req)
        idx += 1
    for _ in range(num_decoy):
        req = make_ablation_decoy_request(idx, path, prefix_costs, shortest_cost, rng)
        name, pickup, dropoff, demand, request_time, wait_limit = req
        if rng.random() < 0.70:
            pickup = sample_side_node(graph, pickup, path_nodes, rng, max_walk=10)
        if rng.random() < 0.70:
            dropoff = sample_side_node(graph, dropoff, path_nodes, rng, max_walk=10)
        requests.append((name, pickup, dropoff, demand, request_time, wait_limit))
        idx += 1

    rng.shuffle(requests)
    return requests


def generate_requests_for_graph(
    graph: Graph,
    num_requests: int,
    rng: random.Random,
    profile: str = "effectiveness",
) -> Tuple[str, str, float, List[Tuple[str, str, str, int, float, float]]]:
    source, target, path, shortest_cost = choose_long_route(graph, rng)
    requests = generate_requests_for_path(graph, path, shortest_cost, num_requests, rng, profile=profile)
    return source, target, shortest_cost, requests


def write_requests(path: Path, requests: Iterable[Tuple[str, str, str, int, float, float]]) -> None:
    lines = ["# Name,pickup,dropoff,demand,request_time,wait_limit"]
    for name, pickup, dropoff, demand, request_time, wait_limit in requests:
        lines.append(f"{name},{pickup},{dropoff},{demand},{request_time},{wait_limit}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_requests(path: Path) -> List[Tuple[str, str, str, int, float, float]]:
    requests: List[Tuple[str, str, str, int, float, float]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            raise ValueError(f"invalid request line in {path.name}: {raw}")
        requests.append(
            (
                parts[0],
                parts[1],
                parts[2],
                int(parts[3]),
                float(parts[4]),
                float(parts[5]),
            )
        )
    return requests


def prepare_real_dataset(
    name: str,
    edge_file: Path,
    requests_output: Path,
    num_requests: int,
    rng_seed: int,
    profile: str = "effectiveness",
    budget_multiplier: float = 1.42,
) -> DatasetConfig:
    rng = random.Random(rng_seed)
    graph = read_edge_list(str(edge_file), directed=False, ignore_first_col=True)
    source, target, shortest_cost, requests = generate_requests_for_graph(graph, num_requests, rng, profile=profile)
    write_requests(requests_output, requests)
    time_budget = round(shortest_cost * budget_multiplier, 1)
    return DatasetConfig(
        name=name,
        edge_file=edge_file.name,
        requests_file=requests_output.name,
        source=source,
        target=target,
        time_budget=time_budget,
        ignore_first_col=True,
        directed=False,
        num_requests=num_requests,
        shortest_path_cost=round(shortest_cost, 1),
    )


def prepare_synthetic_dataset(
    name: str,
    edge_file: Path,
    requests_output: Path,
    num_requests: int,
    time_budget: float | None,
    rng_seed: int,
    profile: str = "effectiveness",
    auto_budget_multiplier: float = 1.55,
) -> DatasetConfig:
    graph = read_edge_list(str(edge_file), directed=False, ignore_first_col=True)
    source, target = resolve_endpoints(graph)
    path, shortest_cost = graph.shortest_path(source, target)
    if not path or shortest_cost == float("inf"):
        rng = random.Random(rng_seed)
        source, target, path, shortest_cost = choose_long_route(graph, rng, samples=32)
    if time_budget is None or time_budget <= 0:
        time_budget = round(float(shortest_cost) * auto_budget_multiplier, 1)
    rng = random.Random(rng_seed)
    generated_requests = generate_requests_for_path(
        graph, path, float(shortest_cost), num_requests, rng, profile=profile
    )
    write_requests(requests_output, generated_requests)
    return DatasetConfig(
        name=name,
        edge_file=edge_file.name,
        requests_file=requests_output.name,
        source=source,
        target=target,
        time_budget=time_budget,
        ignore_first_col=True,
        directed=False,
        num_requests=len(generated_requests),
        shortest_path_cost=round(float(shortest_cost), 1),
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Prepare paper datasets for KMOR")
    ap.add_argument("--workdir", default=".", help="Workspace directory")
    ap.add_argument("--ol-edge", default="OL.cedge.txt")
    ap.add_argument("--tg-edge", default="TG.cedge.txt")
    ap.add_argument("--ol-requests", default="requests_ol.txt")
    ap.add_argument("--tg-requests", default="requests_tg.txt")
    ap.add_argument("--ol-ablation-requests", default="requests_ol_ablation.txt")
    ap.add_argument("--tg-ablation-requests", default="requests_tg_ablation.txt")
    ap.add_argument("--ol-num-requests", type=int, default=12)
    ap.add_argument("--tg-num-requests", type=int, default=12)
    ap.add_argument("--ol-ablation-num-requests", type=int, default=15)
    ap.add_argument("--tg-ablation-num-requests", type=int, default=15)
    ap.add_argument("--syn-m-edge", default="CD.edges.txt")
    ap.add_argument("--syn-m-requests", default="requests_cd.txt")
    ap.add_argument("--syn-m-ablation-requests", default="requests_cd_ablation.txt")
    ap.add_argument("--syn-m-num-requests", type=int, default=15)
    ap.add_argument("--syn-m-ablation-num-requests", type=int, default=18)
    ap.add_argument("--syn-l-edge", default="NY.edges.txt")
    ap.add_argument("--syn-l-requests", default="requests_ny.txt")
    ap.add_argument("--syn-l-ablation-requests", default="requests_ny_ablation.txt")
    ap.add_argument("--syn-l-num-requests", type=int, default=15)
    ap.add_argument("--syn-l-ablation-num-requests", type=int, default=18)
    ap.add_argument(
        "--syn-m-budget",
        type=float,
        default=0.0,
        help="Synthetic medium time budget; <=0 means auto from shortest path",
    )
    ap.add_argument(
        "--syn-l-budget",
        type=float,
        default=0.0,
        help="Synthetic large time budget; <=0 means auto from shortest path",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-config", default="paper_datasets.json")
    ap.add_argument("--ablation-output-config", default="paper_ablation_datasets.json")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    workdir = Path(args.workdir).resolve()

    ol_config = prepare_real_dataset(
        "OL",
        workdir / args.ol_edge,
        workdir / args.ol_requests,
        args.ol_num_requests,
        args.seed,
        profile="effectiveness",
        budget_multiplier=1.42,
    )
    tg_config = prepare_real_dataset(
        "TG",
        workdir / args.tg_edge,
        workdir / args.tg_requests,
        args.tg_num_requests,
        args.seed + 1000,
        profile="effectiveness",
        budget_multiplier=1.42,
    )
    syn_m_config = prepare_synthetic_dataset(
        "CD",
        workdir / args.syn_m_edge,
        workdir / args.syn_m_requests,
        args.syn_m_num_requests,
        args.syn_m_budget,
        args.seed + 2000,
        profile="effectiveness",
        auto_budget_multiplier=1.55,
    )
    syn_l_config = prepare_synthetic_dataset(
        "NY",
        workdir / args.syn_l_edge,
        workdir / args.syn_l_requests,
        args.syn_l_num_requests,
        args.syn_l_budget,
        args.seed + 3000,
        profile="effectiveness",
        auto_budget_multiplier=1.55,
    )

    configs = [ol_config, tg_config, syn_m_config, syn_l_config]
    output_path = workdir / args.output_config
    output_path.write_text(
        json.dumps({"datasets": [asdict(cfg) for cfg in configs]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for cfg in configs:
        print(
            f"[{cfg.name}] source={cfg.source}, target={cfg.target}, "
            f"requests={cfg.num_requests}, shortest={cfg.shortest_path_cost}, budget={cfg.time_budget}"
        )
    print(f"Saved config to {output_path.name}")

    ol_ablation = prepare_real_dataset(
        "OL",
        workdir / args.ol_edge,
        workdir / args.ol_ablation_requests,
        args.ol_ablation_num_requests,
        args.seed + 5000,
        profile="ablation",
        budget_multiplier=1.28,
    )
    tg_ablation = prepare_real_dataset(
        "TG",
        workdir / args.tg_edge,
        workdir / args.tg_ablation_requests,
        args.tg_ablation_num_requests,
        args.seed + 6000,
        profile="ablation",
        budget_multiplier=1.28,
    )
    syn_m_ablation = prepare_synthetic_dataset(
        "CD",
        workdir / args.syn_m_edge,
        workdir / args.syn_m_ablation_requests,
        args.syn_m_ablation_num_requests,
        0.0,
        args.seed + 7000,
        profile="ablation",
        auto_budget_multiplier=1.34,
    )
    syn_l_ablation = prepare_synthetic_dataset(
        "NY",
        workdir / args.syn_l_edge,
        workdir / args.syn_l_ablation_requests,
        args.syn_l_ablation_num_requests,
        0.0,
        args.seed + 8000,
        profile="ablation",
        auto_budget_multiplier=1.34,
    )
    ablation_configs = [ol_ablation, tg_ablation, syn_m_ablation, syn_l_ablation]
    ablation_path = workdir / args.ablation_output_config
    ablation_path.write_text(
        json.dumps({"datasets": [asdict(cfg) for cfg in ablation_configs]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for cfg in ablation_configs:
        print(
            f"[Ablation-{cfg.name}] source={cfg.source}, target={cfg.target}, "
            f"requests={cfg.num_requests}, shortest={cfg.shortest_path_cost}, budget={cfg.time_budget}"
        )
    print(f"Saved ablation config to {ablation_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
