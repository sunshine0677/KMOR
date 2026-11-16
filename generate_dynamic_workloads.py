#!/usr/bin/env python3
"""Generate reproducible dynamic KMOR workloads for paper experiments."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from Dpkmor import Graph, Request, read_edge_list


EVENT_FIELDS = [
    "type",
    "event_time",
    "edge_id",
    "u",
    "v",
    "new_weight",
    "name",
    "pickup",
    "dropoff",
    "demand",
    "request_time",
    "wait_limit",
]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    edge_file: str
    source: str
    target: str
    time_budget: float
    ignore_first_col: bool = True
    directed: bool = False


@dataclass(frozen=True)
class EdgeRecord:
    edge_id: str
    u: str
    v: str
    weight: float


def edge_key(u: str, v: str, directed: bool) -> Tuple[str, str]:
    return (u, v) if directed else tuple(sorted((u, v)))  # type: ignore[return-value]


def load_configs(path: Path, selected: Sequence[str]) -> List[DatasetConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected_names = {name.upper() for name in selected}
    configs = []
    for item in payload["datasets"]:
        if selected_names and str(item["name"]).upper() not in selected_names:
            continue
        configs.append(
            DatasetConfig(
                name=str(item["name"]),
                edge_file=str(item["edge_file"]),
                source=str(item["source"]),
                target=str(item["target"]),
                time_budget=float(item["time_budget"]),
                ignore_first_col=bool(item.get("ignore_first_col", True)),
                directed=bool(item.get("directed", False)),
            )
        )
    return configs


def path_prefix_costs(graph: Graph, path: Sequence[str]) -> List[float]:
    prefix = [0.0]
    for u, v in zip(path, path[1:]):
        weights = [weight for neighbor, weight in graph.neighbors(u) if neighbor == v]
        if not weights:
            raise ValueError(f"missing path edge {u}->{v}")
        prefix.append(prefix[-1] + min(weights))
    return prefix


def generate_diverse_corridors(
    graph: Graph,
    source: str,
    target: str,
    corridor_count: int = 3,
    max_cost_ratio: float = 1.5,
) -> List[Tuple[List[str], List[float], float]]:
    """Generate deterministic, cost-bounded source-target corridors.

    Each alternate is obtained by blocking one representative edge of an
    already accepted corridor.  This is intentionally deterministic so every
    method and repeated experiment receives exactly the same request geometry.
    """
    base_path, base_cost = graph.shortest_path(str(source), str(target))
    if not base_path or math.isinf(base_cost):
        return []

    limit = max(1, int(corridor_count))
    ratio = max(1.0, float(max_cost_ratio))
    corridors: List[Tuple[List[str], List[float], float]] = [
        (list(base_path), path_prefix_costs(graph, base_path), float(base_cost))
    ]
    known = {tuple(base_path)}
    queue: List[Tuple[List[str], frozenset[Tuple[str, str]]]] = [
        (list(base_path), frozenset())
    ]
    cursor = 0
    while cursor < len(queue) and len(corridors) < limit:
        seed_path, inherited_blocks = queue[cursor]
        cursor += 1
        edges = list(zip(seed_path, seed_path[1:]))
        middle = len(edges) // 2
        edge_order = sorted(
            range(len(edges)),
            key=lambda index: (abs(index - middle), index),
        )
        for edge_index in edge_order:
            blocked = set(inherited_blocks)
            blocked.add(edges[edge_index])
            alternate, cost = graph.shortest_path_avoiding_edges(
                str(source),
                str(target),
                blocked,
            )
            identity = tuple(alternate)
            if (
                len(alternate) < 3
                or math.isinf(cost)
                or cost > base_cost * ratio + 1e-9
                or identity in known
            ):
                continue
            known.add(identity)
            corridor = list(alternate)
            corridors.append(
                (corridor, path_prefix_costs(graph, corridor), float(cost))
            )
            queue.append((corridor, frozenset(blocked)))
            if len(corridors) >= limit:
                break
    return corridors


def parse_edge_line(
    line: str,
    line_number: int,
    ignore_first_col: bool,
) -> EdgeRecord | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    parts = stripped.replace(",", " ").split()
    if ignore_first_col:
        if len(parts) < 4:
            return None
        edge_id, u, v, weight_text = parts[:4]
    else:
        if len(parts) < 3:
            return None
        edge_id = str(line_number)
        u, v, weight_text = parts[:3]
    try:
        weight = float(weight_text)
    except ValueError:
        return None
    return EdgeRecord(str(edge_id), str(u), str(v), weight)


def collect_traffic_edges(
    edge_path: Path,
    corridor_path: Sequence[str] | Sequence[Sequence[str]],
    config: DatasetConfig,
    reservoir_size: int,
    rng: random.Random,
) -> Tuple[List[EdgeRecord], List[EdgeRecord]]:
    if corridor_path and isinstance(corridor_path[0], str):
        corridor_paths = [corridor_path]  # type: ignore[list-item]
    else:
        corridor_paths = list(corridor_path)  # type: ignore[arg-type]
    ordered_keys: List[Tuple[str, str]] = []
    corridor_keys = set()
    for path in corridor_paths:
        for u, v in zip(path, path[1:]):
            key = edge_key(u, v, config.directed)
            if key not in corridor_keys:
                corridor_keys.add(key)
                ordered_keys.append(key)
    corridor: Dict[Tuple[str, str], EdgeRecord] = {}
    background: List[EdgeRecord] = []
    background_seen = 0

    with edge_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle):
            record = parse_edge_line(line, line_number, config.ignore_first_col)
            if record is None:
                continue
            key = edge_key(record.u, record.v, config.directed)
            if key in corridor_keys:
                old = corridor.get(key)
                if old is None or record.weight < old.weight:
                    corridor[key] = record
                continue

            background_seen += 1
            if len(background) < reservoir_size:
                background.append(record)
            else:
                replacement = rng.randrange(background_seen)
                if replacement < reservoir_size:
                    background[replacement] = record

    ordered_corridor = [corridor[key] for key in ordered_keys if key in corridor]
    return ordered_corridor, background


def conditioned_poisson_times(
    count: int,
    horizon: float,
    rng: random.Random,
) -> List[float]:
    # Conditional on N events, homogeneous Poisson arrival times are sorted uniforms.
    return sorted(rng.uniform(0.05 * horizon, 0.85 * horizon) for _ in range(count))


def choose_request_nodes(
    path: Sequence[str],
    prefix: Sequence[float],
    event_time: float,
    horizon: float,
    rng: random.Random,
    initial: bool = False,
    wait_approach_factor: float = 1.1,
    wait_slack_ratio: float = 0.005,
    preferred_pickup_indices: Sequence[int] = (),
) -> Tuple[str, str, float]:
    if len(path) < 3:
        raise ValueError("source-target path needs at least three nodes")

    if initial:
        progress_distance = 0.0
        center = 1
    else:
        # Edge costs and event timestamps share the same travel-time unit.
        # A no-detour vehicle therefore advances roughly event_time cost units.
        progress_distance = min(prefix[-1], event_time)
        center = min(len(path) - 2, bisect.bisect_left(prefix, progress_distance))

    preferred = [
        index
        for index in preferred_pickup_indices
        if max(1, center) <= index <= len(path) - 2
    ]
    if preferred:
        nearest_distance = min(abs(index - center) for index in preferred)
        nearest = [
            index for index in preferred if abs(index - center) == nearest_distance
        ]
        pickup_index = rng.choice(nearest)
    else:
        pickup_window = max(1, len(path) // 10)
        pickup_low = max(1, center)
        pickup_high = min(len(path) - 2, pickup_low + pickup_window)
        pickup_index = rng.randint(pickup_low, pickup_high)

    drop_window = max(2, len(path) // 5)
    drop_low = pickup_index + 1
    drop_high = min(len(path) - 1, pickup_index + drop_window)
    drop_index = rng.randint(drop_low, drop_high)

    approach = max(0.0, prefix[pickup_index] - progress_distance)
    wait_limit = max(
        prefix[-1] * wait_slack_ratio,
        approach * wait_approach_factor + prefix[-1] * wait_slack_ratio,
    )
    return path[pickup_index], path[drop_index], wait_limit


def generate_requests(
    dataset_name: str,
    seed: int,
    path: Sequence[str],
    prefix: Sequence[float],
    horizon: float,
    initial_count: int,
    arrival_count: int,
    rng: random.Random,
    corridors: Sequence[Tuple[Sequence[str], Sequence[float], float]] | None = None,
    wait_approach_factor: float = 1.1,
    wait_slack_ratio: float = 0.005,
    demand_two_probability: float = 0.35,
) -> Tuple[List[Request], List[Request]]:
    corridor_pool = list(corridors or [(path, prefix, prefix[-1])])
    if not corridor_pool:
        raise ValueError("at least one request corridor is required")
    wait_approach_factor = max(1.0, float(wait_approach_factor))
    wait_slack_ratio = max(0.0, float(wait_slack_ratio))
    demand_two_probability = min(1.0, max(0.0, float(demand_two_probability)))
    corridor_node_sets = [set(corridor[0]) for corridor in corridor_pool]
    preferred_indices: List[List[int]] = []
    for corridor_index, (corridor_path, _, _) in enumerate(corridor_pool):
        other_nodes = set().union(
            *(
                nodes
                for index, nodes in enumerate(corridor_node_sets)
                if index != corridor_index
            )
        )
        preferred_indices.append(
            [
                index
                for index, node in enumerate(corridor_path)
                if node not in other_nodes and 0 < index < len(corridor_path) - 1
            ]
        )
    initial_requests = []
    for index in range(initial_count):
        corridor_index = index % len(corridor_pool)
        request_path, request_prefix, _ = corridor_pool[corridor_index]
        pickup, dropoff, wait_limit = choose_request_nodes(
            request_path,
            request_prefix,
            0.0,
            horizon,
            rng,
            initial=True,
            wait_approach_factor=wait_approach_factor,
            wait_slack_ratio=wait_slack_ratio,
            preferred_pickup_indices=preferred_indices[corridor_index],
        )
        initial_requests.append(
            Request(
                name=f"{dataset_name}_s{seed}_init{index + 1}",
                pickup=pickup,
                dropoff=dropoff,
                demand=2 if rng.random() < demand_two_probability else 1,
                request_time=0.0,
                wait_limit=wait_limit,
            )
        )

    arrivals = []
    for index, event_time in enumerate(
        conditioned_poisson_times(arrival_count, horizon, rng),
        start=1,
    ):
        corridor_index = rng.randrange(len(corridor_pool))
        request_path, request_prefix, _ = corridor_pool[corridor_index]
        pickup, dropoff, wait_limit = choose_request_nodes(
            request_path,
            request_prefix,
            event_time,
            horizon,
            rng,
            wait_approach_factor=wait_approach_factor,
            wait_slack_ratio=wait_slack_ratio,
            preferred_pickup_indices=preferred_indices[corridor_index],
        )
        arrivals.append(
            Request(
                name=f"{dataset_name}_s{seed}_r{index}",
                pickup=pickup,
                dropoff=dropoff,
                demand=2 if rng.random() < demand_two_probability else 1,
                request_time=event_time,
                wait_limit=wait_limit,
            )
        )
    return initial_requests, arrivals


def empty_event_row(event_type: str, event_time: float) -> Dict[str, object]:
    row: Dict[str, object] = {field: "" for field in EVENT_FIELDS}
    row["type"] = event_type
    row["event_time"] = round(event_time, 6)
    return row


def request_arrival_row(request: Request) -> Dict[str, object]:
    row = empty_event_row("RequestArrival", request.request_time)
    row.update(
        name=request.name,
        pickup=request.pickup,
        dropoff=request.dropoff,
        demand=request.demand,
        request_time=round(request.request_time, 6),
        wait_limit=round(request.wait_limit, 6),
    )
    return row


def generate_event_rows(
    arrivals: Sequence[Request],
    corridor_edges: Sequence[EdgeRecord],
    background_edges: Sequence[EdgeRecord],
    edge_update_count: int,
    corridor_edge_ratio: float,
    cancellation_probability: float,
    traffic_factors: Sequence[float],
    horizon: float,
    rng: random.Random,
    cancellation_rng: random.Random | None = None,
    traffic_rng: random.Random | None = None,
) -> List[Dict[str, object]]:
    cancel_random = cancellation_rng or rng
    traffic_random = traffic_rng or rng
    rows = [request_arrival_row(request) for request in arrivals]

    for request in arrivals:
        if cancel_random.random() >= cancellation_probability:
            continue
        cancel_time = (
            request.request_time
            + cancel_random.uniform(0.15, 0.65) * request.wait_limit
        )
        if cancel_time >= horizon:
            continue
        row = empty_event_row("RequestCancel", cancel_time)
        row["name"] = request.name
        rows.append(row)

    relevant_count = min(
        len(corridor_edges),
        round(edge_update_count * corridor_edge_ratio),
    )
    background_count = min(
        len(background_edges),
        edge_update_count - relevant_count,
    )
    selected_edges = []
    if relevant_count:
        selected_edges.extend(
            traffic_random.sample(list(corridor_edges), relevant_count)
        )
    if background_count:
        selected_edges.extend(
            traffic_random.sample(list(background_edges), background_count)
        )

    update_times = sorted(
        traffic_random.uniform(0.10 * horizon, 0.75 * horizon)
        for _ in range(len(selected_edges))
    )
    for record, start_time in zip(selected_edges, update_times):
        factor = traffic_random.choice(list(traffic_factors))
        duration = traffic_random.uniform(0.05 * horizon, 0.15 * horizon)
        end_time = min(0.98 * horizon, start_time + duration)

        start = empty_event_row("EdgeUpdate", start_time)
        start.update(
            edge_id=record.edge_id,
            u=record.u,
            v=record.v,
            new_weight=round(record.weight * factor, 6),
        )
        rows.append(start)

        restore = empty_event_row("EdgeUpdate", end_time)
        restore.update(
            edge_id=record.edge_id,
            u=record.u,
            v=record.v,
            new_weight=round(record.weight, 6),
        )
        rows.append(restore)

    event_priority = {"EdgeUpdate": 0, "RequestCancel": 1, "RequestArrival": 2}
    rows.sort(key=lambda row: (float(row["event_time"]), event_priority[str(row["type"])]))
    return rows


def write_requests(path: Path, requests: Iterable[Request]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Name", "pickup", "dropoff", "demand", "request_time", "wait_limit"])
        for request in requests:
            writer.writerow(
                [
                    request.name,
                    request.pickup,
                    request.dropoff,
                    request.demand,
                    f"{request.request_time:.6f}",
                    f"{request.wait_limit:.6f}",
                ]
            )


def write_events(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_csv_values(text: str, cast) -> List:
    return [cast(value.strip()) for value in text.split(",") if value.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate dynamic KMOR workloads")
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--config", default="paper_datasets.json")
    parser.add_argument("--output-dir", default="dynamic_workloads")
    parser.add_argument("--datasets", default="OL,TG,CD,NY")
    parser.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--initial-requests", type=int, default=4)
    parser.add_argument("--arrival-count", type=int, default=40)
    parser.add_argument("--edge-updates", type=int, default=20)
    parser.add_argument("--horizon-ratio", type=float, default=0.8)
    parser.add_argument(
        "--corridor-edge-ratio",
        "--affected-ratio",
        dest="corridor_edge_ratio",
        type=float,
        default=0.5,
        help="Fraction of traffic updates sampled from the baseline route corridor",
    )
    parser.add_argument("--cancel-probability", type=float, default=0.15)
    parser.add_argument("--traffic-factors", default="1.2,1.5,2.0")
    parser.add_argument("--corridor-count", type=int, default=3)
    parser.add_argument("--corridor-max-cost-ratio", type=float, default=1.5)
    parser.add_argument("--wait-approach-factor", type=float, default=1.1)
    parser.add_argument("--wait-slack-ratio", type=float, default=0.005)
    parser.add_argument("--demand-two-probability", type=float, default=0.35)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workdir = Path(args.workdir).resolve()
    config_path = (workdir / args.config).resolve()
    output_dir = (workdir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = parse_csv_values(args.datasets, str)
    seeds = parse_csv_values(args.seeds, int)
    traffic_factors = parse_csv_values(args.traffic_factors, float)
    configs = load_configs(config_path, datasets)
    manifest: Dict[str, object] = {
        "generator": {
            "config": str(config_path),
            "seeds": seeds,
            "initial_requests": args.initial_requests,
            "arrival_count": args.arrival_count,
            "edge_updates": args.edge_updates,
            "horizon_ratio": args.horizon_ratio,
            "corridor_edge_ratio": args.corridor_edge_ratio,
            "cancel_probability": args.cancel_probability,
            "traffic_factors": traffic_factors,
            "corridor_count": args.corridor_count,
            "corridor_max_cost_ratio": args.corridor_max_cost_ratio,
            "wait_approach_factor": args.wait_approach_factor,
            "wait_slack_ratio": args.wait_slack_ratio,
            "demand_two_probability": args.demand_two_probability,
        },
        "workloads": [],
    }

    for config in configs:
        edge_path = (workdir / config.edge_file).resolve()
        print(f"[{config.name}] loading graph and source-target corridor...")
        graph = read_edge_list(
            str(edge_path),
            directed=config.directed,
            ignore_first_col=config.ignore_first_col,
        )
        corridors = generate_diverse_corridors(
            graph,
            config.source,
            config.target,
            corridor_count=args.corridor_count,
            max_cost_ratio=args.corridor_max_cost_ratio,
        )
        if not corridors:
            raise RuntimeError(f"{config.name}: source and target are unreachable")
        path, prefix, base_cost = corridors[0]
        horizon = max(1.0, base_cost * args.horizon_ratio)

        sampling_rng = random.Random(f"{config.name}:traffic-sampling")
        corridor_edges, background_edges = collect_traffic_edges(
            edge_path,
            [corridor[0] for corridor in corridors],
            config,
            reservoir_size=max(100, args.edge_updates * 8),
            rng=sampling_rng,
        )
        print(
            f"[{config.name}] path_nodes={len(path)} base_cost={base_cost:.2f} "
            f"corridors={len(corridors)} corridor_edges={len(corridor_edges)}"
        )

        for seed in seeds:
            rng = random.Random(f"dynamic-kmor:{config.name}:{seed}")
            initial, arrivals = generate_requests(
                config.name,
                seed,
                path,
                prefix,
                horizon,
                args.initial_requests,
                args.arrival_count,
                rng,
                corridors=corridors,
                wait_approach_factor=args.wait_approach_factor,
                wait_slack_ratio=args.wait_slack_ratio,
                demand_two_probability=args.demand_two_probability,
            )
            rows = generate_event_rows(
                arrivals,
                corridor_edges,
                background_edges,
                args.edge_updates,
                args.corridor_edge_ratio,
                args.cancel_probability,
                traffic_factors,
                horizon,
                rng,
            )
            stem = f"{config.name.lower()}_seed{seed}"
            request_path = output_dir / f"initial_requests_{stem}.txt"
            event_path = output_dir / f"events_{stem}.csv"
            write_requests(request_path, initial)
            write_events(event_path, rows)

            event_counts: Dict[str, int] = {}
            for row in rows:
                event_type = str(row["type"])
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            workload = {
                "dataset": config.name,
                "seed": seed,
                "edge_file": config.edge_file,
                "initial_requests_file": str(request_path.relative_to(workdir)),
                "events_file": str(event_path.relative_to(workdir)),
                "source": config.source,
                "target": config.target,
                "time_budget": config.time_budget,
                "ignore_first_col": config.ignore_first_col,
                "directed": config.directed,
                "base_cost": base_cost,
                "path_nodes": len(path),
                "corridor_count": len(corridors),
                "corridor_costs": [corridor[2] for corridor in corridors],
                "horizon": horizon,
                "event_counts": event_counts,
            }
            manifest["workloads"].append(workload)  # type: ignore[union-attr]
            print(f"[{config.name} seed={seed}] events={event_counts}")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved workload manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
