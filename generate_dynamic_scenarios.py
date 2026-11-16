#!/usr/bin/env python3
"""Generate one-factor-at-a-time dynamic KMOR scenario workloads."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List

from Dpkmor import read_edge_list
from generate_dynamic_workloads import (
    collect_traffic_edges,
    generate_event_rows,
    generate_diverse_corridors,
    generate_requests,
    load_configs,
    parse_csv_values,
    path_prefix_costs,
    write_events,
    write_requests,
)


def load_scenarios(path: Path, selected: List[str]) -> List[Dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    defaults = dict(payload["defaults"])
    selected_names = {name.lower() for name in selected}
    scenarios = []
    for raw in payload["scenarios"]:
        if selected_names and str(raw["name"]).lower() not in selected_names:
            continue
        settings = dict(defaults)
        settings.update(
            {
                key: value
                for key, value in raw.items()
                if key not in {"name", "sweep", "parameter", "value"}
            }
        )
        scenarios.append(
            {
                "name": str(raw["name"]),
                "sweep": str(raw["sweep"]),
                "parameter": str(raw["parameter"]),
                "value": raw["value"],
                "settings": settings,
            }
        )
    return scenarios


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate dynamic KMOR parameter scenarios"
    )
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--dataset-config", default="paper_datasets.json")
    parser.add_argument("--scenario-config", default="dynamic_scenarios.json")
    parser.add_argument("--output-dir", default="dynamic_scenarios")
    parser.add_argument("--datasets", default="OL,TG,CD,NY")
    parser.add_argument("--scenarios", default="")
    parser.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workdir = Path(args.workdir).resolve()
    output_dir = (workdir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_config_path = (workdir / args.dataset_config).resolve()
    scenario_config_path = (workdir / args.scenario_config).resolve()
    datasets = load_configs(
        dataset_config_path,
        parse_csv_values(args.datasets, str),
    )
    scenarios = load_scenarios(
        scenario_config_path,
        parse_csv_values(args.scenarios, str),
    )
    seeds = parse_csv_values(args.seeds, int)
    if not scenarios:
        raise ValueError("no dynamic scenarios selected")

    maximum_updates = max(
        int(scenario["settings"]["edge_updates"])  # type: ignore[index]
        for scenario in scenarios
    )
    manifest: Dict[str, object] = {
        "dataset_config": str(dataset_config_path),
        "scenario_config": str(scenario_config_path),
        "seeds": seeds,
        "scenarios": [
            {
                "name": scenario["name"],
                "sweep": scenario["sweep"],
                "parameter": scenario["parameter"],
                "value": scenario["value"],
                **dict(scenario["settings"]),  # type: ignore[arg-type]
            }
            for scenario in scenarios
        ],
        "workloads": [],
    }

    for config in datasets:
        edge_path = (workdir / config.edge_file).resolve()
        print(f"[{config.name}] loading graph and source-target corridor...")
        graph = read_edge_list(
            str(edge_path),
            directed=config.directed,
            ignore_first_col=config.ignore_first_col,
        )
        corridor_count = max(
            int(scenario["settings"].get("corridor_count", 3))  # type: ignore[union-attr]
            for scenario in scenarios
        )
        corridor_max_cost_ratio = max(
            float(scenario["settings"].get("corridor_max_cost_ratio", 1.5))  # type: ignore[union-attr]
            for scenario in scenarios
        )
        corridors = generate_diverse_corridors(
            graph,
            config.source,
            config.target,
            corridor_count=corridor_count,
            max_cost_ratio=corridor_max_cost_ratio,
        )
        if not corridors:
            raise RuntimeError(f"{config.name}: source and target are unreachable")
        path, prefix, base_cost = corridors[0]

        traffic_sampling_rng = random.Random(f"{config.name}:scenario-edge-pool")
        corridor_edges, background_edges = collect_traffic_edges(
            edge_path,
            [corridor[0] for corridor in corridors],
            config,
            reservoir_size=max(100, maximum_updates * 8),
            rng=traffic_sampling_rng,
        )
        print(
            f"[{config.name}] path_nodes={len(path)} base_cost={base_cost:.2f} "
            f"corridors={len(corridors)} corridor_edges={len(corridor_edges)}"
        )

        for scenario in scenarios:
            settings = dict(scenario["settings"])  # type: ignore[arg-type]
            scenario_name = str(scenario["name"])
            horizon = max(1.0, base_cost * float(settings["horizon_ratio"]))
            scenario_dir = output_dir / scenario_name
            scenario_dir.mkdir(parents=True, exist_ok=True)

            for seed in seeds:
                # Initial requests remain identical across all one-factor sweeps.
                request_rng = random.Random(
                    f"dynamic-requests:{config.name}:{seed}"
                )
                initial, arrivals = generate_requests(
                    config.name,
                    seed,
                    path,
                    prefix,
                    horizon,
                    int(settings["initial_requests"]),
                    int(settings["arrival_count"]),
                    request_rng,
                    corridors=corridors[
                        : max(1, int(settings.get("corridor_count", corridor_count)))
                    ],
                    wait_approach_factor=float(
                        settings.get("wait_approach_factor", 1.1)
                    ),
                    wait_slack_ratio=float(settings.get("wait_slack_ratio", 0.005)),
                    demand_two_probability=float(
                        settings.get("demand_two_probability", 0.35)
                    ),
                )
                cancellation_rng = random.Random(
                    f"dynamic-cancel:{config.name}:{seed}:"
                    f"{settings['arrival_count']}:{settings['cancel_probability']}"
                )
                traffic_rng = random.Random(
                    f"dynamic-traffic:{config.name}:{seed}:"
                    f"{settings['edge_updates']}:{settings['corridor_edge_ratio']}"
                )
                rows = generate_event_rows(
                    arrivals,
                    corridor_edges,
                    background_edges,
                    int(settings["edge_updates"]),
                    float(settings["corridor_edge_ratio"]),
                    float(settings["cancel_probability"]),
                    [float(value) for value in settings["traffic_factors"]],
                    horizon,
                    request_rng,
                    cancellation_rng=cancellation_rng,
                    traffic_rng=traffic_rng,
                )

                stem = f"{config.name.lower()}_seed{seed}"
                request_path = scenario_dir / f"initial_requests_{stem}.txt"
                event_path = scenario_dir / f"events_{stem}.csv"
                write_requests(request_path, initial)
                write_events(event_path, rows)
                event_counts: Dict[str, int] = {}
                for row in rows:
                    event_type = str(row["type"])
                    event_counts[event_type] = event_counts.get(event_type, 0) + 1

                workload = {
                    "scenario": scenario_name,
                    "sweep": scenario["sweep"],
                    "parameter": scenario["parameter"],
                    "parameter_value": scenario["value"],
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
                    "settings": settings,
                }
                manifest["workloads"].append(workload)  # type: ignore[union-attr]
                print(
                    f"[{config.name} {scenario_name} seed={seed}] "
                    f"events={event_counts}"
                )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved scenario manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
