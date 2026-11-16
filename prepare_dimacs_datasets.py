#!/usr/bin/env python3
"""Download and prepare additional DIMACS 9 road-network datasets.

The script deliberately writes a new combined configuration instead of
modifying ``paper_datasets.json``.  This keeps the four-dataset experiments
reproducible while providing a separate eight-dataset benchmark suite.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import ssl
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

from Dpkmor import read_edge_list
from prepare_paper_datasets import DatasetConfig, generate_requests_for_path, write_requests


DIMACS_BASE_URL = "https://www.diag.uniroma1.it/challenge9/data/USA-road-d"
DIMACS_DATASETS: Dict[str, Tuple[str, int, int]] = {
    "BAY": ("San Francisco Bay Area", 321_270, 800_172),
    "COL": ("Colorado", 435_666, 1_057_066),
    "FLA": ("Florida", 1_070_376, 2_712_798),
    "NW": ("Northwest USA", 1_207_945, 2_840_208),
}


def parse_names(text: str) -> list[str]:
    names = [item.strip().upper() for item in text.split(",") if item.strip()]
    unknown = [name for name in names if name not in DIMACS_DATASETS]
    if unknown:
        raise ValueError(f"unsupported DIMACS dataset(s): {', '.join(unknown)}")
    return names


def download(url: str, destination: Path, force: bool = False, attempts: int = 4) -> None:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        print(f"[reuse] {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"[download] {url}")
    ssl_context = ssl.create_default_context()
    try:
        import certifi

        ssl_context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    request = urllib.request.Request(url, headers={"User-Agent": "KMOR-DIMACS-preparer/1.0"})
    last_error: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            with urllib.request.urlopen(
                request, timeout=180, context=ssl_context
            ) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
            temporary.replace(destination)
            return
        except Exception as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt >= attempts:
                break
            delay = min(15, 2**attempt)
            print(f"[retry {attempt}/{attempts}] {exc}; waiting {delay}s")
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def convert_dimacs_graph(source_gz: Path, destination: Path) -> tuple[int, int]:
    """Convert DIMACS ``a u v weight`` records to the project edge format."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    declared_nodes = 0
    declared_arcs = 0
    written_arcs = 0
    with gzip.open(source_gz, "rt", encoding="ascii", errors="strict") as source, temporary.open(
        "w", encoding="utf-8", newline="\n"
    ) as out:
        out.write("# edge_id start end cost\n")
        for raw in source:
            if not raw or raw[0] == "c":
                continue
            parts = raw.split()
            if not parts:
                continue
            if parts[0] == "p" and len(parts) >= 4:
                declared_nodes = int(parts[-2])
                declared_arcs = int(parts[-1])
            elif parts[0] == "a" and len(parts) >= 4:
                out.write(f"{written_arcs} {parts[1]} {parts[2]} {parts[3]}\n")
                written_arcs += 1
    if declared_arcs and written_arcs != declared_arcs:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"{source_gz.name}: declared {declared_arcs} arcs but converted {written_arcs}"
        )
    temporary.replace(destination)
    return declared_nodes, written_arcs


def coordinate_extremes(coordinates_gz: Path) -> tuple[str, str]:
    """Select geographically separated endpoints using two diagonal extrema."""
    minimum: tuple[int, str] | None = None
    maximum: tuple[int, str] | None = None
    with gzip.open(coordinates_gz, "rt", encoding="ascii", errors="strict") as source:
        for raw in source:
            if not raw or raw[0] != "v":
                continue
            parts = raw.split()
            if len(parts) < 4:
                continue
            node = parts[1]
            score = int(parts[2]) + int(parts[3])
            if minimum is None or score < minimum[0]:
                minimum = (score, node)
            if maximum is None or score > maximum[0]:
                maximum = (score, node)
    if minimum is None or maximum is None or minimum[1] == maximum[1]:
        raise ValueError(f"failed to select endpoints from {coordinates_gz.name}")
    return minimum[1], maximum[1]


def prepare_dataset(
    name: str,
    edge_path: Path,
    coordinate_path: Path,
    request_path: Path,
    num_requests: int,
    seed: int,
    budget_multiplier: float,
) -> tuple[DatasetConfig, dict[str, object]]:
    import random

    source, target = coordinate_extremes(coordinate_path)
    print(f"[{name}] loading converted graph")
    graph = read_edge_list(str(edge_path), directed=True, ignore_first_col=True)
    path, shortest_cost = graph.shortest_path(source, target)
    if not path or shortest_cost == float("inf"):
        raise RuntimeError(f"{name}: selected endpoints are unreachable")
    requests = generate_requests_for_path(
        graph,
        path,
        float(shortest_cost),
        num_requests,
        random.Random(seed),
        profile="effectiveness",
    )
    write_requests(request_path, requests)
    config = DatasetConfig(
        name=name,
        edge_file=edge_path.as_posix(),
        requests_file=request_path.as_posix(),
        source=source,
        target=target,
        time_budget=round(float(shortest_cost) * budget_multiplier, 1),
        ignore_first_col=True,
        directed=True,
        num_requests=len(requests),
        shortest_path_cost=round(float(shortest_cost), 1),
    )
    details = {
        "endpoint_source": source,
        "endpoint_target": target,
        "path_nodes": len(path),
        "shortest_path_cost": round(float(shortest_cost), 1),
    }
    return config, details


def relative_config(config: DatasetConfig, workdir: Path) -> dict[str, object]:
    payload = asdict(config)
    for key in ("edge_file", "requests_file"):
        payload[key] = Path(str(payload[key])).resolve().relative_to(workdir).as_posix()
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--datasets", default="BAY,COL,FLA,NW")
    parser.add_argument("--data-dir", default="datasets/dimacs9")
    parser.add_argument("--base-config", default="paper_datasets.json")
    parser.add_argument("--output-config", default="paper_datasets_8.json")
    parser.add_argument("--metadata", default="datasets/dimacs9/metadata.json")
    parser.add_argument("--num-requests", type=int, default=15)
    parser.add_argument("--budget-multiplier", type=float, default=1.55)
    parser.add_argument("--seed", type=int, default=9042)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-convert", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workdir = Path(args.workdir).resolve()
    data_dir = (workdir / args.data_dir).resolve()
    names = parse_names(args.datasets)
    additions: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []

    for index, name in enumerate(names):
        graph_gz = data_dir / f"USA-road-d.{name}.gr.gz"
        coordinates_gz = data_dir / f"USA-road-d.{name}.co.gz"
        edge_path = data_dir / f"{name}.edges.txt"
        request_path = data_dir / f"requests_{name.lower()}.txt"
        download(f"{DIMACS_BASE_URL}/USA-road-d.{name}.gr.gz", graph_gz, args.force_download)
        download(f"{DIMACS_BASE_URL}/USA-road-d.{name}.co.gz", coordinates_gz, args.force_download)
        if args.force_convert or not edge_path.exists():
            nodes, arcs = convert_dimacs_graph(graph_gz, edge_path)
        else:
            _, expected_nodes, expected_arcs = DIMACS_DATASETS[name]
            nodes, arcs = expected_nodes, expected_arcs
            print(f"[reuse] {edge_path}")
        description, expected_nodes, expected_arcs = DIMACS_DATASETS[name]
        if (nodes, arcs) != (expected_nodes, expected_arcs):
            raise ValueError(
                f"{name}: expected ({expected_nodes}, {expected_arcs}), got ({nodes}, {arcs})"
            )
        config, details = prepare_dataset(
            name,
            edge_path,
            coordinates_gz,
            request_path,
            args.num_requests,
            args.seed + index * 1000,
            args.budget_multiplier,
        )
        additions.append(relative_config(config, workdir))
        metadata.append(
            {
                "name": name,
                "description": description,
                "source": "9th DIMACS Implementation Challenge",
                "metric": "distance",
                "nodes": nodes,
                "arcs": arcs,
                "directed": True,
                **details,
            }
        )
        print(
            f"[{name}] nodes={nodes:,} arcs={arcs:,} path_nodes={details['path_nodes']} "
            f"shortest={details['shortest_path_cost']}"
        )

    base_path = (workdir / args.base_config).resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    existing_names = {str(item["name"]).upper() for item in base["datasets"]}
    combined = list(base["datasets"])
    combined.extend(item for item in additions if str(item["name"]).upper() not in existing_names)
    write_json((workdir / args.output_config).resolve(), {"datasets": combined})
    write_json(
        (workdir / args.metadata).resolve(),
        {
            "source_url": DIMACS_BASE_URL,
            "selected": names,
            "datasets": metadata,
        },
    )
    print(f"Saved combined config to {args.output_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
