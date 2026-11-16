#!/usr/bin/env python3
"""Validate archived dynamic inputs and create optional portable manifest copies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_input(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"portable workload input must be relative: {value}")
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"workload input escapes repository: {value}")
    if not target.is_file():
        raise FileNotFoundError(target)
    return target


def portable_manifest(root: Path, source: Path) -> dict:
    manifest = json.loads(source.read_text(encoding="utf-8"))
    result = dict(manifest)
    for key in ("dataset_config", "scenario_config"):
        historical = Path(manifest[key])
        # The archived absolute prefix is not trusted; match only a known
        # configuration filename already present at the repository root.
        relative = historical.name if historical.is_absolute() else manifest[key]
        resolve_input(root, relative)
        result[key] = relative
    for workload in manifest.get("workloads", []):
        for key in ("edge_file", "initial_requests_file", "events_file"):
            resolve_input(root, workload[key])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repository_root())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--write-portable", type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    source = args.manifest if args.manifest.is_absolute() else root / args.manifest
    source = source.resolve()
    result = portable_manifest(root, source)
    print(f"Validated {len(result.get('workloads', []))} workloads from {source}")
    if args.write_portable:
        target = args.write_portable if args.write_portable.is_absolute() else root / args.write_portable
        target = target.resolve()
        if target == source:
            parser.error("portable output must not overwrite the historical manifest")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote portable copy: {target}")


if __name__ == "__main__":
    main()
