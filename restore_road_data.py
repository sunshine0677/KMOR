#!/usr/bin/env python3
"""Restore compressed road files and verify every staged input SHA-256."""

import gzip
import hashlib
import json
import shutil
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify compressed and plain inputs without extracting")
    args = parser.parse_args()
    records = json.loads((ROOT / "DATA_FILES.json").read_text(encoding="utf-8"))["files"]
    for item in records:
        source = ROOT / item["uploaded_path"]
        target = ROOT / item["required_path"]
        if not source.is_file():
            raise FileNotFoundError(source)
        if args.verify_only and item["compressed"]:
            value = hashlib.sha256()
            with gzip.open(source, "rb") as inp:
                for block in iter(lambda: inp.read(1024 * 1024), b""):
                    value.update(block)
            if value.hexdigest() != item["sha256_raw"]:
                raise ValueError(f"Compressed input does not match original: {source}")
            print(f"OK {item['dataset']} {item['kind']}: {item['uploaded_path']}")
            continue
        if item["compressed"] and not target.exists():
            with gzip.open(source, "rb") as inp, target.open("xb") as out:
                shutil.copyfileobj(inp, out, 1024 * 1024)
        if not target.is_file() or sha256(target) != item["sha256_raw"]:
            raise ValueError(f"Missing or mismatched input: {target}")
        print(f"OK {item['dataset']} {item['kind']}: {item['required_path']}")


if __name__ == "__main__":
    main()
