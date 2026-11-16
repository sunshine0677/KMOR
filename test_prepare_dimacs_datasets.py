from __future__ import annotations

import gzip

from prepare_dimacs_datasets import (
    convert_dimacs_graph,
    coordinate_extremes,
    parse_names,
)


def test_parse_names_normalizes_and_validates() -> None:
    assert parse_names("bay, COL") == ["BAY", "COL"]


def test_convert_dimacs_graph(tmp_path) -> None:
    source = tmp_path / "tiny.gr.gz"
    destination = tmp_path / "tiny.edges.txt"
    with gzip.open(source, "wt", encoding="ascii") as out:
        out.write("c tiny graph\n")
        out.write("p sp 3 2\n")
        out.write("a 1 2 7\n")
        out.write("a 2 3 11\n")

    assert convert_dimacs_graph(source, destination) == (3, 2)
    assert destination.read_text(encoding="utf-8").splitlines() == [
        "# edge_id start end cost",
        "0 1 2 7",
        "1 2 3 11",
    ]


def test_coordinate_extremes(tmp_path) -> None:
    coordinates = tmp_path / "tiny.co.gz"
    with gzip.open(coordinates, "wt", encoding="ascii") as out:
        out.write("c coordinates\n")
        out.write("p aux sp co 3\n")
        out.write("v 1 -10 -20\n")
        out.write("v 2 2 3\n")
        out.write("v 3 30 40\n")

    assert coordinate_extremes(coordinates) == ("1", "3")
