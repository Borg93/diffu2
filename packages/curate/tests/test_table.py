"""Features joined to a Lance table as columns on a key, and answers kept whole as JSON."""

from __future__ import annotations

import json
from pathlib import Path

import lance
import pyarrow as pa
import pytest

from diffu_curate.table import add_features, json_column


def scans(path: Path) -> Path:
    lance.write_dataset(pa.table({"image": ["a.jpg", "b.jpg", "c.jpg"], "lines": [10, 20, 30]}), str(path))
    return path


def test_add_features_joins_columns_on_the_key_and_leaves_rows_without_a_feature_null(tmp_path: Path) -> None:
    table = scans(tmp_path / "scans.lance")

    matched = add_features(table, pa.table({"image": ["c.jpg", "a.jpg", "z.jpg"], "kind": ["table", "text", "?"]}))

    rows = lance.dataset(str(table)).to_table().to_pylist()
    assert matched == 2
    assert rows == [
        {"image": "a.jpg", "lines": 10, "kind": "text"},
        {"image": "b.jpg", "lines": 20, "kind": None},
        {"image": "c.jpg", "lines": 30, "kind": "table"},
    ]


def test_add_features_updates_a_column_the_table_holds_and_keeps_the_values_it_does_not_name(tmp_path: Path) -> None:
    table = scans(tmp_path / "scans.lance")
    add_features(table, pa.table({"image": ["a.jpg", "c.jpg"], "kind": ["text", "blank"]}))
    lance.dataset(str(table)).create_scalar_index("kind", "BITMAP")

    add_features(table, pa.table({"image": ["a.jpg", "b.jpg"], "kind": ["table", "text"], "odd": [0.5, 0.1]}))

    held = lance.dataset(str(table))
    assert held.to_table(columns=["image", "lines", "kind", "odd"]).to_pylist() == [
        {"image": "a.jpg", "lines": 10, "kind": "table", "odd": 0.5},
        {"image": "b.jpg", "lines": 20, "kind": "text", "odd": 0.1},
        {"image": "c.jpg", "lines": 30, "kind": "blank", "odd": None},
    ]
    assert held.to_table(columns=["image"], filter="kind = 'table'").column("image").to_pylist() == ["a.jpg"]


def test_add_features_refuses_two_rows_for_one_key(tmp_path: Path) -> None:
    table = scans(tmp_path / "scans.lance")

    with pytest.raises(ValueError, match="a.jpg"):
        add_features(table, pa.table({"image": ["a.jpg", "a.jpg"], "kind": ["table", "text"]}))


def test_a_json_column_is_filtered_by_what_an_answer_holds(tmp_path: Path) -> None:
    table = scans(tmp_path / "scans.lance")
    answers = [{"layout": "table", "columns": 3}, None, {"layout": "running_text"}]

    add_features(table, pa.table({"image": ["a.jpg", "b.jpg", "c.jpg"], "said": json_column(answers)}))

    held = lance.dataset(str(table))
    tables = held.to_table(columns=["image"], filter="json_get_string(said, 'layout') = 'table'")
    assert tables.column("image").to_pylist() == ["a.jpg"]
    assert [value and json.loads(value) for value in held.to_table().column("said").to_pylist()] == answers
