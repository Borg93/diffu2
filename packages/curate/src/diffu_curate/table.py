"""Rows of pydantic models in Lance tables: the schema read off the model, rows appended, and what a table holds."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import NoneType, UnionType
from typing import get_args, get_origin

import lance
import pyarrow as pa
from pydantic import BaseModel

_ARROW: dict[type, pa.DataType] = {str: pa.utf8(), int: pa.int32(), float: pa.float32(), bool: pa.bool_()}


def arrow_schema(model: type[BaseModel]) -> pa.Schema:
    """The Arrow schema of ``model``'s rows: a column a field, text, 32-bit numbers and booleans, nullable where the
    field may be None.

    Raises:
        TypeError: If a field is of another type.
    """
    fields = []
    for name, info in model.model_fields.items():
        annotation = info.annotation
        nullable = get_origin(annotation) is UnionType and NoneType in get_args(annotation)
        base = next(a for a in get_args(annotation) if a is not NoneType) if nullable else annotation
        if base not in _ARROW:
            raise TypeError(f"{model.__name__}.{name}: no Arrow type for {annotation}")
        fields.append(pa.field(name, _ARROW[base], nullable=nullable))
    return pa.schema(fields)


def append_rows(table: Path, rows: Sequence[BaseModel]) -> None:
    """Append ``rows`` (all of one model) to the Lance table at ``table``, which is made if it is not there."""
    if not rows:
        return
    data = pa.Table.from_pylist([row.model_dump() for row in rows], schema=arrow_schema(type(rows[0])))
    lance.write_dataset(data, str(table), mode="append" if table.exists() else "create")


def write_rows(table: Path, rows: Sequence[BaseModel], model: type[BaseModel]) -> None:
    """Write ``rows`` of ``model`` as the Lance table at ``table``, in place of what it held."""
    data = pa.Table.from_pylist([row.model_dump() for row in rows], schema=arrow_schema(model))
    lance.write_dataset(data, str(table), mode="overwrite")


def json_column(values: Sequence[object | None]) -> pa.Array:
    """``values`` (each a dict or list, or None) as a JSON column, which Lance stores as binary JSON and filters with
    ``json_get_string(column, 'key') = '...'`` and its sibling functions."""
    return pa.array([None if value is None else json.dumps(value, ensure_ascii=False) for value in values], pa.json_())


def add_features(table: Path, features: pa.Table, key: str = "image") -> int:
    """Join ``features`` to the Lance table at ``table`` as columns, on ``key``; return how many of its rows matched.

    Lance attaches the new columns as their own files beside the table's rows, which are not rewritten. A row of the
    table without a feature holds null; a feature row whose key the table lacks is left out. A column the table holds
    already is replaced whole, so ``features`` carries every row that is to hold a value.

    Raises:
        ValueError: If ``features`` holds two rows for one key.
    """
    keys = features.column(key)
    counts = keys.value_counts()
    twice = [entry["values"].as_py() for entry in counts if entry["counts"].as_py() > 1]
    if twice:
        raise ValueError(f"{len(twice)} keys of the features twice, the first {twice[0]}")
    held = lance.dataset(str(table))
    replaced = [name for name in features.column_names if name != key and name in held.schema.names]
    if replaced:
        held.drop_columns(replaced)
    held.merge(features, key)
    return held.count_rows(filter=f"{key} IN ({', '.join(_quoted(value) for value in keys.to_pylist())})") if len(keys) else 0


def _quoted(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def held_values(table: Path, column: str, where: str) -> set[str]:
    """The values of ``column`` in the rows of ``table`` that ``where`` (a Lance filter) keeps; none if there is no
    table."""
    if not table.exists():
        return set()
    return set(lance.dataset(str(table)).to_table(columns=[column], filter=where).column(column).to_pylist())


def compact(table: Path) -> None:
    """Merge the table's many small appends into few files."""
    if table.exists():
        lance.dataset(str(table)).optimize.compact_files()
