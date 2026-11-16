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

    A column new to the table is attached as its own files beside the rows (Lance ``merge``); a row without a feature
    holds null. A column the table holds already is updated in place, fragment by fragment: rows the features name
    take the new value, the others keep theirs, and no row moves. A feature row whose key the table lacks is left
    out. Indexes on updated columns are brought up to date. Nothing else may write the table meanwhile: both
    operations conflict with every concurrent write.

    Raises:
        ValueError: If ``features`` holds two rows for one key.
    """
    keys = features.column(key)
    twice = [entry["values"].as_py() for entry in keys.value_counts() if entry["counts"].as_py() > 1]
    if twice:
        raise ValueError(f"{len(twice)} keys of the features twice, the first {twice[0]}")
    held = lance.dataset(str(table))
    matched = len(set(keys.to_pylist()) & set(held.to_table(columns=[key]).column(key).to_pylist()))
    names = [name for name in features.column_names if name != key]
    known = [name for name in names if name in held.schema.names]
    if len(known) < len(names):
        held.merge(features.select([key, *(name for name in names if name not in known)]), key)
    if known:
        updated, modified = [], set()
        for fragment in held.get_fragments():
            metadata, fields = fragment.update_columns(features.select([key, *known]), left_on=key, right_on=key)
            updated.append(metadata)
            modified.update(fields)
        operation = lance.LanceOperation.Update(updated_fragments=updated, fields_modified=sorted(modified))
        held = lance.LanceDataset.commit(str(table), operation, read_version=held.version)
        if held.describe_indices():
            held.optimize.optimize_indices()
    return matched


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
