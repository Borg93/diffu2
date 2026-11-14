"""Rows of pydantic models in Lance tables: the schema read off the model, rows appended, and what a table holds."""

from __future__ import annotations

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
