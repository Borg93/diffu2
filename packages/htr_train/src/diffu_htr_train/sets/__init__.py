"""The sets htr-train builds from diffu-reader-set's: a dev split by volume, Jev's pairs, and DSpark's responses."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import JsonValue

from diffu_htr_train.records import options_path


class SetRefused(ValueError):
    """A set that cannot be built as asked: its input is missing, mismatched, or holds what must not be trained on."""


def read_options(path: Path) -> JsonValue:
    """A set's own record of how it was built (its ``<stem>.options.json``, as written), None where it has none.

    Raises:
        SetRefused: If the record is not JSON.
    """
    record = options_path(path)
    if not record.is_file():
        return None
    try:
        return json.loads(record.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SetRefused(f"{record} is not JSON: {error}") from error
