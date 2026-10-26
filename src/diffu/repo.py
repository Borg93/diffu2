"""A Modular Diffusers model repo's ``modular_model_index.json``: the blocks it names and where each component loads
from. diffu-init writes it, diffu-export rewrites it for a trained checkpoint, and the page pipeline reads which
blocks (FLUX.3's or diffu's) a repo is drawn with."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError

INDEX = "modular_model_index.json"


def index_entry(library: str, cls: type, *, repo: str, revision: str | None, subfolder: str) -> list[object]:
    """One component of a model index: its library, its class and where it loads from."""
    spec = {
        "pretrained_model_name_or_path": repo,
        "revision": revision,
        "subfolder": subfolder,
        "type_hint": [library, cls.__name__],
        "variant": None,
    }
    return [library, cls.__name__, spec]


class ModelIndex(BaseModel):
    """A model index: the blocks class it names, and its other entries (the pipeline class, component entries, diffu's
    text layers) kept as they are."""

    model_config = ConfigDict(extra="allow", frozen=True, populate_by_name=True)

    blocks_class_name: str = Field(alias="_blocks_class_name")

    @classmethod
    def read(cls, repo: Path) -> Self:
        """The index of the repo at ``repo``.

        Raises:
            ValueError: If the repo has no index, or its index names no blocks class.
        """
        path = repo / INDEX
        if not path.is_file():
            raise ValueError(f"{repo} holds no {INDEX}")
        try:
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as error:
            raise ValueError(f"{path} names no _blocks_class_name: {error.error_count()} errors") from error

    def with_components(self, components: dict[str, list[object]]) -> ModelIndex:
        """This index with ``components`` (index entries by component name) loading from where they name."""
        return self.model_copy(update=components)

    def write(self, repo: Path) -> None:
        """Write the index into ``repo``."""
        (repo / INDEX).write_text(f"{json.dumps(self.model_dump(by_alias=True, mode='json'), indent=2)}\n", "utf-8")
