"""A crop's ``metadata.jsonl`` row: the imagefolder fields NeMo reads, its style and its line records."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from diffu_page.captions.look import ScanLook
from diffu_page.captions.prompt import LayoutRow, LineRecord
from diffu_page.captions.provenance import Provenance, provenance_of


class CropRecord(BaseModel):
    """A ``metadata.jsonl`` row: the imagefolder fields NeMo reads (``file_name``, ``caption``) and the rest."""

    file_name: str
    caption: str
    collection: str
    page: str
    width: int
    height: int
    greyscale: bool
    paper: str
    ink: str
    layout: list[LayoutRow]
    lines: list[LineRecord]

    @property
    def provenance(self) -> Provenance:
        """The archive and years the crop's caption names, from its collection and page path."""
        return provenance_of(self.collection, Path(self.page))

    @property
    def look(self) -> ScanLook:
        """The scan, paper and ink the crop's caption names."""
        return ScanLook(greyscale=self.greyscale, paper=self.paper, ink=self.ink)


def read_records(metadata: Path) -> list[CropRecord]:
    """The crop records of a split's ``metadata.jsonl``."""
    with metadata.open(encoding="utf-8") as rows:
        return [CropRecord.model_validate_json(row) for row in rows]
