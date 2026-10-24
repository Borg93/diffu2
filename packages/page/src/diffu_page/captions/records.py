"""A crop's ``metadata.jsonl`` row: the imagefolder fields NeMo reads, its style and its line records."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from diffu_page.captions.look import Ink, Paper, ScanLook
from diffu_page.captions.prompt import LayoutRow, LineRecord, caption_for
from diffu_page.captions.provenance import Provenance, provenance_of
from diffu_page.crops.geometry import Crop


class CropRecord(BaseModel):
    """A ``metadata.jsonl`` row: the imagefolder fields NeMo reads (``file_name``, ``caption``) and the rest."""

    file_name: str
    caption: str
    collection: str
    page: str
    width: int
    height: int
    greyscale: bool
    paper: Paper
    ink: Ink
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


def crop_record(crop: Crop, *, file_name: str, collection: str, page: Path, look: ScanLook) -> CropRecord:
    """A training crop's record: its caption, naming the page's archive and years and the scan's look, and its lines.

    Args:
        crop: The crop.
        file_name: The crop's image, in its split's directory.
        collection: The GT collection the page belongs to.
        page: The page's PAGE XML, relative to the GT root (``<collection>/<volume>/<volume>/page/<name>.xml``).
        look: The scan, paper and ink of the crop's image.
    """
    caption = caption_for(crop, provenance_of(collection, page), look)
    return CropRecord(
        file_name=file_name,
        caption=caption.text,
        collection=collection,
        page=str(page),
        width=crop.size[0],
        height=crop.size[1],
        **look.model_dump(),
        layout=caption.layout,
        lines=caption.lines,
    )


def read_records(metadata: Path) -> list[CropRecord]:
    """The crop records of a split's ``metadata.jsonl``."""
    with metadata.open(encoding="utf-8") as rows:
        return [CropRecord.model_validate_json(row) for row in rows]
