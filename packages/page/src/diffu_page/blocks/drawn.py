"""A drawn crop, judged and labelled, and how the crops of a page are named and seeded."""

from __future__ import annotations

from PIL import Image
from pydantic import BaseModel, ConfigDict

from diffu_page.judge import LineVerdict
from diffu_page.labels import LabelFormat, PageLabels, export
from diffu_page.plan import CropPlan


def crop_identity(page_name: str, seed: int, k: int) -> tuple[str, int]:
    """Crop ``k`` of a page: its name and the noise seed it is drawn from."""
    return f"{page_name}_{k}", seed + k


class DrawnCrop(BaseModel):
    """One drawn crop: its name and plan, its image, the judge's verdict on each line, its labels and the XML format
    they are written in."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    plan: CropPlan
    image: Image.Image
    verdicts: list[LineVerdict]
    labels: PageLabels
    label_format: LabelFormat

    @property
    def labels_xml(self) -> str:
        """The labels as ``label_format`` XML."""
        return export(self.labels, self.label_format)

    @property
    def unclear_lines(self) -> int:
        """How many of the crop's lines its labels tag ``unclear``."""
        return sum(line.tag == "unclear" for line in self.labels.lines)

    @property
    def line_reads(self) -> list[str]:
        """What the judge read in each line box."""
        return [verdict.read for verdict in self.verdicts]

    @property
    def line_cers(self) -> list[float]:
        """Each line's read-back CER against its text."""
        return [verdict.cer for verdict in self.verdicts]
