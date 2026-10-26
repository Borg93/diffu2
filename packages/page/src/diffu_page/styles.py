"""Styles by name: a :class:`StylePreset` holds all a page is drawn in (the archive and its years and the scan's look,
which its captions name, the hand's LoRA and a reference crop), so a job names its style instead of repeating it.

A presets file is a JSON object of presets by name (the repo's ``styles.json`` holds our per-hand LoRAs, in the archive,
years and look their evaluations drew them in); its relative paths start from the file's own directory, so it reads the
same from any working directory. A page in a named style is drawn by the preset's LoRA alone, loaded under the adapter
name ``hand``, and from the preset's reference unless the page gives its own (:class:`diffu_page.jobs.PageRequest`).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, ValidationInfo, field_validator

from diffu_page.captions.look import ScanLook
from diffu_page.captions.prompt import PageStyle
from diffu_page.captions.provenance import Provenance
from diffu_page.lora import lora_model_dir
from diffu_page.output import validation_problems
from diffu_page.plan import Reference

HAND = "hand"  # the adapter name a preset's LoRA loads under, as the hand evaluations loaded theirs
FILE_DIR = "file_dir"  # the validation context's key for the directory a presets file's relative paths start from


class StylePreset(BaseModel):
    """A style by name: the archive and years and the scan's look a page's captions name, the hand's LoRA (a NeMo LoRA
    ``model/`` dir, checked as ``--lora`` checks one) and a reference crop to draw from.

    Validated with a ``file_dir`` in its context (:func:`read_style_presets`), its relative paths start from there.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")  # a misspelt "lora" would drop the hand without a word

    provenance: Provenance
    look: ScanLook
    lora: Path | None = None
    reference: Reference | None = None

    @field_validator("lora")
    @classmethod
    def _a_nemo_lora_model_dir(cls, lora: Path | None, info: ValidationInfo) -> Path | None:
        return None if lora is None else lora_model_dir(_from_file_dir(lora, info))

    @field_validator("reference")
    @classmethod
    def _reference_from_file_dir(cls, reference: Reference | None, info: ValidationInfo) -> Reference | None:
        if reference is None:
            return None
        return reference.model_copy(update={"image": _from_file_dir(reference.image, info)})

    @property
    def style(self) -> PageStyle:
        """The style a page in this preset is captioned in: its archive and years, and its scan's look."""
        return PageStyle(provenance=self.provenance, look=self.look)

    @property
    def loras(self) -> dict[str, Path]:
        """The LoRAs a page in this preset is drawn with, by adapter name: its hand, if it has one."""
        return {} if self.lora is None else {HAND: self.lora}


def _from_file_dir(path: Path, info: ValidationInfo) -> Path:
    """``path`` as the presets file it was read from means it: from the file's directory (an absolute path stays)."""
    file_dir = (info.context or {}).get(FILE_DIR)
    return path if file_dir is None else file_dir / path


_PRESETS = TypeAdapter(dict[str, StylePreset])


def read_style_presets(path: Path) -> dict[str, StylePreset]:
    """The style presets of a presets file: a JSON object of :class:`StylePreset` s by name, their relative paths from
    the file's directory.

    Raises:
        ValueError: If the file is not such an object, naming the file, each bad preset and what is wrong with it.
    """
    try:
        return _PRESETS.validate_json(path.read_bytes(), context={FILE_DIR: path.parent})
    except ValidationError as error:
        raise ValueError(f"{path} is not a file of style presets: {validation_problems(error, 'file')}") from error
