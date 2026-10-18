"""Generation config for the ``diffu-page`` CLI — one typed, YAML-loadable description of a corpus run.

Everything that defines *what corpus to generate* — page geometry, layout template, which renderer and its
params, and the scan-augmentation knobs — lives on :class:`GenConfig`. Load it from a YAML file with
:func:`load_gen_config`; the CLI then overrides a few common fields from flags (CLI > YAML > default).
Adding a knob is one field here (or on :class:`~diffu_page.augment.AugmentConfig`) — instantly settable
from YAML, no CLI plumbing.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from .augment import AugmentConfig
from .labels import FORMATS
from .layout import TEMPLATES

# Derived from the layout registry so every template is reachable from YAML/--template and adding
# a new one is a single edit in layout.TEMPLATES.
Template = StrEnum("Template", {name: name for name in TEMPLATES})

# Derived from the label-format registry so adding a format is a single edit in labels.FORMATS.
Fmt = StrEnum("Fmt", {name: name for name in FORMATS})


class Renderer(StrEnum):
    placeholder = "placeholder"
    diffu = "diffu"
    eruku = "eruku"


class GenConfig(BaseModel):
    """A full ``diffu-page`` corpus run described in one place. Load from YAML via :func:`load_gen_config`;
    the CLI overrides individual fields. Unknown keys are rejected so a typo in the YAML fails loudly."""

    model_config = {"frozen": True, "extra": "forbid"}

    # page geometry + layout
    width: int = Field(default=1654, gt=0)
    height: int = Field(default=2339, gt=0)
    template: Template = Template.single_column
    n_columns: int = Field(default=2, ge=1)
    binding_prob: float = Field(default=0.4, ge=0.0, le=1.0)  # fraction of pages drawn with a book binding
    line_height: int = Field(default=58, ge=8)  # body text height in px — smaller packs more lines per page
    line_gap: float = Field(default=0.18, ge=0.0)  # extra gap between lines, as a fraction of line_height
    para_gap: float = Field(default=0.4, ge=0.0)  # blank gap between paragraphs, as a fraction of line_height
    furniture_prob: float = Field(  # fraction of pages that get a header + marginalia + page number
        default=0.5, ge=0.0, le=1.0
    )
    typeset_furniture: bool = False  # table headers/row-numbers/money as PRE-PRINTED type; else handwritten
    # per BODY LINE, so keep tiny — at 0.06 every ~15-line page had an edit, which looks absurd; real
    # manuscripts show the odd correction, not one on every page.
    insertion_prob: float = Field(default=0.012, ge=0.0, le=1.0)  # interlinear add above a line
    deletion_prob: float = Field(default=0.008, ge=0.0, le=1.0)  # a struck-through deletion

    # rendering
    renderer: Renderer = Renderer.placeholder
    ckpt: Path | None = None  # Diffu checkpoint (renderer=diffu)
    style: Path | None = None  # writer-style reference line (diffu/eruku)
    styles: list[Path] = Field(default_factory=list)  # writer-style POOL: sampled per page for multi-writer
    cfg_scale: float = 3.0  # Diffu CFG scale
    eruku_cfg: float = 1.25  # Eruku CFG scale
    steps: int = Field(default=24, ge=1)  # Diffu sampling steps
    fast: bool = False  # batched rendering — faster for big runs, but spreads strokes; per-line is sharper
    lora: str | None = None  # Stage-3 LoRA adapter (dir or Hub id) applied on top of ckpt (renderer=diffu)
    lora_scale: float = 1.0  # LoRA strength in [0, 1+] (set_adapters weight); 1.0 = full

    # io
    paper_dir: Path | None = None  # real blank-paper crops to seat lines on (else procedural)
    formats: list[Fmt] = Field(default_factory=lambda: [Fmt.json])  # label formats to write
    crop_lines: bool = False  # also export per-line image+text crops (TrOCR/PyLaia-style) + lines.jsonl
    val_frac: float = Field(default=0.1, ge=0.0, le=1.0)  # manifest train/val/test split fractions; a page
    test_frac: float = Field(
        default=0.1, ge=0.0, le=1.0
    )  # (and all its line crops) lands wholly in one split

    # scan look
    augment: AugmentConfig = Field(default_factory=AugmentConfig)

    @model_validator(mode="after")
    def _splits_within_one(self) -> GenConfig:
        if self.val_frac + self.test_frac > 1.0:
            raise ValueError(f"val_frac + test_frac must be <= 1.0 (got {self.val_frac + self.test_frac})")
        return self


def load_gen_config(path: Path | None) -> GenConfig:
    """Load a :class:`GenConfig` from a YAML file, or return defaults when ``path`` is None. Unknown keys
    and out-of-range values fail fast with a Pydantic ``ValidationError``."""
    if path is None:
        return GenConfig()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return GenConfig.model_validate(data)
