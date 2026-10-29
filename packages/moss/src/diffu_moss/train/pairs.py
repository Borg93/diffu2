"""Pairs as MOSS-Image trains on them, of two kinds. A style pair: a target crop, a reference crop of the same hand (its
page or the facing page, sharing none of its lines) and the target's caption in the described reference form
(``diffu_page.data.style_pairs --reference page --caption described``: v3d), read from the image-edit manifest the
FLUX.3 runs trained on. A page pair (:class:`PagePair`): a whole single page and a sample page of its volume sharing
none of its lines, its Layout prompt in each form compiled by ``diffu_page.data.page_pairs``, read from its page-pair
manifest. :func:`read_manifest` reads either.

Each pair is drawn into one of three prompt forms, its :class:`Variant`, the mix fixed per pair by a seed
(:func:`assign_variants`): a fifth without the reference (the caption's citation dropped, its look words kept: plain
text-to-image, as MOSS-Image was trained), and of the rest half with the sample image and the look words, half with
the sample image and no look words (the plain reference caption), so the look must come from the image. Each form is
compiled as MOSS's Layout prompt of the target (:mod:`diffu_moss.layout_prompt`), as the page pipeline compiles a crop
drawn with or without its reference (``MossLayoutPromptStep``); a page pair's rows carry theirs compiled.
"""

from __future__ import annotations

import json
import random
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from diffu_moss.layout_prompt import caption_texts, layout_prompt
from diffu_moss.sample_caption import with_sample, without_look, without_sample


class Variant(StrEnum):
    """The prompt form a pair is drawn in."""

    NO_REFERENCE = "no_reference"  # no reference image: the citation dropped, the look words kept
    LOOK = "look"  # the sample image, and the look in words too
    PLAIN = "plain"  # the sample image alone: the look words dropped

    @property
    def takes_reference(self) -> bool:
        """Whether a step of this form gives the DiT the reference image."""
        return self is not Variant.NO_REFERENCE


NO_REFERENCE_SHARE = 0.2
PLAIN_SHARE = 0.4  # of all pairs: half of those drawn with the reference


class Media(BaseModel):
    """One image of a manifest row: the target to draw, or the context (the reference) to draw it from."""

    role: str
    file_name: str  # relative to the manifest's directory


class ManifestRow(BaseModel):
    """One row of the image-edit manifest a style-pair set is written as (its ``train.jsonl`` or ``val/val.jsonl``)."""

    id: str
    prompt: str
    media: list[Media]


class Pair(BaseModel):
    """A target crop, its reference crop and the target's described reference caption (citing ``<ref_image_0>``)."""

    model_config = ConfigDict(frozen=True)

    id: str
    target: Path
    reference: Path
    caption: str

    def prompt(self, variant: Variant, size: tuple[int, int]) -> str:
        """The target's MOSS Layout prompt in ``variant``'s form, on its canvas of ``size`` (width, height) px.

        Raises:
            ValueError: If the caption has no layout list, does not quote a line's text, or (for :attr:`Variant.PLAIN`)
                describes no look to drop.
        """
        match variant:
            case Variant.NO_REFERENCE:
                caption = without_sample(self.caption)
            case Variant.LOOK:
                caption = with_sample(self.caption)
            case Variant.PLAIN:
                caption = with_sample(without_look(self.caption))
        return layout_prompt(caption, caption_texts(self.caption), size)

    def texts(self) -> set[str]:
        """The texts of the target's lines, as its caption quotes them."""
        return set(caption_texts(self.caption).values())


class PagePairRow(BaseModel):
    """One row of a page-pair manifest (``diffu_page.data.page_pairs``' ``train.jsonl`` or ``val.jsonl``): a whole
    single page, a sample page of its volume and the page's MOSS Layout prompt in every form, with its length in the
    text encoder's tokens, compiled for the page's size."""

    id: str
    target: str  # relative to the manifest's directory
    reference: str
    size: tuple[int, int]  # the target's (width, height) px
    volume: str  # "<collection>/<volume>"
    prompts: dict[Variant, str]
    tokens: dict[Variant, int]


class PagePair(BaseModel):
    """A whole page, its sample page and its Layout prompt in every form."""

    model_config = ConfigDict(frozen=True)

    id: str
    target: Path
    reference: Path
    size: tuple[int, int]
    prompts: dict[Variant, str]

    def prompt(self, variant: Variant, size: tuple[int, int]) -> str:
        """The page's Layout prompt in ``variant``'s form.

        Raises:
            ValueError: If its target is not of ``size`` (width, height), the size its prompts were compiled for.
        """
        if size != self.size:
            raise ValueError(f"{self.target} is {size} px, its prompts were compiled for {self.size}")
        return self.prompts[variant]

    def texts(self) -> set[str]:
        """The texts of the page's lines, as its Layout prompt holds them."""
        items = json.loads(self.prompts[Variant.NO_REFERENCE])["items"]
        return {item["text"] for item in items if item["label"] == "text"}


type TrainingPair = Pair | PagePair


def read_pairs(manifest: Path) -> list[Pair]:
    """The pairs of an image-edit manifest, their images resolved against its directory.

    Raises:
        ValueError: If a row has not exactly one target and one context image.
    """
    pairs = []
    with manifest.open(encoding="utf-8") as rows:
        for line in rows:
            if not line.strip():
                continue
            row = ManifestRow.model_validate_json(line)
            media = {item.role: item.file_name for item in row.media}
            if sorted(media) != ["context", "target"] or len(row.media) != 2:
                raise ValueError(f"{manifest}: row {row.id} has not one target and one context image")
            pairs.append(
                Pair(
                    id=row.id,
                    target=manifest.parent / media["target"],
                    reference=manifest.parent / media["context"],
                    caption=row.prompt,
                )
            )
    return pairs


def read_page_pairs(manifest: Path) -> list[PagePair]:
    """The pairs of a page-pair manifest, their images resolved against its directory."""
    with manifest.open(encoding="utf-8") as rows:
        read = [PagePairRow.model_validate_json(line) for line in rows if line.strip()]
    return [
        PagePair(
            id=row.id,
            target=manifest.parent / row.target,
            reference=manifest.parent / row.reference,
            size=row.size,
            prompts=row.prompts,
        )
        for row in read
    ]


def read_manifest(manifest: Path) -> list[TrainingPair]:
    """The pairs of a style-pair or a page-pair manifest, told apart by its first row (a page pair's carries its
    prompts).

    Raises:
        ValueError: If a style pair's row has not exactly one target and one context image.
    """
    with manifest.open(encoding="utf-8") as rows:
        first = next((line for line in rows if line.strip()), "{}")
    if "prompts" in json.loads(first):
        return list(read_page_pairs(manifest))
    return list(read_pairs(manifest))


def apart(pairs: list[TrainingPair]) -> list[TrainingPair]:
    """The pairs whose reference holds none of their target's lines, by the texts its own row gives (a reference is an
    image of the same set, a target of its own row, by its file's stem): consecutive scans can show the same lines,
    and a reference holding the target's own lines teaches copying. A reference without a row of its own is kept."""
    texts = {pair.target.stem: pair.texts() for pair in pairs}
    return [pair for pair in pairs if not texts[pair.target.stem] & texts.get(pair.reference.stem, set())]


def assign_variants(count: int, seed: int) -> list[Variant]:
    """The prompt forms of ``count`` pairs, in pair order: exactly :data:`NO_REFERENCE_SHARE` of them (rounded)
    without the reference, :data:`PLAIN_SHARE` with the sample image and no look words, the rest with both, placed by a
    shuffle seeded by ``seed``."""
    no_reference = round(count * NO_REFERENCE_SHARE)
    plain = round(count * PLAIN_SHARE)
    variants = [Variant.NO_REFERENCE] * no_reference + [Variant.PLAIN] * plain
    variants += [Variant.LOOK] * (count - len(variants))
    random.Random(seed).shuffle(variants)
    return variants
