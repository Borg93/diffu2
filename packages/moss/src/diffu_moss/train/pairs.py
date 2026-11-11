"""Pairs as MOSS-Image trains on them, of two kinds. A style pair: a target crop, a reference crop of the same hand (its
page or the facing page, sharing none of its lines) and the target's caption in the described reference form
(``diffu_page.data.style_pairs``, v3d's set), read from the image-edit manifest FLUX.3's style-reference LoRA trains
on. A page pair (:class:`PagePair`): a whole single page, or a window of one, and a sample page of its volume sharing
none of its lines, its Layout prompt in each form compiled by ``diffu_page.data.page_pairs``, read from its page-pair
manifest. :func:`read_manifest` reads either.

Each pair is drawn into one prompt form, its :class:`Variant`, by a mix of forms fixed per pair by a seed
(:func:`assign_variants`, :data:`MIXES`). The named mix (:attr:`PromptMix.NAMED`, the default): a fifth without the
reference (the caption's citation dropped, its look words kept: plain text-to-image, as MOSS-Image was trained), a
fifth with the sample image and the look words, a fifth with the sample image and no look words (a page pair's prompt
still names its hand), and two fifths with the sample image alone, naming no hand and no look, its paper "the paper of
the sample image", so the hand and the look must come from the image. Three of its four forms name a page pair's
archive and years, and with them its hand: trained so, a model draws a page's hand from the name and leaves its sample
unread. The content mix (:attr:`PromptMix.CONTENT`) names none: three fifths with the sample image alone, a fifth with
the sample image and the look in words but no archive, years or hand, and a fifth without the reference naming nothing
at all, the page's content alone (the reference dropped, as guidance needs). A style pair's caption names no hand, so
its plain and sample forms differ in the paper alone; it describes the reference's hand in words, so it has no form of
the content mix but the sample form. Each form is compiled as MOSS's Layout prompt of the target
(:mod:`diffu_moss.layout_prompt`), as the page pipeline compiles a crop drawn with or without its reference
(``MossLayoutPromptStep``); a page pair's rows carry theirs compiled.

A page pair's sample is shrunk under half a megapixel by default, or cut at its target's writing scale
(:attr:`SampleSize.LEGIBLE`): the rule its row records, which a cache, a training and a model record after it, so that
drawing sizes a sample as its model was trained on.
"""

from __future__ import annotations

import json
import random
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from diffu_moss.layout_prompt import caption_texts, layout_prompt
from diffu_moss.sample_caption import SAMPLE_PAPER_ROW, with_sample, without_look, without_sample


class Variant(StrEnum):
    """The prompt form a pair is drawn in."""

    NO_REFERENCE = "no_reference"  # no reference image: the citation dropped, the look words kept
    LOOK = "look"  # the sample image, and the look in words too
    PLAIN = "plain"  # the sample image, the look words dropped (a page pair's prompt still names its hand)
    SAMPLE = "sample"  # the sample image alone: no hand named, no look words, the paper the sample image's
    SAMPLE_LOOK = "sample_look"  # the sample image and the look in words, no archive, years or hand named
    GENERIC = "generic"  # no reference image, nothing named: no archive, years, hand or look, the content alone

    @property
    def takes_reference(self) -> bool:
        """Whether a step of this form gives the DiT the reference image."""
        return self not in (Variant.NO_REFERENCE, Variant.GENERIC)


class PromptMix(StrEnum):
    """Which prompt forms a cache's entries take, in which shares (:data:`MIXES`)."""

    NAMED = "named"  # the forms first trained: three of four name a page's archive and years
    CONTENT = "content"  # no form names an archive, years or hand: the hand only from the sample image


class MixShares(BaseModel):
    """A mix of prompt forms: the share of all pairs each form of ``shares`` takes (rounded), in that order, and the
    form that takes the rest."""

    model_config = ConfigDict(frozen=True)

    shares: dict[Variant, float]
    rest: Variant

    @property
    def forms(self) -> list[Variant]:
        """The mix's forms in :class:`Variant`'s order: the forms a pair of an every-form cache takes."""
        return [variant for variant in Variant if variant in self.shares or variant is self.rest]


MIXES = {
    PromptMix.NAMED: MixShares(
        shares={Variant.NO_REFERENCE: 0.2, Variant.PLAIN: 0.2, Variant.SAMPLE: 0.4}, rest=Variant.LOOK
    ),
    PromptMix.CONTENT: MixShares(shares={Variant.GENERIC: 0.2, Variant.SAMPLE_LOOK: 0.2}, rest=Variant.SAMPLE),
}


class SampleSize(StrEnum):
    """How a sample page is sized before the DiT reads it, in training and so in drawing (a model's ``sample_size``)."""

    HALF = "half"  # shrunk under half a megapixel (diffu_page's as_reference), whatever its target
    LEGIBLE = "legible"  # at its target's writing scale: a page's at its pixel count, a window's at its line pitch


class Media(BaseModel):
    """One image of a manifest row: the target to draw, or the context (the reference) to draw it from."""

    role: str
    file_name: str  # relative to the manifest's directory


class ManifestRow(BaseModel):
    """One row of the image-edit manifest a style-pair set is written as (its ``train.jsonl`` or ``val/val.jsonl``)."""

    id: str
    prompt: str
    media: list[Media]


STYLE_PAIR_FORMS = frozenset({Variant.NO_REFERENCE, Variant.LOOK, Variant.PLAIN, Variant.SAMPLE})


class Pair(BaseModel):
    """A target crop, its reference crop and the target's described reference caption (citing ``<ref_image_0>``). Its
    reference was shrunk under half a megapixel (``SampleSize.HALF``)."""

    model_config = ConfigDict(frozen=True)

    id: str
    target: Path
    reference: Path
    caption: str
    sample_size: SampleSize = SampleSize.HALF

    def prompt(self, variant: Variant, size: tuple[int, int]) -> str:
        """The target's MOSS Layout prompt in ``variant``'s form, on its canvas of ``size`` (width, height) px. The
        caption names no hand, so its plain and sample forms differ in the paper row alone: the sample form's cites
        the sample image's paper.

        Raises:
            ValueError: If the caption has no layout list, does not quote a line's text, (for the plain and the sample
                form) describes no look to drop, or ``variant`` is no form of a style pair (:meth:`forms`).
        """
        match variant:
            case Variant.NO_REFERENCE:
                caption = without_sample(self.caption)
            case Variant.LOOK:
                caption = with_sample(self.caption)
            case Variant.PLAIN:
                caption = with_sample(without_look(self.caption))
            case Variant.SAMPLE:
                caption = with_sample(without_look(self.caption, paper_row=SAMPLE_PAPER_ROW))
            case _:
                raise ValueError(f"{self.id}: a style pair has no {variant} form (its forms: {sorted(self.forms())})")
        return layout_prompt(caption, caption_texts(self.caption), size)

    def forms(self) -> frozenset[Variant]:
        """The forms its caption is compiled in: not the content mix's own, as its words describe the hand."""
        return STYLE_PAIR_FORMS

    def sample_id(self) -> str:
        """Its reference's name in its set: a target's crop of its own row, by its file's stem."""
        return self.reference.stem

    def texts(self) -> set[str]:
        """The texts of the target's lines, as its caption quotes them."""
        return set(caption_texts(self.caption).values())


class PageWindow(BaseModel):
    """What a window row's target is a window of: its page's id (its row's, when the page is a pair) and its rectangle
    there, px of the page's own coordinates."""

    page: str
    rect: tuple[float, float, float, float]  # x0, y0, x1, y1


class PagePairRow(BaseModel):
    """One row of a page-pair manifest (``diffu_page.data.page_pairs``' ``train.jsonl`` or ``val.jsonl``): a whole
    single page, or a window of one (``window``), a sample page of its volume and the target's MOSS Layout prompt in
    every form, with its length in the text encoder's tokens, compiled for the target's size (a set built before a form
    lacks it: ``diffu_page.data.page_pair_forms`` adds it). A page's row has no ``window`` key, so a set built without
    windows is written as one was before them. Its sample is its sample page's ``refs/`` image, shrunk under half a
    megapixel, or, cut for this row at its target's writing scale (``sample_size`` legible:
    ``diffu_page.data.page_pair_samples``), its own image, the sample page then named by ``sample``; a row of a shrunk
    sample has neither key."""

    id: str
    target: str  # relative to the manifest's directory
    reference: str
    size: tuple[int, int]  # the target's (width, height) px
    volume: str  # "<collection>/<volume>"
    prompts: dict[Variant, str]
    tokens: dict[Variant, int]
    window: PageWindow | None = Field(default=None, exclude_if=lambda window: window is None)  # None: a whole page
    sample_size: SampleSize = Field(default=SampleSize.HALF, exclude_if=lambda size: size is SampleSize.HALF)
    sample: str | None = Field(default=None, exclude_if=lambda sample: sample is None)  # its sample page's id


class PagePair(BaseModel):
    """A whole page, its sample page (by its id, ``sample``, when its image is not named by it), how the sample was
    sized and the page's Layout prompt in every form."""

    model_config = ConfigDict(frozen=True)

    id: str
    target: Path
    reference: Path
    size: tuple[int, int]
    prompts: dict[Variant, str]
    sample_size: SampleSize = SampleSize.HALF
    sample: str | None = None

    def forms(self) -> frozenset[Variant]:
        """The forms its row holds a prompt in."""
        return frozenset(self.prompts)

    def sample_id(self) -> str:
        """Its sample page's id: its reference's file's stem, a target of the set's own row, unless cut for this row."""
        return self.reference.stem if self.sample is None else self.sample

    def prompt(self, variant: Variant, size: tuple[int, int]) -> str:
        """The page's Layout prompt in ``variant``'s form.

        Raises:
            ValueError: If its target is not of ``size`` (width, height), the size its prompts were compiled for, or
                its row has no prompt in that form (its set was built before the form).
        """
        if size != self.size:
            raise ValueError(f"{self.target} is {size} px, its prompts were compiled for {self.size}")
        if variant not in self.prompts:
            raise ValueError(
                f"{self.id} has no {variant} prompt, built before that form: give its manifest every form "
                "(diffu_page.data.page_pair_forms)"
            )
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
            sample_size=row.sample_size,
            sample=row.sample,
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
    image of the same set, a target of its own row, by its id: :meth:`PagePair.sample_id`): consecutive scans can show
    the same lines, and a reference holding the target's own lines teaches copying. A reference without a row of its
    own is kept."""
    texts = {pair.target.stem: pair.texts() for pair in pairs}
    return [pair for pair in pairs if not texts[pair.target.stem] & texts.get(pair.sample_id(), set())]


def assign_variants(count: int, seed: int, mix: PromptMix = PromptMix.NAMED) -> list[Variant]:
    """The prompt forms of ``count`` pairs, in pair order: of each form of ``mix``'s shares (:data:`MIXES`) exactly
    its share of them (rounded), its ``rest`` form the rest, placed by a shuffle seeded by ``seed`` from a stream of
    its own. A cache shuffles its pairs by the seed itself, and two lists of one length shuffled by one stream take
    the same order: drawn from the pairs' stream, each pair's form would be the form at its own place in the manifest
    (a page-pair manifest's pages, written first, all the first forms and its windows the last)."""
    shares = MIXES[mix]
    variants = [variant for variant, share in shares.shares.items() for _ in range(round(count * share))]
    variants += [shares.rest] * (count - len(variants))
    random.Random(f"forms/{seed}").shuffle(variants)  # apart from the stream that shuffles a cache's pairs
    return variants
