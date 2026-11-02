"""diffu-moss-cache-pairs: style pairs or page pairs as a reference training reads them, encoded once on the GPU.

    diffu-moss-cache-pairs --model runs/moss/base --pairs data_flux3/style_pairs_v3d/train.jsonl \\
        --out data_flux3/cache_moss_ref_v3d --limit 4000
    diffu-moss-cache-pairs --model runs/moss/base --pairs data_flux3/style_pairs_v3d/val/val.jsonl \\
        --out data_flux3/cache_moss_ref_v3d_val --every-form
    diffu-moss-cache-pairs --model runs/moss/base --pairs data_flux3/page_pairs/train.jsonl \\
        --out data_flux3/cache_moss_pages

The manifest is a style-pair set's or a page-pair set's (:func:`diffu_moss.train.pairs.read_manifest`): a page pair's
row carries its Layout prompt in every form, a style pair's caption is compiled into it here; both are cached alike.
The pairs whose reference repeats their target's lines are left out (:func:`diffu_moss.train.pairs.apart`); the rest
are shuffled by ``--seed`` and the first ``--limit`` kept, each in the prompt form the mix gives it
(:func:`diffu_moss.train.pairs.assign_variants`), or in every form (``--every-form``, a validation set). Each entry is
encoded exactly as drawing encodes it, by the MOSS model directory's own components loaded as the page pipeline loads
them (the text encoder in bfloat16, the VAE in float32): the target's Layout prompt by ``MossTextEncoderStep``, the
target and the reference by :func:`diffu_moss.latents.encode_latents` from :func:`diffu_moss.latents.image_pixels` (a
style pair's reference is already on the reference grid, as ``ReferenceGridStep`` puts one). Written under ``--out``:
one safetensors file per entry (``prompt_embeds``, ``target`` and, drawn with the sample, ``reference``, in bfloat16;
the entry's facts in its metadata), each written whole before it is named, so a stopped run resumes where it stopped;
then ``entries.jsonl`` and ``cache.json``, last.

With ``--latents-only`` the text encoder is not loaded and no entry holds ``prompt_embeds``: a training that encodes
each prompt in its step (``diffu-moss-train-ref --encode-text``, :class:`PromptEncoder`) reads such a cache; the
entries still give their prompts and token counts (by the tokenizer, as the encoder reads them).

With ``--restyle-share s`` the share ``s`` of the entries in the sample form (rounded), picked by a stream seeded by
``--seed`` apart from the one that picks the pairs and forms (so the cache holds the same entries as without), each
take one look change drawn from that stream (:mod:`diffu_moss.train.restyle`: microfilm, colourised or tinted), the
same for the target and its sample, before both are encoded: the sample form names no look, so its prompt stays true
and the look can only come from the sample. Each such entry records its change (``restyle``, its kind and parameters),
the record how many took each kind; the default, 0, caches the images as scanned, every file as before.
``--hand-restyle-share s`` gives the share ``s`` of the entries in the sample form, picked by a stream of its own
(so the look changes fall where they would without it, and an entry may take both), one hand change each
(:class:`diffu_moss.train.restyle.HandRestyle`: its strokes, slant and width), the same for the target and its sample,
before any look change; the target's prompt moves with its lines (``HandRestyle.moved_prompt``) and its size is the
changed page's. An entry whose change would move a line off its page or onto another is cached as planned without it,
its reason recorded (``hand_refused``); the record counts the changes made and the refusals by reason.

``--mix content`` draws the entries' forms from the content mix (:data:`diffu_moss.train.pairs.MIXES`: no form names
an archive, years or hand), which a page-pair manifest's rows must hold (``diffu_page.data.page_pair_forms`` adds
them); the default, ``named``, draws them as before. A cache records its mix, and how its pairs' samples were sized
(``sample_size``, which every pair must share: a training records it beside its checkpoints, and drawing sizes a
sample by it), each left out when it is the first (``named``, ``half``).
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Annotated

import torch
import typer
from diffusers.modular_pipelines import ModularPipeline, SequentialPipelineBlocks
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor

from diffu_core.loading import COMPONENT_DTYPES, require_loaded
from diffu_moss.latents import encode_latents, image_pixels
from diffu_moss.modular_pipelines.moss import MossModularPipeline
from diffu_moss.modular_pipelines.moss.encoders import MossReferenceEncodeStep, MossTextEncoderStep, prompt_tokens
from diffu_moss.train.pairs import (
    MIXES,
    PromptMix,
    SampleSize,
    TrainingPair,
    Variant,
    apart,
    assign_variants,
    read_manifest,
)
from diffu_moss.train.records import CACHE_RECORD, CacheRecord
from diffu_moss.train.restyle import (
    HandRefused,
    HandRestyle,
    Restyle,
    draw_hand_restyles,
    draw_restyles,
    restyled,
)

app = typer.Typer(add_completion=False, rich_markup_mode="rich")
console = Console()

ENTRIES = "entries.jsonl"
CACHE_DTYPE = torch.bfloat16
ENCODERS = ("text_encoder", "tokenizer", "vae")
LATENT_ENCODERS = ("tokenizer", "vae")  # a latents-only cache's: the tokenizer counts each prompt's tokens
TEXT_ENCODERS = ("text_encoder", "tokenizer")


class CacheEntry(BaseModel):
    """One cached training sample: a pair in one prompt form, its tensors in ``file``, its target's size and its
    sample's as encoded, the look change and the hand change its target and sample took (``restyle``, ``hand``), and
    why a hand change planned for it was refused (``hand_refused``); each of the last three left out of the entry's
    JSON when there is none, so an entry of images as scanned is written as before."""

    model_config = ConfigDict(frozen=True)

    file: str
    pair: str
    variant: Variant
    prompt: str
    tokens: int
    target_size: tuple[int, int]  # (width, height) px
    reference_size: tuple[int, int] | None
    restyle: Restyle | None = Field(default=None, exclude_if=lambda restyle: restyle is None)
    hand: HandRestyle | None = Field(default=None, exclude_if=lambda hand: hand is None)
    hand_refused: str | None = Field(default=None, exclude_if=lambda reason: reason is None)


class PlannedEntry(BaseModel):
    """A pair to cache, the prompt form it is cached in and the look and hand changes its images take (None: as
    scanned)."""

    model_config = ConfigDict(frozen=True)

    pair: TrainingPair
    variant: Variant
    restyle: Restyle | None = None
    hand: HandRestyle | None = None


class CachedSample(BaseModel):
    """A cache entry's tensors: the prompt's embeddings ``[1, tokens, text_dim]`` (cached, or encoded as it is
    loaded), the target's latents ``[1, latent_channels, H / 16, W / 16]`` and the reference's (None for a sample drawn
    without it)."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    entry: CacheEntry
    prompt_embeds: Tensor
    target: Tensor
    reference: Tensor | None


def encoder_pipeline(model: Path, device: str, names: tuple[str, ...] = ENCODERS) -> ModularPipeline:
    """The encoder steps of the MOSS model directory ``model`` (its text encoder, and its reference encoder for the
    VAE it names; called without an image, it encodes none), with its components ``names`` (by default its text
    encoder, tokenizer and VAE) loaded as the page pipeline loads them (``COMPONENT_DTYPES``), on ``device``.

    Raises:
        RuntimeError: If a component failed to load.
    """
    blocks = SequentialPipelineBlocks.from_blocks_dict(
        {"text_encoder": MossTextEncoderStep(), "reference_encoder": MossReferenceEncodeStep()}
    )
    pipe = MossModularPipeline(blocks=blocks, pretrained_model_name_or_path=str(model))
    pipe.load_components(names=list(names), dtype=COMPONENT_DTYPES)
    require_loaded(pipe, names)
    pipe.to(device)
    return pipe


class PromptEncoder:
    """The frozen text encoder of the MOSS model directory ``model`` on ``device``, loaded as the cache loads it
    (:func:`encoder_pipeline`), encoding a prompt in inference mode exactly as the cache encodes it: a training's step
    reads its prompts from it (``--encode-text``) instead of from cached embeddings.

    Attributes:
        pipe: The encoder pipeline, its text encoder frozen and in eval mode.
    """

    def __init__(self, model: Path, device: str) -> None:
        self.pipe = encoder_pipeline(model, device, TEXT_ENCODERS)
        self.pipe.text_encoder.requires_grad_(False).eval()

    def encode(self, prompt: str) -> Tensor:
        """The prompt's embeddings ``[1, tokens, text_dim]`` as a cache entry holds them (in the encoder's bfloat16), on
        the encoder's device: a plain tensor, which the DiT's backward may save."""
        with torch.inference_mode():
            embeds = self.pipe(prompt=prompt, output="prompt_embeds")
        return embeds.clone()


def styled(image: Image.Image, *, restyle: Restyle | None = None, hand: HandRestyle | None = None) -> Image.Image:
    """``image`` as an entry's image is encoded: in ``hand``'s hand (:meth:`HandRestyle.apply`), then in
    ``restyle``'s look (:func:`diffu_moss.train.restyle.restyled`); as it is without either."""
    drawn = image if hand is None else hand.apply(image)
    return drawn if restyle is None else restyled(drawn, restyle)


def image_latents(
    pipe: ModularPipeline, path: Path, *, restyle: Restyle | None = None, hand: HandRestyle | None = None
) -> tuple[Tensor, tuple[int, int]]:
    """The image at ``path`` as the DiT's latents, as the reference encode step encodes a reference (changed as
    :func:`styled` changes it), and the size it was encoded at."""
    with Image.open(path) as image:
        drawn = styled(image, restyle=restyle, hand=hand)
        return encode_latents(pipe.vae, image_pixels(drawn)), drawn.size


def cache_entry(
    pipe: ModularPipeline, planned: PlannedEntry, file: str, *, latents_only: bool = False
) -> tuple[CacheEntry, dict[str, Tensor]]:
    """``planned``'s pair in its prompt form, its images in its hand and look, encoded: its entry and its tensors (in
    :data:`CACHE_DTYPE`, on the CPU; with ``latents_only`` no ``prompt_embeds``). A hand change moves the target's
    prompt with its lines (:meth:`HandRestyle.moved_prompt`); one it refuses is left out, its reason recorded.

    Raises:
        ValueError: If the pair's caption cannot be compiled in that form, or an image is off the latent grid.
    """
    pair, variant, hand = planned.pair, planned.variant, planned.hand
    with Image.open(pair.target) as image:
        prompt = pair.prompt(variant, image.size)
        size = image.size
    refused = None
    if hand is not None:
        try:
            prompt, _ = hand.moved_prompt(prompt, size)
        except HandRefused as refusal:
            hand, refused = None, refusal.reason
    target, target_size = image_latents(pipe, pair.target, restyle=planned.restyle, hand=hand)
    tensors = {"target": target}
    if latents_only:
        tokens = prompt_tokens(pipe.tokenizer, prompt)
    else:
        tensors["prompt_embeds"] = pipe(prompt=prompt, output="prompt_embeds")
        tokens = tensors["prompt_embeds"].shape[1]
    reference_size = None
    if variant.takes_reference:
        tensors["reference"], reference_size = image_latents(pipe, pair.reference, restyle=planned.restyle, hand=hand)
    entry = CacheEntry(
        file=file,
        pair=pair.id,
        variant=variant,
        prompt=prompt,
        tokens=tokens,
        target_size=target_size,
        reference_size=reference_size,
        restyle=planned.restyle,
        hand=hand,
        hand_refused=refused,
    )
    return entry, {name: tensor.to("cpu", CACHE_DTYPE).contiguous() for name, tensor in tensors.items()}


class CacheSettings(BaseModel):
    """Which pairs a cache holds, in which mix of prompt forms or in every form of the mix, which shares of the entries
    in the sample form take a look change and a hand change, and where it is encoded."""

    model_config = ConfigDict(frozen=True)

    limit: int | None = None
    seed: int = 0
    mix: PromptMix = PromptMix.NAMED
    every_form: bool = False
    restyle_share: float = Field(default=0.0, ge=0, le=1)
    hand_restyle_share: float = Field(default=0.0, ge=0, le=1)
    latents_only: bool = False
    device: str = "cuda"


def _sample_places(variants: list[Variant]) -> list[int]:
    """The places of the entries in the sample form, given the entries' prompt forms ``variants`` in order."""
    return [index for index, variant in enumerate(variants) if variant is Variant.SAMPLE]


def planned_restyles(variants: list[Variant], share: float, seed: int) -> dict[int, Restyle]:
    """The look changes of a cache's entries, by their place, given their prompt forms ``variants`` in order: of the
    entries in the sample form exactly ``share`` (rounded), picked by a stream seeded by ``seed``, given restyles drawn
    from that stream, a third of each kind (:func:`diffu_moss.train.restyle.draw_restyles`), in place order."""
    places = _sample_places(variants)
    stream = random.Random(f"restyle/{seed}")  # apart from the stream that picks the pairs and their forms
    picked = sorted(stream.sample(places, round(len(places) * share)))
    return dict(zip(picked, draw_restyles(len(picked), stream), strict=True))


def planned_hands(variants: list[Variant], share: float, seed: int) -> dict[int, HandRestyle]:
    """The hand changes of a cache's entries, by their place, as :func:`planned_restyles` plans the look changes but
    from a stream of their own (:func:`diffu_moss.train.restyle.draw_hand_restyles`), so they fall independently of
    the look changes and leave those where they would fall without them."""
    places = _sample_places(variants)
    stream = random.Random(f"hand/{seed}")
    picked = sorted(stream.sample(places, round(len(places) * share)))
    return dict(zip(picked, draw_hand_restyles(len(picked), stream), strict=True))


def planned_entries(pairs: list[TrainingPair], settings: CacheSettings) -> list[PlannedEntry]:
    """The entries to cache, in order: the pairs shuffled by ``settings.seed``, the first ``settings.limit`` of them
    (all when None), each in the prompt form ``settings.mix`` gives it, or in every form of the mix, and the shares
    ``settings.restyle_share`` and ``settings.hand_restyle_share`` of those in the sample form with a look change and
    a hand change (:func:`planned_restyles`, :func:`planned_hands`)."""
    chosen = list(pairs)
    random.Random(settings.seed).shuffle(chosen)
    chosen = chosen[: settings.limit]
    if settings.every_form:
        forms = [(pair, variant) for pair in chosen for variant in MIXES[settings.mix].forms]
    else:
        forms = list(zip(chosen, assign_variants(len(chosen), settings.seed, settings.mix), strict=True))
    variants = [variant for _, variant in forms]
    restyles = planned_restyles(variants, settings.restyle_share, settings.seed)
    hands = planned_hands(variants, settings.hand_restyle_share, settings.seed)
    return [
        PlannedEntry(pair=pair, variant=variant, restyle=restyles.get(index), hand=hands.get(index))
        for index, (pair, variant) in enumerate(forms)
    ]


def check_forms(manifest: Path, planned: list[PlannedEntry]) -> None:
    """Refuse the entries ``planned`` of ``manifest`` if a pair lacks the form it is planned in, before any is encoded.

    Raises:
        ValueError: If a pair lacks the form it is planned in.
    """
    lacking = [plan.pair.id for plan in planned if plan.variant not in plan.pair.forms()]
    if lacking:
        raise ValueError(
            f"{manifest}: {len(lacking)} pairs lack the form they are planned in (the first: {lacking[0]}); give its "
            "rows every form (diffu_page.data.page_pair_forms), or another mix"
        )


def shared_sample_size(manifest: Path, planned: list[PlannedEntry]) -> SampleSize:
    """How the samples of the pairs of ``manifest`` planned were sized: the one rule they share (the first for none).

    Raises:
        ValueError: If the pairs' samples were sized by more than one rule.
    """
    rules = {plan.pair.sample_size for plan in planned}
    if len(rules) > 1:
        raise ValueError(f"{manifest}: its pairs' samples are sized by more than one rule ({', '.join(sorted(rules))})")
    return rules.pop() if rules else SampleSize.HALF


def write_cache(model: Path, manifest: Path, out: Path, settings: CacheSettings) -> CacheRecord:
    """The pairs of ``manifest`` cached into ``out`` by the MOSS model directory ``model``'s encoders (the module's
    docstring), resuming the entries a stopped run wrote.

    Returns:
        The cache's record.

    Raises:
        ValueError: If ``out`` holds a finished cache already, a pair lacks the form it is planned in, the pairs'
            samples are sized by more than one rule, or a pair's caption cannot be compiled.
        RuntimeError: If an encoder failed to load.
    """
    if (out / CACHE_RECORD).exists():
        raise ValueError(f"{out} holds a finished cache already; give a new directory")
    pairs = read_manifest(manifest)
    kept = apart(pairs)
    planned = planned_entries(kept, settings)
    check_forms(manifest, planned)
    sample_size = shared_sample_size(manifest, planned)
    pipe = encoder_pipeline(model, settings.device, LATENT_ENCODERS if settings.latents_only else ENCODERS)
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, plan in enumerate(planned):
        file = f"{index:06d}_{plan.variant}.safetensors"
        if (out / file).exists():
            with safe_open(str(out / file), framework="pt") as written:
                entries.append(CacheEntry.model_validate_json(written.metadata()["entry"]))
            continue
        entry, tensors = cache_entry(pipe, plan, file, latents_only=settings.latents_only)
        partial = out / f"{file}.partial"
        save_file(tensors, str(partial), metadata={"entry": entry.model_dump_json()})
        partial.rename(out / file)
        entries.append(entry)
        if (index + 1) % 100 == 0:
            console.print(f"{index + 1} of {len(planned)} entries cached")
    (out / ENTRIES).write_text("".join(f"{entry.model_dump_json()}\n" for entry in entries), encoding="utf-8")
    forms = MIXES[settings.mix].forms
    counts = {variant.value: sum(entry.variant is variant for entry in entries) for variant in forms}
    restyled_kinds = Counter(entry.restyle.kind for entry in entries if entry.restyle is not None)
    refused = Counter(entry.hand_refused for entry in entries if entry.hand_refused is not None)
    record = CacheRecord(
        pairs=str(manifest.resolve()),
        model=str(model.resolve()),
        seed=settings.seed,
        pairs_read=len(pairs),
        pairs_sharing_lines=len(pairs) - len(kept),
        entries=counts,
        latents_only=settings.latents_only,
        restyle_share=settings.restyle_share,
        restyled=dict(sorted(restyled_kinds.items())),
        mix=settings.mix,
        sample_size=sample_size,
        hand_restyle_share=settings.hand_restyle_share,
        hand_restyled=sum(entry.hand is not None for entry in entries),
        hand_refused=dict(sorted(refused.items())),
    )
    record.write(out / CACHE_RECORD)
    return record


def read_entries(cache: Path) -> list[CacheEntry]:
    """The entries of the finished cache at ``cache``.

    Raises:
        ValueError: If ``cache`` holds no finished cache.
    """
    if not (cache / CACHE_RECORD).is_file():
        raise ValueError(f"{cache} holds no finished cache (no {CACHE_RECORD}): run diffu-moss-cache-pairs")
    with (cache / ENTRIES).open(encoding="utf-8") as rows:
        return [CacheEntry.model_validate_json(row) for row in rows if row.strip()]


def with_reference(entries: list[CacheEntry], count: int) -> list[CacheEntry]:
    """The first ``count`` of ``entries`` drawn with their reference, in the cache's order: the fixed handful of pairs
    an overfit test trains on (``diffu-moss-train-ref --limit-pairs``) and draws back."""
    return [entry for entry in entries if entry.variant.takes_reference][:count]


def read_record(cache: Path) -> CacheRecord:
    """The record of the finished cache at ``cache``.

    Raises:
        ValueError: If ``cache`` holds no finished cache.
    """
    if not (cache / CACHE_RECORD).is_file():
        raise ValueError(f"{cache} holds no finished cache (no {CACHE_RECORD}): run diffu-moss-cache-pairs")
    return CacheRecord.read(cache / CACHE_RECORD)


def target_shape(cache: Path, entry: CacheEntry) -> list[int]:
    """The shape of an entry's target latents, read from its file's header."""
    with safe_open(str(cache / entry.file), framework="pt") as tensors:
        return list(tensors.get_slice("target").get_shape())


def load_sample(
    cache: Path, entry: CacheEntry, device: torch.device | str, encoder: PromptEncoder | None = None
) -> CachedSample:
    """An entry's tensors, on ``device``: its prompt's embeddings encoded by ``encoder`` when given, else cached.

    Raises:
        ValueError: If no encoder is given and the entry holds no prompt embeddings (a latents-only cache).
    """
    tensors = load_file(str(cache / entry.file), device=str(device))
    if encoder is not None:
        tensors["prompt_embeds"] = encoder.encode(entry.prompt).to(device)
    elif "prompt_embeds" not in tensors:
        raise ValueError(f"{cache} holds no prompt embeddings (a latents-only cache): encode them (--encode-text)")
    return CachedSample(
        entry=entry,
        prompt_embeds=tensors["prompt_embeds"],
        target=tensors["target"],
        reference=tensors.get("reference"),
    )


@app.command()
def main(
    model: Annotated[
        Path, typer.Option(help="a MOSS model directory (diffu-moss-init): its text encoder, tokenizer, VAE")
    ],
    pairs: Annotated[
        Path,
        typer.Option(help="a style-pair manifest (style_pairs' train.jsonl or val/val.jsonl) or a page_pairs one"),
    ],
    out: Annotated[Path, typer.Option(help="the cache's directory (a new one, or a stopped run's to resume)")],
    limit: Annotated[
        int | None, typer.Option(min=1, help="cache this many pairs, picked by --seed (default: all)")
    ] = None,
    seed: Annotated[int, typer.Option(help="picks the pairs and their prompt forms")] = 0,
    mix: Annotated[
        PromptMix,
        typer.Option(
            help="the mix of prompt forms: named (three of four forms name a page's archive and years) or content "
            "(no form names an archive, years or hand: the sample form, the sample with the look in words, and no "
            "sample with nothing named)"
        ),
    ] = PromptMix.NAMED,
    every_form: Annotated[
        bool, typer.Option("--every-form", help="cache every pair in every prompt form of the mix (a validation set)")
    ] = False,
    restyle_share: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="give this share of the entries in the sample form, picked by --seed, one look change for the page "
            "and its sample alike: microfilm, colourised or tinted (diffu_moss.train.restyle); 0: as scanned",
        ),
    ] = 0.0,
    hand_restyle_share: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="give this share of the entries in the sample form, picked by --seed apart from the look changes, one "
            "hand change for the page and its sample alike: strokes, slant and width (diffu_moss.train.restyle), the "
            "page's prompt moved with its lines; 0: as scanned",
        ),
    ] = 0.0,
    latents_only: Annotated[
        bool,
        typer.Option(
            "--latents-only",
            help="cache the latents alone, no prompt embeddings (a training then encodes them: --encode-text)",
        ),
    ] = False,
    device: Annotated[str, typer.Option(help="where the encoders run")] = "cuda",
) -> None:
    """Cache style pairs for a reference training: each pair's prompt embeddings and latents."""
    settings = CacheSettings(
        limit=limit,
        seed=seed,
        mix=mix,
        every_form=every_form,
        restyle_share=restyle_share,
        hand_restyle_share=hand_restyle_share,
        latents_only=latents_only,
        device=device,
    )
    try:
        record = write_cache(model, pairs, out, settings)
    except ValueError as error:  # the arguments' fault: a finished cache, a manifest without the mix's forms, ...
        raise typer.BadParameter(str(error)) from error
    left_out = f"{record.pairs_sharing_lines} of {record.pairs_read} pairs left out for sharing a line"
    console.print(f"{json.dumps(record.entries)} entries ({left_out}) -> {out}")
    if record.hand_restyle_share:
        console.print(f"{record.hand_restyled} hand changes, refused: {json.dumps(record.hand_refused)}")


if __name__ == "__main__":
    app()
