"""A cache of region crops (diffu_moss.train.region_crops) on conftest's tiny pairs: each entry a manifest names keeps
its first cache's target latents and prompt embeddings and gains its crops encoded by the VAE, whole or in shards; a
training reads the crops at their offsets in place of the sample, records their place in its checkpoints' config, and
refuses a cache that gives its references otherwise than it is told."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from diffu_moss.latents import encode_latents, image_pixels
from diffu_moss.train.cache import CACHE_DTYPE, ENTRIES, encoder_pipeline, load_sample, read_entries, read_record
from diffu_moss.train.full import FullSettings
from diffu_moss.train.loop import TrainSettings, flow_matching_loss, train
from diffu_moss.train.records import CACHE_RECORD
from diffu_moss.train.region_crops import (
    CROP_IMAGES,
    REGION_CROP_BUILD,
    REGION_CROPS,
    RegionCropBuild,
    RegionCropCacheSettings,
    RegionCropFile,
    RegionCropRow,
    read_region_crops,
    write_region_crop_cache,
)
from diffu_moss.transformer import MossTransformer

CROPS = (((32, 16), (-0.5, 1.0)), ((16, 32), (0.0, -1.5)))  # each entry's two crops: (width, height) px, offset


def _noise(size: tuple[int, int], seed: int) -> Image.Image:
    pixels = torch.rand(size[1], size[0], 3, generator=torch.Generator().manual_seed(seed))
    return Image.fromarray((pixels * 255).to(torch.uint8).numpy())


@pytest.fixture(scope="module")
def crop_manifest(tiny_pair_cache: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A manifest of region crops for the tiny pairs' cache, as ``diffu_page.data.region_crop_pairs`` writes one: each
    entry drawn with its sample given two noise crops (:data:`CROPS`), the rest left out."""
    out = tmp_path_factory.mktemp("crop_manifest")
    (out / CROP_IMAGES).mkdir()
    entries = read_entries(tiny_pair_cache)
    rows = []
    for k, entry in enumerate(entry for entry in entries if entry.variant.takes_reference):
        crops = []
        for index, (size, offset) in enumerate(CROPS):
            name = f"{CROP_IMAGES}/{k}_{index}.png"
            _noise(size, 10 * k + index).save(out / name)
            crops.append(RegionCropFile(file=name, source=f"s{index}", target=f"t{index}", size=size, offset=offset))
        rows.append(RegionCropRow(entry=entry.file, pair=entry.pair, crops=crops))
    (out / REGION_CROPS).write_text("".join(f"{row.model_dump_json()}\n" for row in rows), encoding="utf-8")
    RegionCropBuild(
        cache=str(tiny_pair_cache),
        pairs=read_record(tiny_pair_cache).pairs,
        place="over",
        max_width=512,
        max_height=512,
        seed=0,
        entries_read=len(entries),
        served=len(rows),
        skipped={"no sample": len(entries) - len(rows)},
    ).write(out / REGION_CROP_BUILD)
    return out / REGION_CROPS


@pytest.fixture(scope="module")
def crop_cache(tiny_pair_cache: Path, crop_manifest: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The tiny pairs' cache with its region crops, in one run."""
    out = tmp_path_factory.mktemp("crop_cache") / "cache"
    write_region_crop_cache(tiny_pair_cache, crop_manifest, out, RegionCropCacheSettings(device="cpu"))
    return out


def test_region_crop_cache_holds_each_entry_s_crops_encoded_beside_its_first_cache_s_target_and_prompt(
    tiny_moss: Path, tiny_pair_cache: Path, crop_manifest: Path, crop_cache: Path
) -> None:
    vae = encoder_pipeline(tiny_moss, "cpu", ("vae",)).vae
    rows = read_region_crops(crop_manifest)
    firsts = {entry.file: entry for entry in read_entries(tiny_pair_cache)}

    entries = read_entries(crop_cache)

    assert [entry.file for entry in entries] == [row.entry for row in rows]
    assert read_record(crop_cache).region_crops is not None
    for entry, row in zip(entries, rows, strict=True):
        cached, first = load_sample(crop_cache, entry, "cpu"), load_sample(tiny_pair_cache, firsts[entry.file], "cpu")
        assert cached.reference is None and entry.reference_size is None
        assert torch.equal(cached.target, first.target) and torch.equal(cached.prompt_embeds, first.prompt_embeds)
        assert [(crop.size, crop.offset) for crop in entry.region_crops or ()] == list(CROPS)
        for latents, crop in zip(cached.region_crops, row.crops, strict=True):
            with Image.open(crop_manifest.parent / crop.file) as image:
                expected = encode_latents(vae, image_pixels(image)).to(CACHE_DTYPE)
            assert torch.equal(latents, expected)


def test_region_crop_cache_in_shards_then_finished_is_the_cache_one_run_writes(
    tiny_pair_cache: Path, crop_manifest: Path, crop_cache: Path, tmp_path: Path
) -> None:
    out = tmp_path / "cache"
    for shard in range(2):
        write_region_crop_cache(
            tiny_pair_cache, crop_manifest, out, RegionCropCacheSettings(device="cpu", shard=shard, shards=2)
        )
    unfinished = sorted(path.name for path in out.iterdir())

    record = write_region_crop_cache(tiny_pair_cache, crop_manifest, out, RegionCropCacheSettings(device="cpu"))

    assert ENTRIES not in unfinished and CACHE_RECORD not in unfinished
    assert (out / ENTRIES).read_text() == (crop_cache / ENTRIES).read_text()
    assert record == read_record(crop_cache)
    for entry in read_entries(out):
        ours, one_run = load_sample(out, entry, "cpu"), load_sample(crop_cache, entry, "cpu")
        assert all(torch.equal(a, b) for a, b in zip(ours.region_crops, one_run.region_crops, strict=True))


def test_flow_matching_loss_reads_an_entry_s_region_crops_each_at_its_offset(
    tiny_moss_dit: Path, crop_cache: Path
) -> None:
    dit = MossTransformer.from_pretrained(str(tiny_moss_dit), torch_dtype=torch.float32).eval()
    sample = load_sample(crop_cache, read_entries(crop_cache)[0], "cpu")
    entry = sample.entry
    moved = [crop.model_copy(update={"offset": (0.0, 0.0)}) for crop in entry.region_crops or ()]
    noise = torch.randn(sample.target.shape, generator=torch.Generator().manual_seed(0))

    with torch.no_grad():
        losses = [
            flow_matching_loss(dit, drawn, 0.5, noise).item()
            for drawn in (
                sample,
                sample.model_copy(update={"region_crops": []}),
                sample.model_copy(update={"entry": entry.model_copy(update={"region_crops": moved})}),
            )
        ]

    assert losses[0] != losses[1]  # the crops are read
    assert losses[0] != losses[2]  # at their offsets


def test_a_full_fine_tune_on_region_crops_records_their_place_in_its_checkpoints_config(
    tiny_moss_dit: Path, crop_cache: Path, tmp_path: Path
) -> None:
    settings = TrainSettings(steps=1, full=FullSettings(), region_crops="over", device="cpu")

    checkpoint = train(tiny_moss_dit, crop_cache, tmp_path / "run", settings)

    assert MossTransformer.from_pretrained(str(checkpoint / "transformer")).config.region_crops == "over"


@pytest.mark.parametrize(
    ("given", "refusal"),
    [
        pytest.param(None, "gives region crops: --region-crops over", id="crops-without-the-option"),
        pytest.param("beside", "gives region crops: --region-crops over", id="crops-at-another-place"),
    ],
)
def test_training_refuses_a_region_crop_cache_given_otherwise(
    crop_cache: Path, tmp_path: Path, given: str | None, refusal: str
) -> None:
    (tmp_path / "no_dit").mkdir()  # refused before the DiT is loaded
    settings = TrainSettings.model_validate(
        {"steps": 1, "full": FullSettings(), "region_crops": given, "device": "cpu"}
    )

    with pytest.raises(ValueError, match=refusal):
        train(tmp_path / "no_dit", crop_cache, tmp_path / "run", settings)


def test_training_refuses_region_crops_on_a_cache_of_whole_samples(tiny_pair_cache: Path, tmp_path: Path) -> None:
    (tmp_path / "no_dit").mkdir()
    settings = TrainSettings(steps=1, full=FullSettings(), region_crops="over", device="cpu")

    with pytest.raises(ValueError, match="gives whole samples: train it without --region-crops"):
        train(tmp_path / "no_dit", tiny_pair_cache, tmp_path / "run", settings)
