"""diffu's training cache on CPU with the tiny diffu repo: each crop written where NeMo's TextToImageDataset reads it,
holding exactly the latents and prompt embeddings diffu's own pipeline steps make at generation (a glyph map as the
reference encoder and a glyph model's glyph encoder make it), and the empty prompt caption dropout trains on; its
latents' stats measured, and its latents handed to a repo with other stats."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from diffusers import Flux3Transformer2DModel
from diffusers.modular_pipelines import SequentialPipelineBlocks
from PIL import Image
from typer.testing import CliRunner

from diffu.configs import DiffuConfig
from diffu.init import read_latent_stats, write_model_repo, write_vae
from diffu.latents import LatentStats, vae_stats
from diffu.modular_pipelines.diffu import DiffuModularPipeline
from diffu.modular_pipelines.diffu.encoders import DiffuGlyphEncoderStep, DiffuTextEncoderStep
from diffu.modular_pipelines.diffu.modular_blocks_diffu import DiffuVaeEncoderSequentialStep
from diffu.modular_pipelines.loading import COMPONENT_DTYPES
from diffu_page.captions.prompt import LineRecord
from diffu_page.captions.records import CropRecord
from diffu_page.glyphs import glyph_map
from diffu_train.cache import DiffuEncoders, cache_crops, cached_files, write_metadata
from diffu_train.cache_glyphs import cache_glyph_pairs, write_pair_metadata
from diffu_train.latent_stats import RenormalizedLatents
from diffu_train.latent_stats import app as latent_stats_app

CAPTIONS = ["Ett protokoll i brun bläck.", "En dombok på gult papper, två rader."]
PAGES = LatentStats(latents_mean=(0.1, -0.2, 0.3, 0.05), latents_std=(0.5, 0.25, 2.0, 0.75))  # the tiny VAE's 4


def _split(directory: Path) -> list[CropRecord]:
    """Two 64 x 32 crops with their records, as diffu_page.data.build_crops writes a split."""
    directory.mkdir()
    records: list[CropRecord] = []
    for k, caption in enumerate(CAPTIONS):
        Image.linear_gradient("L").resize((64, 32)).rotate(90 * k).convert("RGB").save(directory / f"{k}.jpg")
        line = LineRecord(id="Sv_Text_1", text="Göteborgs poliskammare", role="", bbox=[0, 0, 1000, 1000])
        records.append(
            CropRecord(
                file_name=f"{k}.jpg",
                caption=caption,
                collection="GT-RA_test",
                page="page.xml",
                width=64,
                height=32,
                greyscale=False,
                paper="cream",
                ink="brown",
                lines=[line],
            )
        )
    return records


def _repo_with_stats(tiny_diffu: Path, config: DiffuConfig, stats: LatentStats, out: Path) -> Path:
    """The tiny diffu with its own VAE normalizing with ``stats``, written as diffu-init --latent-stats writes it."""
    transformer = Flux3Transformer2DModel.from_pretrained(tiny_diffu, subfolder="transformer")
    write_model_repo(transformer, config.model_copy(update={"vae": write_vae(config.vae, stats, out)}), out)
    return out


def _glyph_cache(records: list[CropRecord], crops: Path, model: Path, out: Path) -> Path:
    run = cache_glyph_pairs(records, crops, encoders=DiffuEncoders.load(model, "cpu"), out=out, scale=0.5)
    write_pair_metadata(run.entries, out, model=model, scale=0.5)
    return out


def _pipeline(blocks: SequentialPipelineBlocks, model: Path) -> DiffuModularPipeline:
    pipe = DiffuModularPipeline(blocks=blocks, pretrained_model_name_or_path=str(model))
    pipe.load_components(dtype=COMPONENT_DTYPES)
    return pipe


def test_cache_holds_each_crop_as_diffus_pipeline_encodes_it_where_nemo_reads_it(
    tiny_diffu: Path, tmp_path: Path
) -> None:
    records = _split(tmp_path / "crops")
    out = tmp_path / "cache"

    run = cache_crops(
        records, tmp_path / "crops", encoders=DiffuEncoders.load(tiny_diffu, "cpu"), out=out, bucket_id=lambda w, h: 7
    )
    write_metadata(run.entries, out, model=tiny_diffu)

    metadata = json.loads((out / "metadata.json").read_text())
    rows = [row for shard in metadata["shards"] for row in json.loads((out / shard).read_text())]
    samples = [torch.load(row["cache_file"], weights_only=True) for row in rows]
    text = _pipeline(SequentialPipelineBlocks.from_blocks_dict({"text_encoder": DiffuTextEncoderStep()}), tiny_diffu)
    vae = _pipeline(DiffuVaeEncoderSequentialStep(), tiny_diffu)
    assert [(row["prompt"], row["bucket_resolution"], row["bucket_id"]) for row in rows] == [
        (caption, [64, 32], 7) for caption in CAPTIONS
    ]
    for k, (caption, sample) in enumerate(zip(CAPTIONS, samples, strict=True)):
        with Image.open(tmp_path / "crops" / f"{k}.jpg") as image:
            latents = vae(image=image.convert("RGB"), output="image_cond_latents")[0][0]
        assert torch.equal(sample["prompt_embeds"], text(prompt=caption, output="prompt_embeds")[0])
        assert torch.equal(sample["latent"], latents.float())


def test_glyph_cache_pairs_each_scaled_crop_with_its_glyph_map_as_diffus_pipeline_encodes_them(
    tiny_diffu: Path, tmp_path: Path
) -> None:
    records = _split(tmp_path / "crops")
    out = tmp_path / "cache"

    run = cache_glyph_pairs(
        records, tmp_path / "crops", encoders=DiffuEncoders.load(tiny_diffu, "cpu"), out=out, scale=0.5
    )
    write_pair_metadata(run.entries, out, model=tiny_diffu, scale=0.5)

    rows = [
        row
        for shard in json.loads((out / "metadata.json").read_text())["shards"]
        for row in json.loads((out / shard).read_text())
    ]
    vae = _pipeline(DiffuVaeEncoderSequentialStep(), tiny_diffu)
    glyph_encoder = _pipeline(DiffuGlyphEncoderStep(), tiny_diffu)  # a glyph model's (E2c); the reference's above (E2b)
    for k, (row, record) in enumerate(zip(rows, records, strict=True)):
        sample = torch.load(out / row["cache_file"], weights_only=True)
        with Image.open(tmp_path / "crops" / f"{k}.jpg") as image:
            crop = image.convert("RGB").resize((32, 16), Image.Resampling.LANCZOS)
        glyphs = glyph_map(record.lines, (32, 16))
        assert torch.equal(sample["target_latent"], vae(image=crop, output="image_cond_latents")[0][0].float())
        assert torch.equal(sample["context_latents"][0], vae(image=glyphs, output="image_cond_latents")[0][0].float())
        glyph_latents = glyph_encoder(glyph_image=glyphs, output="glyph_latents")[0].float()
        assert torch.equal(sample["context_latents"][0], glyph_latents)
        signature = {
            "target": list(sample["target_latent"].shape),
            "contexts": [list(sample["context_latents"][0].shape)],
        }
        assert row["compound_bucket_signature"] == signature == sample["metadata"]["compound_bucket_signature"]


def test_cache_records_a_crop_it_cannot_read_and_caches_the_rest(tiny_diffu: Path, tmp_path: Path) -> None:
    records = _split(tmp_path / "crops")
    (tmp_path / "crops" / "1.jpg").unlink()

    run = cache_crops(
        records,
        tmp_path / "crops",
        encoders=DiffuEncoders.load(tiny_diffu, "cpu"),
        out=tmp_path / "cache",
        bucket_id=lambda w, h: 7,
    )

    assert [entry.prompt for entry in run.entries] == [CAPTIONS[0]]
    assert list(run.failed) == ["1.jpg"]


def test_encoders_refuse_a_repo_whose_vae_does_not_load(tiny_diffu: Path, tmp_path: Path) -> None:
    repo = shutil.copytree(tiny_diffu, tmp_path / "repo")
    (repo / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"not a safetensors file")

    with pytest.raises(RuntimeError, match=r"failed to load .*\['vae'\]"):
        DiffuEncoders.load(repo, "cpu")


def test_write_metadata_refuses_to_mark_an_empty_cache_finished(tiny_diffu: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nothing was cached"):
        write_metadata([], tmp_path, model=tiny_diffu)

    assert not (tmp_path / "metadata.json").exists()


def test_a_cache_renormalized_for_a_repo_holds_what_that_repo_caches(
    tiny_diffu: Path, tiny_diffu_config: DiffuConfig, tmp_path: Path
) -> None:
    records = _split(tmp_path / "crops")
    pages = _repo_with_stats(tiny_diffu, tiny_diffu_config, PAGES, tmp_path / "pages")
    cached = [
        torch.load(path, weights_only=True)
        for path in cached_files(_glyph_cache(records, tmp_path / "crops", tiny_diffu, tmp_path / "cache"))
    ]
    expected = [
        torch.load(path, weights_only=True)
        for path in cached_files(_glyph_cache(records, tmp_path / "crops", pages, tmp_path / "pages_cache"))
    ]

    renormalized = RenormalizedLatents(cached, cached=vae_stats(tiny_diffu), wanted=vae_stats(pages))

    for k, sample in enumerate(expected):
        torch.testing.assert_close(renormalized[k]["target_latent"], sample["target_latent"])
        torch.testing.assert_close(renormalized[k]["context_latents"][0], sample["context_latents"][0])


def test_latent_stats_of_a_cache_are_the_mean_and_std_of_its_crops_raw_latents(
    tiny_diffu: Path, tiny_diffu_config: DiffuConfig, tmp_path: Path
) -> None:
    records = _split(tmp_path / "crops")
    unit = LatentStats(latents_mean=(0.0,) * 4, latents_std=(1.0,) * 4)  # a VAE that caches its raw latents
    raw = _repo_with_stats(tiny_diffu, tiny_diffu_config, unit, tmp_path / "raw")
    raw_latents = torch.cat(
        [
            torch.load(path, weights_only=True)["target_latent"]
            .double()
            .reshape(4, -1)  # VAE channel c: packed 4c-4c+3
            for path in cached_files(_glyph_cache(records, tmp_path / "crops", raw, tmp_path / "raw_cache"))
        ],
        dim=1,
    )
    cache = _glyph_cache(records, tmp_path / "crops", tiny_diffu, tmp_path / "cache")

    result = CliRunner().invoke(latent_stats_app, ["--cache", str(cache), "--out", str(tmp_path / "stats.json")])

    assert result.exit_code == 0, result.output
    stats = read_latent_stats(tmp_path / "stats.json", channels=4)
    torch.testing.assert_close(
        torch.tensor(stats.latents_mean, dtype=torch.float64), raw_latents.mean(dim=1), rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        torch.tensor(stats.latents_std, dtype=torch.float64),
        raw_latents.std(dim=1, correction=0),
        rtol=1e-5,
        atol=1e-6,
    )
