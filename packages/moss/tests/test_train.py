"""MOSS-Image's reference training on CPU (conftest's tiny MOSS and tiny style pairs): the caption forms, the mix of
prompt forms in the cache (the pairs that share a line with their reference left out) and over a run, MOSS's noise
levels, the flow-matching loss over the drawn image alone, a LoRA that learns at the recipe's learning rate, the
validation's paired noise, a training that lowers the loss of what it trains on, and one that refuses an empty
cache."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from peft import get_peft_model_state_dict
from safetensors.torch import load_file

from diffu_moss.sample_caption import with_sample, without_sample
from diffu_moss.train.cache import CachedSample, CacheEntry, CacheSettings, load_sample, read_entries, write_cache
from diffu_moss.train.export import export_merged
from diffu_moss.train.loop import (
    LOG,
    TrainSettings,
    flow_matching_loss,
    lora_transformer,
    train,
    validation_losses,
)
from diffu_moss.train.pairs import Variant, read_pairs
from diffu_moss.train.records import ADAPTER_WEIGHTS
from diffu_moss.train.sigmas import sigma_grid, sigma_weights
from diffu_moss.transformer import MossTransformer

INK_HEX = "#5A4632"  # the described caption's ink, one of its look words


@pytest.mark.parametrize(
    ("form", "paragraph", "line"),
    [
        pytest.param(
            with_sample,
            "Create a flat, evenly lit archival scan of a new handwritten page in the hand, ink and paper of the sample "
            "image: right-slanted cursive",
            'Handwritten line of text reading "Anno 1700 den 1 Maj", right-slanted cursive in the sample image\'s '
            "hand, in brown ink.",
            id="with-sample",
        ),
        pytest.param(
            without_sample,
            "Create a flat, evenly lit archival scan of a new handwritten page: right-slanted cursive",
            'Handwritten line of text reading "Anno 1700 den 1 Maj", right-slanted cursive, in brown ink.',
            id="without-sample",
        ),
    ],
)
def test_sample_caption_names_the_sample_image_or_drops_the_citation_keeping_every_other_word(
    tiny_pairs: Path, form: Callable[[str], str], paragraph: str, line: str
) -> None:
    described = read_pairs(tiny_pairs / "train.jsonl")[0].caption

    caption = form(described)

    rows = json.loads(caption[caption.find(" [{") + 1 :])
    assert caption.startswith(paragraph)
    assert rows[1]["desc"] == line
    assert rows[0]["from"] == "ref_image_0"  # the layout rows' fields are left as they are


def test_moss_training_noise_levels_have_the_quartiles_of_mosss_own_sampler() -> None:
    cumulative = sigma_weights().cumsum(0)

    quartiles = [
        sigma_grid()[torch.searchsorted(cumulative, torch.tensor(q, dtype=torch.float64))] for q in (0.25, 0.5, 0.75)
    ]

    assert [round(level.item(), 3) for level in quartiles] == [0.635, 0.781, 0.886]  # its median documented as 0.78


def test_flow_matching_loss_is_the_drawn_images_velocity_error_with_the_reference_clean(tiny_moss_dit: Path) -> None:
    model = MossTransformer.from_pretrained(str(tiny_moss_dit)).eval()
    generator = torch.Generator().manual_seed(0)
    target, noise = torch.randn(1, 16, 2, 4, generator=generator), torch.randn(1, 16, 2, 4, generator=generator)
    reference, text = torch.randn(1, 16, 2, 2, generator=generator), torch.randn(1, 5, 16, generator=generator)
    entry = CacheEntry(
        file="x", pair="x", variant=Variant.LOOK, prompt="", tokens=5, target_size=(64, 32), reference_size=(32, 32)
    )
    sample = CachedSample(entry=entry, prompt_embeds=text, target=target, reference=reference)
    mask = torch.ones(1, 5, dtype=torch.bool)
    with torch.no_grad():
        velocity = model(0.25 * target + 0.75 * noise, torch.tensor([750.0]), text, mask, references=[reference])

        loss = flow_matching_loss(model, sample, 0.75, noise)

    assert loss.item() == pytest.approx(((velocity - (noise - target)) ** 2).mean().item(), rel=1e-6)


def test_cache_holds_a_fifth_of_the_pairs_without_the_reference_and_half_the_rest_without_look_words(
    tiny_pair_cache: Path,
) -> None:
    entries = read_entries(tiny_pair_cache)

    counts = {variant: sum(entry.variant is variant for entry in entries) for variant in Variant}
    forms = {
        (
            entry.variant,
            load_sample(tiny_pair_cache, entry, "cpu").reference is not None,
            "the sample image" in entry.prompt,
            INK_HEX in entry.prompt,
        )
        for entry in entries
    }
    # 10 of the 11 pairs: vol_03's reference (vol_04's crop) holds its first line, so it is left out
    assert counts == {Variant.NO_REFERENCE: 2, Variant.LOOK: 4, Variant.PLAIN: 4}
    assert forms == {
        (Variant.NO_REFERENCE, False, False, True),  # no image, no citation, the look in words
        (Variant.LOOK, True, True, True),
        (Variant.PLAIN, True, True, False),  # the look from the image alone
    }


def test_training_draws_every_entry_once_a_pass_so_a_run_keeps_the_caches_mix(
    tiny_moss_dit: Path, tiny_pair_cache: Path, tmp_path: Path
) -> None:
    train(tiny_moss_dit, tiny_pair_cache, tmp_path, TrainSettings(steps=5, save_every=5, device="cpu"))

    steps = [json.loads(line) for line in (tmp_path / LOG).read_text(encoding="utf-8").splitlines()]
    drawn = [variant for step in steps for variant in step["variants"]]
    # 5 steps of 2: one pass over the 10 entries, each once
    assert sorted(pair for step in steps for pair in step["pairs"]) == sorted(
        entry.pair for entry in read_entries(tiny_pair_cache)
    )
    assert {variant: drawn.count(variant) for variant in set(drawn)} == {"no_reference": 2, "look": 4, "plain": 4}


def test_training_limited_to_n_pairs_trains_on_the_caches_first_n_drawn_with_their_reference(
    tiny_moss_dit: Path, tiny_pair_cache: Path, tmp_path: Path
) -> None:
    train(tiny_moss_dit, tiny_pair_cache, tmp_path, TrainSettings(steps=3, limit_pairs=3, save_every=3, device="cpu"))

    steps = [json.loads(line) for line in (tmp_path / LOG).read_text(encoding="utf-8").splitlines()]
    # 3 steps of 2: two passes over vol_08, vol_09 (look) and vol_06 (plain); vol_01, cached before vol_06, takes none
    assert sorted(pair for step in steps for pair in step["pairs"]) == sorted(["vol_08", "vol_09", "vol_06"] * 2)


def test_training_moves_every_lora_a_weight_at_the_recipes_learning_rate(
    tiny_moss_dit: Path, tiny_reference_run: Path
) -> None:
    initial = get_peft_model_state_dict(lora_transformer(tiny_moss_dit, TrainSettings().lora, device="cpu", seed=0))

    trained = load_file(str(tiny_reference_run / "checkpoints" / "step_000005" / ADAPTER_WEIGHTS))

    moved = [(trained[name] != initial[name]).float().mean().item() for name in initial if ".lora_A." in name]
    assert min(moved) > 0.99  # warmup steps of 2e-6 to 1e-5: below bfloat16's resolution on weights of about 0.1


def test_validation_scores_a_pairs_three_forms_at_the_same_noise_level_and_noise(
    tiny_moss: Path, tiny_pairs: Path, tmp_path: Path
) -> None:
    write_cache(tiny_moss, tiny_pairs / "train.jsonl", tmp_path, CacheSettings(limit=2, every_form=True, device="cpu"))
    blind = MossTransformer(num_layers=0, dim=32, ffn_dim=64, num_heads=2, text_dim=16, freq_dim=16, latent_channels=16)

    losses = validation_losses(blind.eval(), tmp_path, read_entries(tmp_path), device="cpu", seed=0)

    # a DiT without blocks reads neither the prompt nor the reference: a pair's three forms differ in nothing else
    assert sorted(losses) == ["look", "no_reference", "plain"]
    assert losses["look"] == losses["plain"] == losses["no_reference"]


def test_training_lowers_the_loss_of_the_sample_it_trains_on(
    tiny_moss: Path, tiny_moss_dit: Path, tiny_pairs: Path, tmp_path: Path
) -> None:
    write_cache(tiny_moss, tiny_pairs / "train.jsonl", tmp_path / "one", CacheSettings(limit=1, device="cpu"))
    sample = load_sample(tmp_path / "one", read_entries(tmp_path / "one")[0], "cpu")
    noise = torch.randn(sample.target.shape, generator=torch.Generator().manual_seed(7))
    settings = TrainSettings(steps=6, learning_rate=1e-2, warmup=1, save_every=6, device="cpu")

    checkpoint = train(tiny_moss_dit, tmp_path / "one", tmp_path / "run", settings)

    export_merged(tiny_moss_dit, checkpoint, tmp_path / "export")
    losses = []
    for folder in (tiny_moss_dit, tmp_path / "export"):
        model = MossTransformer.from_pretrained(str(folder), torch_dtype=torch.bfloat16).eval()
        with torch.no_grad():
            losses.append(flow_matching_loss(model, sample, 0.6, noise).item())
    assert losses[1] < 0.9 * losses[0]


def test_training_refuses_a_cache_without_entries_before_it_loads_the_dit(tiny_moss: Path, tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text("", encoding="utf-8")
    write_cache(tiny_moss, tmp_path / "train.jsonl", tmp_path / "empty", CacheSettings(device="cpu"))
    (tmp_path / "no_dit").mkdir()  # a folder without a DiT: loading one from it would fail otherwise

    with pytest.raises(ValueError, match="holds no entry to train on"):
        train(tmp_path / "no_dit", tmp_path / "empty", tmp_path / "run", TrainSettings(steps=1, device="cpu"))
