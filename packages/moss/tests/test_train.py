"""MOSS-Image's reference training on CPU (conftest's tiny MOSS and tiny style pairs): the caption forms, the mix of
prompt forms in the cache (the pairs that share a line with their reference left out, page pairs too; a mix whose
forms the pairs lack refused) and over a run, the look changes of the cache's sample form (exactly the share asked,
the target and its sample alike, by transforms that move no pixel; a cache without them written as before), its hand
changes (each box moved exactly as the shear and the width move it, each slant drawn within the largest slant the
page's lines leave room for, every line's ink inside its moved box, the target and its sample changed alike, a change
that would move a line off its page or onto another refused), a page pair that
refuses a prompt its row was not compiled for (another size, a form built after its set), how a cache's samples were
sized recorded from its pairs to the model index (and a validation cache sized otherwise refused), MOSS's noise levels,
the flow-matching loss over the drawn image alone, a LoRA that learns at the recipe's learning rate, the validation's
paired noise, a training that lowers the loss of what it trains on, one that refuses an empty cache, prompts
encoded in the step as the cache encodes them, by a frozen encoder, from a cache of latents alone, and a sample order
that groups a round's samples by size (every entry still once a pass, the rounds counted on across passes)."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import torch
from peft import get_peft_model_state_dict
from PIL import Image, ImageDraw
from safetensors import safe_open
from safetensors.torch import load_file

from diffu_core.repo import ModelIndex
from diffu_moss.init import write_moss_repo
from diffu_moss.latents import encode_latents, image_pixels
from diffu_moss.sample_caption import with_sample, without_sample
from diffu_moss.train.cache import (
    CACHE_DTYPE,
    ENTRIES,
    CachedSample,
    CacheEntry,
    CacheSettings,
    PromptEncoder,
    encoder_pipeline,
    load_sample,
    planned_entries,
    read_entries,
    read_record,
    styled,
    write_cache,
)
from diffu_moss.train.export import export_merged
from diffu_moss.train.full import FullSettings
from diffu_moss.train.loop import (
    LOG,
    SampleOrder,
    SizeGroups,
    TrainSettings,
    entry_tokens,
    flow_matching_loss,
    lora_transformer,
    train,
    validation_losses,
)
from diffu_moss.train.pairs import (
    MIXES,
    PagePair,
    PagePairRow,
    Pair,
    PromptMix,
    SampleSize,
    TrainingPair,
    Variant,
    apart,
    assign_variants,
    read_pairs,
)
from diffu_moss.train.records import ADAPTER_WEIGHTS, CACHE_RECORD
from diffu_moss.train.restyle import (
    OFF_PAGE,
    OVERLAP,
    SLANTS,
    Colourise,
    HandRefused,
    HandRestyle,
    Microfilm,
    Restyle,
    Tint,
    restyled,
)
from diffu_moss.train.sigmas import draw_sigmas, sigma_grid, sigma_weights
from diffu_moss.transformer import MossTransformer

INK_HEX = "#5A4632"  # the described caption's ink, one of its look words
# an entry's and a cache record's keys as written before look changes
ENTRY_KEYS = ["file", "pair", "variant", "prompt", "tokens", "target_size", "reference_size"]
RECORD_KEYS = ["pairs", "model", "seed", "pairs_read", "pairs_sharing_lines", "entries", "latents_only"]


@pytest.fixture(scope="module")
def latents_only_cache(tiny_moss: Path, tiny_pairs: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The tiny pairs cached as ``tiny_pair_cache`` is, but their latents alone (``--latents-only``)."""
    cache = tmp_path_factory.mktemp("latents_only") / "cache"
    write_cache(tiny_moss, tiny_pairs / "train.jsonl", cache, CacheSettings(latents_only=True, device="cpu"))
    return cache


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


def _page_pair(page: str, texts: list[str], sample: str) -> PagePair:
    """A page pair as diffu_page.data.page_pairs writes one: page ``page`` of lines ``texts`` (64 x 32 px), drawn from
    page ``sample``; its prompt in every form a Layout prompt of the paper and those lines."""
    items = [{"label": "object", "caption": "Aged paper.", "bbox": [-16, -32, 16, 32]}]
    items += [
        {"label": "text", "caption": "Handwritten line.", "text": text, "bbox": [0, -30, 8, 30]} for text in texts
    ]
    prompt = json.dumps({"summary": {"caption": "A page."}, "items": items})
    return PagePair(
        id=page,
        target=Path("pages", f"{page}.jpg"),
        reference=Path("refs", f"{sample}.jpg"),
        size=(64, 32),
        prompts=dict.fromkeys(Variant, prompt),
    )


def test_apart_leaves_out_a_page_pair_whose_sample_page_holds_a_line_of_its_page() -> None:
    pairs: list[TrainingPair] = [
        _page_pair("a", ["Anno 1701", "den 5 Maj"], sample="b"),
        _page_pair("b", ["Anno 1701", "den 6 Maj"], sample="c"),
        _page_pair("c", ["den 7 Maj"], sample="a"),
    ]

    kept = apart(pairs)

    assert [pair.id for pair in kept] == ["b", "c"]


@pytest.mark.parametrize(
    ("variant", "size", "message"),
    [
        pytest.param(
            Variant.LOOK, (64, 48), r"is \(64, 48\) px, its prompts were compiled for \(64, 32\)", id="another-size"
        ),
        pytest.param(Variant.SAMPLE, (64, 32), "has no sample prompt, built before that form", id="a-form-it-lacks"),
    ],
)
def test_page_pair_refuses_a_prompt_its_row_was_not_compiled_for(
    variant: Variant, size: tuple[int, int], message: str
) -> None:
    pair = _page_pair("a", ["Anno 1701"], sample="b")
    built_before_the_sample_form = pair.model_copy(
        update={"prompts": {form: prompt for form, prompt in pair.prompts.items() if form is not Variant.SAMPLE}}
    )

    with pytest.raises(ValueError, match=message):
        built_before_the_sample_form.prompt(variant, size)


def test_moss_training_noise_levels_have_the_quartiles_of_mosss_own_sampler() -> None:
    cumulative = sigma_weights().cumsum(0)

    quartiles = [
        sigma_grid()[torch.searchsorted(cumulative, torch.tensor(q, dtype=torch.float64))] for q in (0.25, 0.5, 0.75)
    ]

    assert [round(level.item(), 3) for level in quartiles] == [0.635, 0.781, 0.886]  # its median documented as 0.78


def test_shifted_noise_levels_move_mosss_draws_towards_the_clean_end_by_the_timestep_shift() -> None:
    shift = 1 / 3

    drawn = draw_sigmas(100_000, torch.Generator().manual_seed(0), shift)

    unshifted = draw_sigmas(100_000, torch.Generator().manual_seed(0)).double()
    assert torch.equal(drawn, (shift * unshifted / (1 + (shift - 1) * unshifted)).float())
    quartiles = drawn.quantile(torch.tensor([0.25, 0.5, 0.75]))
    assert [round(level, 3) for level in quartiles.tolist()] == [0.367, 0.543, 0.723]  # MOSS's own: 0.635, 0.781, 0.887


def test_training_with_a_sigma_shift_draws_its_samples_at_the_shifted_levels(
    tiny_moss_dit: Path, tiny_pair_cache: Path, tiny_reference_run: Path, tmp_path: Path
) -> None:
    settings = TrainSettings(steps=1, save_every=1, sigma_shift=0.5, device="cpu")

    train(tiny_moss_dit, tiny_pair_cache, tmp_path, settings)

    logged = json.loads((tmp_path / LOG).read_text(encoding="utf-8").splitlines()[0])["sigmas"]
    unshifted = json.loads((tiny_reference_run / LOG).read_text(encoding="utf-8").splitlines()[0])["sigmas"]
    # the same seed draws the same levels, each shifted: 0.5 sigma / (1 - 0.5 sigma)
    assert logged == [torch.tensor(0.5 * sigma / (1 - 0.5 * sigma)).float().item() for sigma in unshifted]


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


def test_cache_holds_a_fifth_of_the_pairs_in_each_form_but_two_fifths_with_the_sample_image_alone(
    tiny_pair_cache: Path,
) -> None:
    entries = read_entries(tiny_pair_cache)

    counts = Counter(entry.variant for entry in entries)
    forms = {
        (
            entry.variant,
            load_sample(tiny_pair_cache, entry, "cpu").reference is not None,
            "the sample image" in entry.prompt,
            INK_HEX in entry.prompt,
            json.loads(entry.prompt)["items"][0]["caption"],  # the paper's
        )
        for entry in entries
    }
    # 10 of the 11 pairs: vol_03's reference (vol_04's crop) holds its first line, so it is left out
    assert counts == {Variant.NO_REFERENCE: 2, Variant.LOOK: 2, Variant.PLAIN: 2, Variant.SAMPLE: 4}
    described_paper = "Aged cream paper (#D8C8A8) of a bound archival volume."
    assert forms == {
        (Variant.NO_REFERENCE, False, False, True, described_paper),  # no image, no citation, the look in words
        (Variant.LOOK, True, True, True, described_paper),
        (Variant.PLAIN, True, True, False, "Aged paper of a bound archival volume."),  # the look from the image alone
        # a style pair's caption names no hand: its plain form, but for its paper, the sample image's
        (Variant.SAMPLE, True, True, False, "The paper of the sample image."),
    }


def test_cache_without_look_changes_writes_its_entries_and_record_as_before(tiny_pair_cache: Path) -> None:
    lines = (tiny_pair_cache / ENTRIES).read_text(encoding="utf-8").splitlines()

    metadata = []
    for entry in read_entries(tiny_pair_cache):
        with safe_open(str(tiny_pair_cache / entry.file), framework="pt") as written:
            metadata.append(written.metadata()["entry"])

    assert [list(json.loads(line)) for line in lines] == [ENTRY_KEYS] * len(lines)
    assert metadata == lines  # each file's entry as entries.jsonl lists it
    assert list(json.loads((tiny_pair_cache / CACHE_RECORD).read_text(encoding="utf-8"))) == RECORD_KEYS


def test_cache_shard_caches_only_its_places_of_the_plan_and_leaves_the_cache_unfinished(
    tiny_moss: Path, tiny_pairs: Path, tiny_pair_cache: Path, tmp_path: Path
) -> None:
    whole = [entry.file for entry in read_entries(tiny_pair_cache)]

    record = write_cache(
        tiny_moss, tiny_pairs / "train.jsonl", tmp_path, CacheSettings(device="cpu", shard=1, shards=3)
    )

    assert record is None
    assert sorted(path.name for path in tmp_path.iterdir()) == whole[1::3]  # no entries.jsonl, no cache.json


def test_cache_shards_then_a_run_without_shards_write_the_cache_a_single_run_writes(
    tiny_moss: Path, tiny_pairs: Path, tiny_pair_cache: Path, tmp_path: Path
) -> None:
    manifest = tiny_pairs / "train.jsonl"
    for shard in (0, 2):  # shard 1 left unrun: the finishing run caches its entries itself
        write_cache(tiny_moss, manifest, tmp_path, CacheSettings(device="cpu", shard=shard, shards=3))

    write_cache(tiny_moss, manifest, tmp_path, CacheSettings(device="cpu"))

    assert (tmp_path / ENTRIES).read_text(encoding="utf-8") == (tiny_pair_cache / ENTRIES).read_text(encoding="utf-8")
    assert (tmp_path / CACHE_RECORD).read_text(encoding="utf-8") == (tiny_pair_cache / CACHE_RECORD).read_text(
        encoding="utf-8"
    )
    for entry in read_entries(tiny_pair_cache):
        sharded, single = load_file(str(tmp_path / entry.file)), load_file(str(tiny_pair_cache / entry.file))
        assert sharded.keys() == single.keys()
        assert all(torch.equal(sharded[name], single[name]) for name in single)


def test_cache_gives_exactly_its_share_of_the_sample_forms_entries_a_look_change_and_the_rest_none(
    tiny_pair_cache: Path, tiny_restyled_cache: Path
) -> None:
    plain = read_entries(tiny_pair_cache)

    entries = read_entries(tiny_restyled_cache)

    changed = [entry for entry in entries if entry.restyle is not None]
    # the entries without: the same pairs in the same forms and prompts, and of the 4 in the sample form 2 changed
    assert [entry.model_copy(update={"restyle": None}) for entry in entries] == plain
    assert [(entry.pair, entry.variant, entry.restyle.kind) for entry in changed if entry.restyle] == [
        ("vol_08", Variant.SAMPLE, "microfilm"),
        ("vol_09", Variant.SAMPLE, "colourise"),
    ]
    record = read_record(tiny_restyled_cache)
    assert (record.restyle_share, record.restyled) == (0.4, {"colourise": 1, "microfilm": 1})
    unchanged = [entry.file for entry in entries if entry.restyle is None]
    differ = [
        file for file in unchanged if (tiny_restyled_cache / file).read_bytes() != (tiny_pair_cache / file).read_bytes()
    ]
    assert differ == []


def test_cache_gives_a_pairs_target_and_sample_the_look_change_its_entry_records(
    tiny_moss: Path, tiny_pairs: Path, tiny_restyled_cache: Path
) -> None:
    pairs = {pair.id: pair for pair in read_pairs(tiny_pairs / "train.jsonl")}
    vae = encoder_pipeline(tiny_moss, "cpu", ("vae",)).vae

    def encoded(path: Path, restyle: Restyle) -> torch.Tensor:
        with Image.open(path) as image:
            return encode_latents(vae, image_pixels(restyled(image, restyle))).to(CACHE_DTYPE)

    changed = [
        (pairs[entry.pair], entry.restyle, load_sample(tiny_restyled_cache, entry, "cpu"))
        for entry in read_entries(tiny_restyled_cache)
        if entry.restyle is not None
    ]

    targets = [
        pair.id for pair, restyle, sample in changed if not torch.equal(sample.target, encoded(pair.target, restyle))
    ]
    references = [
        pair.id
        for pair, restyle, sample in changed
        if sample.reference is None or not torch.equal(sample.reference, encoded(pair.reference, restyle))
    ]
    assert (targets, references) == ([], [])


def _shuffled(image: Image.Image, order: list[int]) -> Image.Image:
    """An RGB image's pixels in ``order``, row by row."""
    data = image.tobytes()
    return Image.frombytes("RGB", image.size, b"".join(data[3 * k : 3 * k + 3] for k in order))


@pytest.mark.parametrize(
    "restyle",
    [
        pytest.param(Microfilm(crush=0.2, paper=0.8, gamma=1.2), id="microfilm"),
        pytest.param(Colourise(crush=0.05, gamma=1.1, paper=(229, 207, 165), ink=(62, 45, 33)), id="colourise"),
        pytest.param(Tint(warmth=0.03, contrast=1.1, brightness=-0.03, gamma=1.1), id="tint"),
    ],
)
def test_restyle_moves_no_pixel_and_keeps_the_images_size(restyle: Restyle) -> None:
    generator = torch.Generator().manual_seed(0)
    noise = torch.randint(0, 256, (48 * 32 * 3,), generator=generator, dtype=torch.uint8)
    image = Image.frombytes("RGB", (48, 32), bytes(noise.tolist()))
    order = torch.randperm(48 * 32, generator=generator).tolist()

    changed = restyled(image, restyle)

    assert (changed.mode, changed.size) == ("RGB", image.size)
    # each pixel's new colour, wherever it is
    assert restyled(_shuffled(image, order), restyle).tobytes() == _shuffled(changed, order).tobytes()


PAGE = (640, 960)  # px (width, height): a page's canvas
LINES = [  # its lines' boxes (y0, x0, y1, x1, px from its centre): two near the top, a margin note, one near the bottom
    [-400, -250, -350, 200],
    [-340, -250, -290, 230],
    [-100, -300, -60, -270],
    [300, -200, 350, 250],
]
PAPER_RGB, INK_RGB = (230, 220, 200), (40, 30, 20)


def _prompt(boxes: list[list[int]], size: tuple[int, int]) -> str:
    """A compiled Layout prompt of a page of ``size``: its paper over the whole canvas, then a line in each box."""
    width, height = size
    paper = {"label": "object", "caption": "Aged paper.", "bbox": [-height // 2, -width // 2, height // 2, width // 2]}
    lines = [
        {"label": "text", "caption": "Handwritten line.", "text": f"rad {k}", "bbox": box}
        for k, box in enumerate(boxes)
    ]
    layout = {"summary": {"caption": "A page."}, "items": [paper, *lines], "bbox_coord_space": "abs_centralized"}
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"))


def _corners_moved(box: list[int], hand: HandRestyle, size: tuple[int, int]) -> list[int]:
    """Where ``hand`` moves a box: each of its four corners sheared (x - tan(slant) y) and scaled to the changed page,
    their bounding box's top and left edges floored and its bottom and right ceiled."""
    (width, height), (changed_width, changed_height) = size, hand.size(size)
    shear = math.tan(math.radians(hand.slant))
    y0, x0, y1, x1 = box
    corners = [
        ((x - shear * y) * changed_width / width, y * changed_height / height) for x in (x0, x1) for y in (y0, y1)
    ]
    xs, ys = [x for x, _ in corners], [y for _, y in corners]
    return [
        math.floor(round(min(ys), 9)),
        math.floor(round(min(xs), 9)),
        math.ceil(round(max(ys), 9)),
        math.ceil(round(max(xs), 9)),
    ]


@pytest.mark.parametrize(
    "hand",
    [
        pytest.param(HandRestyle(slant=7.5, stroke=1.0, width=1.0), id="leaning-right"),
        pytest.param(HandRestyle(slant=-9.0, stroke=-0.5, width=1.0), id="leaning-left"),
        pytest.param(HandRestyle(slant=4.0, stroke=0.5, width=1.15), id="leaning-right-and-wider"),
    ],
)
def test_hand_change_moves_each_box_of_a_prompt_as_the_shear_and_the_width_move_its_corners(hand: HandRestyle) -> None:
    moved, size = hand.moved_prompt(_prompt(LINES, PAGE), PAGE)

    items = json.loads(moved)["items"]
    width, height = hand.size(PAGE)
    assert size == (width, height)
    assert items[0]["bbox"] == [-height // 2, -width // 2, height // 2, width // 2]  # the paper: the whole new page
    assert [item["bbox"] for item in items[1:]] == [_corners_moved(box, hand, PAGE) for box in LINES]
    assert [item["text"] for item in items[1:]] == ["rad 0", "rad 1", "rad 2", "rad 3"]


@pytest.mark.parametrize(
    ("boxes", "reason"),
    [
        pytest.param([[-400, 0, -350, 310]], OFF_PAGE, id="a-line-near-the-top-off-the-right-edge"),
        pytest.param([[300, -310, 350, 0]], OFF_PAGE, id="a-line-near-the-bottom-off-the-left-edge"),
        pytest.param([[-60, -300, 0, -2], [-60, 0, 0, 300]], OVERLAP, id="lines-side-by-side-onto-each-other"),
    ],
)
def test_hand_change_refuses_to_move_a_line_off_its_page_or_onto_another(boxes: list[list[int]], reason: str) -> None:
    leaning_right = HandRestyle(slant=5.0, stroke=1.0, width=1.0)

    with pytest.raises(HandRefused, match=reason):
        leaning_right.moved_prompt(_prompt(boxes, PAGE), PAGE)


def _largest_slant_kept(boxes: list[list[int]], size: tuple[int, int]) -> float:
    """The largest slant, degrees (to within a ten-thousandth, up to the largest drawn), that a page of ``size`` with
    lines in ``boxes`` can be sheared by either way without ``moved_prompt`` refusing it: found by halving."""
    prompt = _prompt(boxes, size)

    def kept(slant: float) -> bool:
        try:
            for sign in (1, -1):
                HandRestyle(slant=sign * slant, stroke=0.0, width=1.0).moved_prompt(prompt, size)
        except HandRefused:
            return False
        return True

    low, high = 0.0, SLANTS[1]
    if kept(high):
        return high
    while high - low > 1e-4:
        middle = (low + high) / 2
        low, high = (middle, high) if kept(middle) else (low, middle)
    return low


@pytest.mark.parametrize(
    ("boxes", "room", "slant"),
    [
        pytest.param([[300, -310, 350, 0]], 1.636, -0.818, id="a-line-near-the-bottom-left-corner"),
        pytest.param([[-60, -300, 0, -4], [-60, 0, 0, 300]], 3.814, -1.907, id="two-lines-side-by-side"),
        pytest.param([[-10, -50, 10, 50]], 10.0, -5.0, id="more-room-than-the-largest-slant"),
        pytest.param([[300, -317, 350, 0]], 0.491, 0.0, id="under-a-degree-of-room-no-slant"),
        pytest.param([[-400, -250, -350, 320]], 0.0, 0.0, id="a-line-at-the-edge-no-room"),
    ],
)
def test_hand_change_slants_a_page_within_the_largest_slant_its_lines_leave_room_for(
    boxes: list[list[int]], room: float, slant: float
) -> None:
    drawn = HandRestyle(slant=-5.0, stroke=0.5, width=1.1)  # half the largest slant drawn, leaning left

    within = drawn.within(_prompt(boxes, PAGE), PAGE)

    assert (within.slant_bound, within.slant, within.stroke, within.width) == (room, slant, 0.5, 1.1)
    assert within.slant_bound == pytest.approx(_largest_slant_kept(boxes, PAGE), abs=1e-3)  # as moved_prompt keeps
    within.moved_prompt(_prompt(boxes, PAGE), PAGE)  # and keeps this one


@pytest.mark.parametrize(
    ("stroke", "ink_gained"),
    [pytest.param(1.0, 1.0, id="a-pixel-thicker"), pytest.param(-0.5, -1.0, id="half-a-pixel-thinner")],
)
def test_hand_change_of_stroke_lays_on_or_takes_off_ink_and_leaves_the_paper_grain_beyond_the_strokes(
    stroke: float, ink_gained: float
) -> None:
    grain = torch.randint(0, 14, (64, 96), generator=torch.Generator().manual_seed(0))  # up to 13 levels darker
    pixels = (torch.tensor(PAPER_RGB).view(3, 1, 1) - grain).permute(1, 2, 0)
    pixels[16:48, 40:50] = torch.tensor(INK_RGB)  # a stroke 10 px wide
    page = Image.fromarray(pixels.to(torch.uint8).numpy())

    changed = HandRestyle(slant=0.0, stroke=stroke, width=1.0).apply(page)

    before, after = (torch.from_numpy(np.array(image.convert("L"))).float() for image in (page, changed))
    beyond = torch.ones(64, 96, dtype=torch.bool)
    beyond[14:50, 38:52] = False  # 2 px around the stroke
    assert torch.equal(before[beyond], after[beyond])  # the paper beyond the stroke keeps every pixel
    assert torch.sign(before.sum() - after.sum()).item() == ink_gained  # darker where the stroke grew


def test_hand_change_keeps_every_lines_ink_inside_its_moved_box_and_none_outside_them() -> None:
    hand = HandRestyle(slant=8.0, stroke=1.0, width=1.1)
    page = Image.new("RGB", PAGE, PAPER_RGB)
    draw = ImageDraw.Draw(page)
    middle_x, middle_y = PAGE[0] // 2, PAGE[1] // 2
    for y0, x0, y1, x1 in LINES:  # each line's ink 4 px inside its box, as a slot's margins hold a line's
        draw.rectangle((middle_x + x0 + 4, middle_y + y0 + 4, middle_x + x1 - 5, middle_y + y1 - 5), fill=INK_RGB)

    changed = hand.apply(page)

    moved, (width, height) = hand.moved_prompt(_prompt(LINES, PAGE), PAGE)
    boxes = [item["bbox"] for item in json.loads(moved)["items"][1:]]
    rows, columns = torch.nonzero(torch.from_numpy(np.asarray(changed.convert("L")) < 128), as_tuple=True)
    ink = {(x - width // 2, y - height // 2) for x, y in zip(columns.tolist(), rows.tolist(), strict=True)}
    inked = [{(x, y) for x, y in ink if y0 <= y < y1 and x0 <= x < x1} for y0, x0, y1, x1 in boxes]
    assert changed.size == (width, height)
    assert [len(line) > 0 for line in inked] == [True] * len(LINES)  # each box holds its line
    assert ink == set().union(*inked)  # and no ink lies outside them


def test_cache_gives_a_pairs_target_and_sample_the_hand_change_its_entry_records_its_prompt_moved_with_its_lines(
    tiny_moss: Path, tiny_pairs: Path, tiny_pair_cache: Path, tiny_hand_cache: Path
) -> None:
    pairs = {pair.id: pair for pair in read_pairs(tiny_pairs / "train.jsonl")}
    unchanged = {entry.file: entry for entry in read_entries(tiny_pair_cache)}
    vae = encoder_pipeline(tiny_moss, "cpu", ("vae",)).vae

    entries = read_entries(tiny_hand_cache)

    def encoded(path: Path, hand: HandRestyle) -> torch.Tensor:
        with Image.open(path) as image:
            return encode_latents(vae, image_pixels(styled(image, hand=hand))).to(CACHE_DTYPE)

    changed = [(entry, entry.hand) for entry in entries if entry.hand is not None]
    # 2 of the 4 in the sample form (vol_08, vol_09, vol_01, vol_06), by a stream of their own
    assert [(entry.pair, entry.variant) for entry, _ in changed] == [
        ("vol_09", Variant.SAMPLE),
        ("vol_06", Variant.SAMPLE),
    ]
    for entry, hand in changed:
        sample, size = load_sample(tiny_hand_cache, entry, "cpu"), unchanged[entry.file].target_size
        assert hand.slant_bound == SLANTS[1]  # a tiny pair's two lines leave room for the largest slant
        assert (entry.prompt, entry.target_size) == hand.moved_prompt(unchanged[entry.file].prompt, size)
        assert torch.equal(sample.target, encoded(pairs[entry.pair].target, hand))
        assert sample.reference is not None
        assert torch.equal(sample.reference, encoded(pairs[entry.pair].reference, hand))
    differ = [
        entry.file
        for entry in entries
        if entry.hand is None
        and (tiny_hand_cache / entry.file).read_bytes() != (tiny_pair_cache / entry.file).read_bytes()
    ]
    assert differ == []
    record = read_record(tiny_hand_cache)
    assert (record.hand_restyle_share, record.hand_restyled, record.hand_no_slant_room, record.hand_refused) == (
        0.5,
        2,
        0,
        {},
    )


def test_content_mix_draws_the_sample_alone_for_three_fifths_and_the_sample_with_the_look_or_nothing_for_a_fifth() -> (
    None
):
    forms = assign_variants(10, seed=0, mix=PromptMix.CONTENT)

    assert Counter(forms) == {Variant.SAMPLE: 6, Variant.SAMPLE_LOOK: 2, Variant.GENERIC: 2}


def test_cache_plan_without_a_limit_draws_each_pairs_form_apart_from_its_place_in_the_manifest() -> None:
    pairs: list[TrainingPair] = [
        Pair(id=f"pair_{k}", target=Path(f"target_{k}.jpg"), reference=Path(f"reference_{k}.jpg"), caption="")
        for k in range(10)
    ]

    planned = planned_entries(pairs, CacheSettings(mix=PromptMix.CONTENT, device="cpu"))

    forms = {plan.pair.id: plan.variant for plan in planned}
    # drawn from the stream that shuffles the pairs, the forms fell in the manifest's order: the first two pairs
    # generic, the next two with the look, the last six the sample alone (a page-pair set's pages all the first forms)
    assert [forms[f"pair_{k}"] for k in range(10)] == [
        Variant.SAMPLE_LOOK,
        Variant.SAMPLE,
        Variant.SAMPLE,
        Variant.SAMPLE_LOOK,
        Variant.GENERIC,
        Variant.SAMPLE,
        Variant.GENERIC,
        Variant.SAMPLE,
        Variant.SAMPLE,
        Variant.SAMPLE,
    ]


def test_cache_refuses_a_mix_whose_forms_its_pairs_lack_before_it_loads_an_encoder(
    tiny_pairs: Path, tmp_path: Path
) -> None:
    (tmp_path / "no_model").mkdir()  # an encoder loaded from it would fail otherwise
    content = CacheSettings(mix=PromptMix.CONTENT, device="cpu")

    # a style pair's caption describes its hand in words: of the content mix it has the sample form alone
    with pytest.raises(ValueError, match=r"4 pairs lack the form they are planned in .* or another mix"):
        write_cache(tmp_path / "no_model", tiny_pairs / "train.jsonl", tmp_path / "cache", content)


PAIR_SIZE = (64, 32)  # px: the tiny pairs' targets


@pytest.fixture(scope="module")
def legible_pairs(tiny_pairs: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The tiny style pairs as a page-pair manifest whose samples were cut at their targets' writing scale (each
    pair's own reference image, as it is), each row's prompts its named mix's forms."""
    forms = MIXES[PromptMix.NAMED].forms
    rows = [
        PagePairRow(
            id=pair.id,
            target=str(pair.target),
            reference=str(pair.reference),
            size=PAIR_SIZE,
            volume="vol",
            prompts={variant: pair.prompt(variant, PAIR_SIZE) for variant in forms},
            tokens=dict.fromkeys(forms, 1),
            sample_size=SampleSize.LEGIBLE,
        )
        for pair in read_pairs(tiny_pairs / "train.jsonl")
    ]
    manifest = tmp_path_factory.mktemp("legible_pairs") / "train.jsonl"
    manifest.write_text("".join(f"{row.model_dump_json()}\n" for row in rows), encoding="utf-8")
    return manifest


@pytest.fixture(scope="module")
def legible_cache(tiny_moss: Path, legible_pairs: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two of the legible pairs cached."""
    cache = tmp_path_factory.mktemp("legible_cache") / "cache"
    write_cache(tiny_moss, legible_pairs, cache, CacheSettings(limit=2, device="cpu"))
    return cache


def test_cache_refuses_pairs_whose_samples_were_sized_by_more_than_one_rule(
    legible_pairs: Path, tmp_path: Path
) -> None:
    rows = legible_pairs.read_text(encoding="utf-8").splitlines()
    shrunk = PagePairRow.model_validate_json(rows[0]).model_copy(update={"sample_size": SampleSize.HALF})
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text("".join(f"{row}\n" for row in [shrunk.model_dump_json(), *rows[1:]]), encoding="utf-8")
    (tmp_path / "no_model").mkdir()

    with pytest.raises(ValueError, match=r"samples are sized by more than one rule \(half, legible\)"):
        write_cache(tmp_path / "no_model", mixed, tmp_path / "cache", CacheSettings(device="cpu"))


def _merged(base: Path, checkpoint: Path, out: Path) -> Path:
    """A LoRA checkpoint's DiT: merged into ``base`` (diffu-moss-export-ref), into ``out``."""
    export_merged(base, checkpoint, out)
    return out


def _its_own(base: Path, checkpoint: Path, out: Path) -> Path:
    """A full fine-tune checkpoint's DiT: its own ``transformer/``."""
    return checkpoint / "transformer"


@pytest.mark.parametrize(
    ("full", "dit_of"),
    [pytest.param(None, _merged, id="lora"), pytest.param(FullSettings(), _its_own, id="full-fine-tune")],
)
def test_model_index_names_how_the_cache_its_dit_trained_on_sized_its_samples(
    full: FullSettings | None,
    dit_of: Callable[[Path, Path, Path], Path],
    tiny_moss_dit: Path,
    legible_cache: Path,
    tmp_path: Path,
) -> None:
    checkpoint = train(tiny_moss_dit, legible_cache, tmp_path / "run", TrainSettings(steps=1, full=full, device="cpu"))
    dit = dit_of(tiny_moss_dit, checkpoint, tmp_path / "export")

    index = write_moss_repo(tmp_path / "repo", base=tiny_moss_dit.parent, transformer=dit)

    extra = index.model_extra or {}
    assert (read_record(legible_cache).sample_size, extra["takes_reference"], extra["sample_size"]) == (
        SampleSize.LEGIBLE,
        True,
        "legible",
    )
    assert ModelIndex.read(tmp_path / "repo") == index


def test_a_full_fine_tune_with_the_sample_above_saves_its_place_and_draws_with_it(
    tiny_moss_dit: Path, legible_cache: Path, tmp_path: Path
) -> None:
    settings = TrainSettings(steps=1, full=FullSettings(), reference_place="above", device="cpu")
    checkpoint = train(tiny_moss_dit, legible_cache, tmp_path / "run", settings)

    above = MossTransformer.from_pretrained(str(checkpoint / "transformer"), torch_dtype=torch.float32).eval()
    over = MossTransformer.from_pretrained(
        str(checkpoint / "transformer"), torch_dtype=torch.float32, reference_place="over"
    ).eval()
    channels = above.config.latent_channels
    generator = torch.Generator().manual_seed(0)
    latents, sample = (torch.randn(1, channels, 2, 3, generator=generator) for _ in range(2))
    text = torch.randn(1, 4, above.config.text_dim, generator=generator)
    with torch.no_grad():
        drawn = [dit(latents, torch.tensor([500.0]), text, None, references=[sample]) for dit in (above, over)]

    assert above.config.reference_place == "above"
    assert not torch.allclose(drawn[0], drawn[1])


def test_a_lora_refuses_a_sample_place_other_than_its_base_dits(legible_cache: Path, tmp_path: Path) -> None:
    (tmp_path / "no_dit").mkdir()  # refused before the DiT is loaded

    with pytest.raises(ValueError, match="--reference-place needs --full"):
        train(
            tmp_path / "no_dit",
            legible_cache,
            tmp_path / "run",
            TrainSettings(steps=1, reference_place="above", device="cpu"),
        )


def test_training_refuses_a_validation_cache_whose_samples_were_sized_otherwise(
    legible_cache: Path, tiny_pair_cache: Path, tmp_path: Path
) -> None:
    (tmp_path / "no_dit").mkdir()  # refused before the DiT is loaded

    with pytest.raises(ValueError, match=r"sized its samples legible, the validation cache .* half"):
        train(
            tmp_path / "no_dit",
            legible_cache,
            tmp_path / "run",
            TrainSettings(steps=1, device="cpu"),
            val_cache=tiny_pair_cache,
        )


def test_training_draws_every_entry_once_a_pass_so_a_run_keeps_the_caches_mix(
    tiny_moss_dit: Path, tiny_pair_cache: Path, tmp_path: Path
) -> None:
    train(tiny_moss_dit, tiny_pair_cache, tmp_path, TrainSettings(steps=5, save_every=5, device="cpu"))

    steps = [json.loads(line) for line in (tmp_path / LOG).read_text(encoding="utf-8").splitlines()]
    drawn = [sample for step in steps for sample in zip(step["pairs"], step["variants"], strict=True)]
    # 5 steps of 2: one pass over the 10 entries, each once, in the form it was cached in
    assert sorted(drawn) == sorted((entry.pair, entry.variant.value) for entry in read_entries(tiny_pair_cache))


def test_entry_tokens_count_the_drawn_images_the_prompts_and_the_samples() -> None:
    entry = CacheEntry(
        file="000000_sample.safetensors",
        pair="vol_00",
        variant=Variant.SAMPLE,
        prompt="{}",
        tokens=5,
        target_size=(64, 32),
        reference_size=(32, 32),
    )

    # a 4 x 2 latent grid, 5 prompt tokens and the sample's 2 x 2; without a sample, the first two alone
    assert entry_tokens(entry) == 8 + 5 + 4
    assert entry_tokens(entry.model_copy(update={"variant": Variant.GENERIC, "reference_size": None})) == 8 + 5


def test_size_grouped_order_draws_every_entry_once_a_pass_in_shuffled_rounds_of_like_size() -> None:
    sizes = [100] * 101 + [200] * 101  # 202 entries of two sizes: 25 rounds of 8 a pass and 2 entries over
    groups = SizeGroups(sizes=sizes, per_round=8, block=32)  # one block holds a whole pass
    order = SampleOrder(len(sizes), torch.Generator().manual_seed(0), groups=groups)

    draws = [next(order) for _ in range(3 * len(sizes))]

    passes = [draws[start : start + len(sizes)] for start in range(0, len(draws), len(sizes))]
    assert [sorted(drawn) for drawn in passes] == [list(range(len(sizes)))] * 3
    rounds = [[sizes[entry] for entry in draws[start : start + 8]] for start in range(0, len(draws) - 7, 8)]
    mixed = [index for index, drawn in enumerate(rounds) if len(set(drawn)) > 1]
    # a pass's sorted entries change size inside one round, and a round two passes share may hold both: 3 + 2 at most,
    # which holds only if each pass's rounds are counted on from where the pass before stopped
    assert len(mixed) <= 5
    first_pass = [drawn[0] for drawn in rounds[:25]]
    assert first_pass != sorted(first_pass)  # its rounds shuffled, not dealt from the smallest to the largest


def test_size_grouped_order_sorts_inside_blocks_of_rounds_so_no_two_rounds_of_a_block_overlap_in_size() -> None:
    sizes = [(index * 37) % 101 for index in range(256)]
    groups = SizeGroups(sizes=sizes, per_round=4, block=8)
    order = SampleOrder(len(sizes), torch.Generator().manual_seed(0), groups=groups)

    drawn = [next(order) for _ in range(len(sizes))]

    assert sorted(drawn) == list(range(len(sizes)))
    for start in range(0, len(drawn), 32):  # a block: 8 rounds of 4
        rounds = [[sizes[entry] for entry in drawn[first : first + 4]] for first in range(start, start + 32, 4)]
        spans = sorted((min(sizes_), max(sizes_)) for sizes_ in rounds)
        assert all(low >= smaller_high for (_, smaller_high), (low, _) in pairwise(spans))


def test_training_in_one_process_refuses_to_group_its_rounds_by_size(
    tiny_moss_dit: Path, tiny_pair_cache: Path, tmp_path: Path
) -> None:
    settings = TrainSettings(steps=1, size_block=4, device="cpu")

    with pytest.raises(ValueError, match="needs several ranks"):
        train(tiny_moss_dit, tiny_pair_cache, tmp_path / "run", settings)


def test_training_limited_to_n_pairs_trains_on_the_caches_first_n_drawn_with_their_reference(
    tiny_moss_dit: Path, tiny_pair_cache: Path, tmp_path: Path
) -> None:
    train(tiny_moss_dit, tiny_pair_cache, tmp_path, TrainSettings(steps=3, limit_pairs=6, save_every=3, device="cpu"))

    steps = [json.loads(line) for line in (tmp_path / LOG).read_text(encoding="utf-8").splitlines()]
    # 3 steps of 2: one pass over vol_08, vol_09, vol_01 and vol_06 (sample), vol_04 (plain) and vol_02 (look); vol_05,
    # cached before vol_02, takes none
    assert sorted(pair for step in steps for pair in step["pairs"]) == sorted(
        ["vol_08", "vol_09", "vol_01", "vol_06", "vol_04", "vol_02"]
    )


def test_training_moves_every_lora_a_weight_at_the_recipes_learning_rate(
    tiny_moss_dit: Path, tiny_reference_run: Path
) -> None:
    initial = get_peft_model_state_dict(lora_transformer(tiny_moss_dit, TrainSettings().lora, device="cpu", seed=0))

    trained = load_file(str(tiny_reference_run / "checkpoints" / "step_000005" / ADAPTER_WEIGHTS))

    moved = [(trained[name] != initial[name]).float().mean().item() for name in initial if ".lora_A." in name]
    assert min(moved) > 0.99  # warmup steps of 2e-6 to 1e-5: below bfloat16's resolution on weights of about 0.1


def test_validation_scores_each_of_a_pairs_forms_at_the_same_noise_level_and_noise(
    tiny_moss: Path, tiny_pairs: Path, tmp_path: Path
) -> None:
    write_cache(tiny_moss, tiny_pairs / "train.jsonl", tmp_path, CacheSettings(limit=2, every_form=True, device="cpu"))
    blind = MossTransformer(num_layers=0, dim=32, ffn_dim=64, num_heads=2, text_dim=16, freq_dim=16, latent_channels=16)

    losses = validation_losses(blind.eval(), tmp_path, read_entries(tmp_path), device="cpu", seed=0)

    # a DiT without blocks reads neither the prompt nor the reference: a pair's forms differ in nothing else
    assert sorted(losses) == ["look", "no_reference", "plain", "sample"]
    assert losses["look"] == losses["plain"] == losses["sample"] == losses["no_reference"]


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


def test_training_with_a_wind_down_steps_its_last_steps_at_a_learning_rate_falling_linearly_to_decay_to(
    tiny_moss_dit: Path, tiny_pair_cache: Path, tmp_path: Path
) -> None:
    settings = TrainSettings(
        steps=6, learning_rate=1e-2, warmup=1, decay_steps=3, decay_to=0.25, save_every=6, device="cpu"
    )

    train(tiny_moss_dit, tiny_pair_cache, tmp_path / "run", settings)

    rates = [json.loads(line)["learning_rate"] for line in (tmp_path / "run" / LOG).read_text().splitlines()]
    assert rates == pytest.approx([1e-2, 1e-2, 1e-2, 0.75e-2, 0.5e-2, 0.25e-2])


def test_training_with_a_profile_dir_writes_its_ranks_trace_of_steps_4_to_6(
    tiny_moss_dit: Path, tiny_pair_cache: Path, tmp_path: Path
) -> None:
    settings = TrainSettings(steps=6, save_every=6, device="cpu")

    train(tiny_moss_dit, tiny_pair_cache, tmp_path / "run", settings, profile_dir=tmp_path / "profile")

    events = json.loads((tmp_path / "profile" / "trace_rank0.json").read_text(encoding="utf-8"))["traceEvents"]
    profiled = sorted({event["name"] for event in events if str(event.get("name", "")).startswith("ProfilerStep#")})
    assert profiled == ["ProfilerStep#3", "ProfilerStep#4", "ProfilerStep#5"]  # the profiler counts from 0


def test_training_refuses_a_cache_without_entries_before_it_loads_the_dit(tiny_moss: Path, tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text("", encoding="utf-8")
    write_cache(tiny_moss, tmp_path / "train.jsonl", tmp_path / "empty", CacheSettings(device="cpu"))
    (tmp_path / "no_dit").mkdir()  # a folder without a DiT: loading one from it would fail otherwise

    with pytest.raises(ValueError, match="holds no entry to train on"):
        train(tmp_path / "no_dit", tmp_path / "empty", tmp_path / "run", TrainSettings(steps=1, device="cpu"))


def test_prompt_encoder_encodes_each_prompt_exactly_as_the_cache_holds_it_by_a_frozen_encoder_in_eval_mode(
    tiny_moss: Path, tiny_pair_cache: Path
) -> None:
    entries = read_entries(tiny_pair_cache)
    encoder = PromptEncoder(tiny_moss, "cpu")

    encoded = [encoder.encode(entry.prompt) for entry in entries]

    cached = [load_sample(tiny_pair_cache, entry, "cpu").prompt_embeds for entry in entries]
    assert [entry.pair for entry, a, b in zip(entries, encoded, cached, strict=True) if not torch.equal(a, b)] == []
    text_encoder = encoder.pipe.text_encoder
    assert [name for name, weight in text_encoder.named_parameters() if weight.requires_grad] == []
    assert text_encoder.training is False
    assert [embeds.is_inference() for embeds in encoded] == [False] * len(entries)  # the DiT's backward may save them


def test_training_that_encodes_its_prompts_from_a_latents_only_cache_logs_exactly_what_the_cached_run_logs(
    tiny_moss_dit: Path, latents_only_cache: Path, tiny_reference_run: Path, tmp_path: Path
) -> None:
    settings = TrainSettings(steps=5, save_every=5, val_every=5, encode_text=True, device="cpu")

    train(tiny_moss_dit, latents_only_cache, tmp_path, settings)

    logs = [
        [
            {key: value for key, value in json.loads(line).items() if key != "seconds"}
            for line in log.read_text(encoding="utf-8").splitlines()
        ]
        for log in (tmp_path / LOG, tiny_reference_run / LOG)
    ]
    assert logs[0] == logs[1]  # the same samples, noise levels, losses and gradient norms, steps 1 to 5
    files = [load_file(str(latents_only_cache / entry.file)) for entry in read_entries(latents_only_cache)]
    assert [name for tensors in files for name in tensors if name == "prompt_embeds"] == []


def test_training_refuses_a_latents_only_cache_when_it_does_not_encode_its_prompts(
    latents_only_cache: Path, tmp_path: Path
) -> None:
    (tmp_path / "no_dit").mkdir()  # refused before the DiT is loaded

    with pytest.raises(ValueError, match="holds no prompt embeddings .* train with --encode-text"):
        train(tmp_path / "no_dit", latents_only_cache, tmp_path / "run", TrainSettings(steps=1, device="cpu"))
