"""Fixtures shared by every workspace member's tests: PAGE pages to plan and draw, a crop and its training record,
diffusers' tiny FLUX.3 test model (with a VAE that downsamples as FLUX.3's does), a tiny diffu model repo (a tiny
Qwen3-VL (tiny FLUX.3's), a tiny Qwen-Image VAE and a tiny transformer, all random weights) and its glyph model, a
tiny MOSS-Image repo (MOSS's own tokenizer, a tiny Qwen3.5, a tiny FLUX.2-class VAE and a tiny DiT, random weights),
and a reference training on it: a tiny style-pair set in the described reference form, its caches, a short training
run and the MOSS repo of its merged export, which takes a reference image."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Protocol

import pytest
import torch
from diffusers import AutoencoderKLFlux2, AutoencoderKLQwenImage, Flux3Transformer2DModel
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors.torch import save_file
from transformers import Qwen3_5Config, Qwen3_5Model

from diffu.configs import DiffuConfig, DiffuSize, TextEncoder, Vae
from diffu.init import write_model_repo
from diffu.repo import ModelIndex
from diffu_moss.init import write_moss_repo
from diffu_moss.train.cache import CacheSettings, write_cache
from diffu_moss.train.export import export_merged
from diffu_moss.train.loop import TrainSettings, train
from diffu_moss.transformer import MossTransformer
from diffu_page.captions.look import ScanLook
from diffu_page.captions.prompt import PageStyle
from diffu_page.captions.provenance import Provenance
from diffu_page.captions.records import CropRecord, crop_record
from diffu_page.crops.geometry import Crop, CropSettings, crops_for_page, regions_of
from diffu_page.layout import load_xml_layout


def _line(line_id: str, x0: int, x1: int, baseline: int, text: str) -> str:
    return (
        f'<TextLine id="{line_id}"><Coords points="{x0},{baseline - 60} {x1},{baseline - 60} {x1},{baseline + 20} '
        f'{x0},{baseline + 20}"/><Baseline points="{x0},{baseline} {x1},{baseline}"/>'
        f"<TextEquiv><Unicode>{text}</Unicode></TextEquiv></TextLine>"
    )


@pytest.fixture
def body_and_notes_page(tmp_path: Path) -> Path:
    """A 2000 x 1900 px page: 14 body lines at a 100 px pitch, and two margin notes left of body lines 3 and 4."""
    body = "".join(_line(f"b{i}", 600, 1900, 300 + 100 * i, f"body line {i} of the running text") for i in range(14))
    notes = "".join(_line(f"n{i}", 120, 520, baseline, f"note {i}") for i, baseline in enumerate((600, 700)))
    path = tmp_path / "page.xml"
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">'
        '<Page imageFilename="page.jpg" imageWidth="2000" imageHeight="1900">'
        f'<TextRegion id="body" type="paragraph">{body}</TextRegion>'
        f'<TextRegion id="notes" type="marginalia">{notes}</TextRegion>'
        "</Page></PcGts>",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def two_column_page(tmp_path: Path) -> Path:
    """A 2500 x 1200 px page with two columns of three lines, planned as two crops of three lines each: the smallest
    page whose crops a pipeline draws more than one of."""
    columns = "".join(
        f'<TextRegion id="{side}" type="paragraph">'
        + "".join(_line(f"{side}{i}", x0, x0 + 1000, 300 + 100 * i, f"{side} line {i} of the text") for i in range(3))
        + "</TextRegion>"
        for side, x0 in (("left", 100), ("right", 1400))
    )
    path = tmp_path / "two_columns.xml"
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">'
        f'<Page imageFilename="two_columns.jpg" imageWidth="2500" imageHeight="1200">{columns}</Page></PcGts>',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def opening_page(tmp_path: Path) -> Path:
    """A 2500 x 1200 px two-page opening: four lines on the left page (x 100-1000), three on the right (x 1400-2400), a
    note in the left page's inner margin (x 1050-1150, read last) and a ruled line across both pages under their first
    lines. Its gutter is at x 1275, the middle of the widest band of its middle no line crosses (1150-1400)."""
    left = "".join(_line(f"l{i}", 100, 1000, 300 + 100 * i, f"left line {i}") for i in range(4))
    right = "".join(_line(f"r{i}", 1400, 2400, 300 + 100 * i, f"right line {i}") for i in range(3))
    path = tmp_path / "opening.xml"
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">'
        '<Page imageFilename="opening.jpg" imageWidth="2500" imageHeight="1200">'
        f'<TextRegion id="left" type="paragraph">{left}</TextRegion>'
        f'<TextRegion id="right" type="paragraph">{right}</TextRegion>'
        f'<TextRegion id="note" type="marginalia">{_line("n", 1050, 1150, 400, "No 12")}</TextRegion>'
        '<SeparatorRegion id="rule"><Coords points="100,330 2400,330 2400,332 100,332"/></SeparatorRegion>'
        "</Page></PcGts>",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def wide_and_narrow_body_page(tmp_path: Path) -> Path:
    """A 1700 x 900 px page: a wide body block of three lines and beside it a narrow one, too narrow to be cropped
    alone, so both are planned into one crop (as the two pages of a spread can be)."""
    blocks = "".join(
        f'<TextRegion id="{side}" type="paragraph">'
        + "".join(_line(f"{side}{i}", x0, x1, 300 + 100 * i, f"{side} line {i}") for i in range(3))
        + "</TextRegion>"
        for side, x0, x1 in (("wide", 100, 1100), ("narrow", 1200, 1500))
    )
    path = tmp_path / "wide_and_narrow.xml"
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">'
        f'<Page imageFilename="wide_and_narrow.jpg" imageWidth="1700" imageHeight="900">{blocks}</Page></PcGts>',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def one_line_page(tmp_path: Path) -> Path:
    """A 1000 x 1000 px page of a single line: no line pitch to measure, so it plans to no crop at all."""
    path = tmp_path / "one_line.xml"
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">'
        '<Page imageFilename="one_line.jpg" imageWidth="1000" imageHeight="1000">'
        '<TextRegion id="r" type="paragraph">'
        + _line("l", 100, 900, 500, "den fjerde dagen i maj")
        + "</TextRegion></Page></PcGts>",
        encoding="utf-8",
    )
    return path


TABLE_COLUMNS = (150, 300, 1100, 1400, 1650)  # the register's column borders, page px
TABLE_ROWS = 15  # a printed header row and 14 handwritten rows, 100 px apart from y 200


@pytest.fixture
def table_page(tmp_path: Path) -> Path:
    """A 1800 x 1800 px ruled register in Transkribus' table form: a printed header row (N:o, Namn, År, Ort) and 14
    handwritten rows of four cells, its five column borders ruled from the top of the table to its bottom."""
    header = ("N:o", "Namn", "År", "Ort")
    cells = []
    for row in range(TABLE_ROWS):
        top = 200 + 100 * row
        for column, (x0, x1) in enumerate(zip(TABLE_COLUMNS, TABLE_COLUMNS[1:], strict=False)):
            text = header[column] if row == 0 else f"r{row}c{column}"
            production = ' production="printed"' if row == 0 else ""
            cells.append(
                f'<TableCell id="c{row}_{column}" row="{row}" col="{column}"{production}>'
                f'<Coords points="{x0},{top} {x1},{top} {x1},{top + 100} {x0},{top + 100}"/>'
                + _line(f"l{row}_{column}", x0 + 20, x1 - 20, top + 72 + 3 * column, text)  # a row's baselines jitter
                + "</TableCell>"
            )
    bottom = 200 + 100 * TABLE_ROWS
    rules = "".join(
        f'<SeparatorRegion id="s{k}"><Coords points="{x - 1},200 {x + 1},200 {x + 1},{bottom} {x - 1},{bottom}"/>'
        "</SeparatorRegion>"
        for k, x in enumerate(TABLE_COLUMNS)
    )
    path = tmp_path / "register.xml"
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">'
        '<Page imageFilename="register.jpg" imageWidth="1800" imageHeight="1800">'
        f'<TableRegion id="register"><Coords points="150,200 1650,200 1650,{bottom} 150,{bottom}"/>'
        f"{''.join(cells)}</TableRegion>{rules}</Page></PcGts>",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def certificate_page(tmp_path: Path) -> Path:
    """A 900 x 1200 px printed certificate filled in by hand, as a form comes: a printed title and five rows of a
    printed label beside a handwritten field, every piece its own one-line region, each field underlined."""
    rows = (
        ("Att", "Anna Lisa Ersdotter"),
        ("född", "den 14 Mars 1832"),
        ("i", "Bro socken och Upsala län"),
        ("flyttar till", "Clara församling"),
        ("Nattvard", "den 3 Maj 1857"),
    )
    regions = [
        '<TextRegion id="title" type="heading" production="printed">'
        + _line("t", 300, 600, 140, "Flyttningsbetyg")
        + "</TextRegion>"
    ]
    rules = []
    for k, (label, field) in enumerate(rows):
        baseline = 260 + 80 * k
        regions.append(
            f'<TextRegion id="label{k}" production="printed">{_line(f"a{k}", 80, 200, baseline, label)}</TextRegion>'
            f'<TextRegion id="field{k}">{_line(f"f{k}", 220, 820, baseline, field)}</TextRegion>'
        )
        rules.append(
            f'<SeparatorRegion id="u{k}"><Coords points="215,{baseline + 22} 825,{baseline + 22} 825,{baseline + 24} '
            f'215,{baseline + 24}"/></SeparatorRegion>'
        )
    path = tmp_path / "certificate.xml"
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">'
        '<Page imageFilename="certificate.jpg" imageWidth="900" imageHeight="1200">'
        f"{''.join(regions)}{''.join(rules)}</Page></PcGts>",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def body_and_notes_crop(body_and_notes_page: Path) -> Crop:
    """The first training crop of ``body_and_notes_page``."""
    layout = load_xml_layout(body_and_notes_page)
    return crops_for_page(regions_of(layout), (layout.width, layout.height), CropSettings())[0]


@pytest.fixture
def page_style() -> PageStyle:
    """A page job's style: an archive of 1700, scanned in colour on cream paper with brown ink."""
    return PageStyle(
        provenance=Provenance(title="the archive", years="1700"),
        look=ScanLook(greyscale=False, paper="cream", ink="brown"),
    )


@pytest.fixture
def page_text() -> str:
    """A page job's running text: some 400 characters of a court record, refilled into a template's lines."""
    return "Anno sextonhundra och tjugo den femte dagen i maj kom for ratten bonden Olof Jonsson ifran Rodon " * 4


ARJEPLOGS = "GT-RA_arjeplogs_tingslag_20260220"
ARJEPLOGS_PAGES = Path(ARJEPLOGS, "Arjeplogs tingslag (1690-1700)", "Arjeplogs tingslag (1690-1700)", "page")
YELLOWED = ScanLook(greyscale=False, paper="yellowed", ink="dark brown")


class RecordMaker(Protocol):
    """``make_record``: ``crop``'s record as ``build_crops`` writes it, as the image ``file_name`` cut from ``page``."""

    def __call__(self, crop: Crop, file_name: str = ..., page: str = ...) -> CropRecord: ...


@pytest.fixture
def make_record() -> RecordMaker:
    """Records a crop as build_crops does, as a crop of a page of Arjeplogs tingslag (1690-1700) scanned yellowed with
    dark brown ink: ``make_record(crop, file_name="crop.jpg", page="0001.xml")``."""

    def make(crop: Crop, file_name: str = "crop.jpg", page: str = "0001.xml") -> CropRecord:
        return crop_record(crop, file_name=file_name, collection=ARJEPLOGS, page=ARJEPLOGS_PAGES / page, look=YELLOWED)

    return make


TINY_FLUX3 = "hf-internal-testing/tiny-flux3-image-modular"
TINY_FLUX3_REVISION = "9130f96f124d6c6fdd1beab63cfb6d2f1c8658a1"  # pinned: a changed fixture must not change results


# 8x downsampling like FLUX.3's own VAE: the Hub test model's downsamples 2x, which made a bucket-sized crop some
# 59,000 tokens and its drawing minutes of CPU attention in the transformer and the VAE
TINY_FLUX3_VAE = {
    "block_out_channels": [16, 16, 16, 16],
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
}


@pytest.fixture(scope="session")
def tiny_flux3(tiny_flux3_snapshot: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """diffusers' tiny FLUX.3 test model (a few MB, downloaded once into the Hugging Face cache), copied with a tiny VAE
    that downsamples 8x as FLUX.3's does, so a crop is drawn from as many tokens as the real model draws it from."""
    snapshot = tiny_flux3_snapshot
    repo = tmp_path_factory.mktemp("flux3")
    shutil.copytree(snapshot, repo, ignore=shutil.ignore_patterns("vae"), dirs_exist_ok=True)
    torch.manual_seed(0)
    vae = AutoencoderKLFlux2.from_config(
        {**AutoencoderKLFlux2.load_config(snapshot, subfolder="vae"), **TINY_FLUX3_VAE}
    )
    vae.save_pretrained(repo / "vae")
    return repo


# 8x downsampling like the real Qwen-Image VAE, so a token is 16 px as in diffu; latent statistics that are not the
# identity, so normalization is exercised
TINY_QWEN_IMAGE_VAE = {
    "base_dim": 16,
    "z_dim": 4,
    "dim_mult": [1, 1, 1, 1],
    "num_res_blocks": 1,
    "temperal_downsample": [False, True, True],
    "latents_mean": [0.5, -0.5, 0.25, 0.0],
    "latents_std": [2.0, 1.0, 0.5, 1.5],
}


@pytest.fixture(scope="session")
def tiny_flux3_snapshot() -> Path:
    """diffusers' tiny FLUX.3 test model as downloaded: its tiny Qwen3-VL text encoder and processor among it."""
    return Path(snapshot_download(TINY_FLUX3, revision=TINY_FLUX3_REVISION))


@pytest.fixture(scope="session")
def tiny_diffu_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("diffu")


@pytest.fixture(scope="session")
def tiny_diffu_config(tiny_flux3_snapshot: Path, tiny_diffu_dir: Path) -> DiffuConfig:
    """A tiny diffu: 2 heads of 8, one per-mode and one joint block, tiny FLUX.3's Qwen3-VL at its middle layer, a
    4-channel VAE kept in the repo."""
    text_encoder = TextEncoder(
        repo=str(tiny_flux3_snapshot),
        revision=None,
        encoder_subfolder="text_encoder",
        processor_subfolder="tokenizer",
        layers=(1,),
        hidden_size=16,
    )
    return DiffuConfig(
        size=DiffuSize(num_attention_heads=2, attention_head_dim=8, mode_layers=1, joint_layers=1),
        text_encoder=text_encoder,
        vae=Vae(repo=str(tiny_diffu_dir), revision=None, subfolder="vae", latent_channels=4),
    )


@pytest.fixture(scope="session")
def tiny_diffu(tiny_diffu_config: DiffuConfig, tiny_diffu_dir: Path) -> Path:
    """The tiny diffu as a model repo. Its transformer keeps its random (not adaLN-Zero) weights, so a prompt or a
    reference changes what it draws."""
    torch.manual_seed(0)
    AutoencoderKLQwenImage(**TINY_QWEN_IMAGE_VAE).save_pretrained(tiny_diffu_dir / "vae")
    transformer = Flux3Transformer2DModel.from_config(tiny_diffu_config.transformer_config())
    write_model_repo(transformer, tiny_diffu_config, tiny_diffu_dir)
    return tiny_diffu_dir


@pytest.fixture(scope="session")
def tiny_diffu_glyph(
    tiny_diffu_config: DiffuConfig, tiny_diffu: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """The tiny diffu as a glyph model (``diffu-init --glyph``), with the tiny diffu's VAE: its 16 glyph channels
    beside its 16 latent channels, its transformer's weights random (the glyph map's among them), so the map changes
    what it draws."""
    repo = tmp_path_factory.mktemp("diffu_glyph")
    config = tiny_diffu_config.model_copy(update={"glyph": True})
    torch.manual_seed(1)
    write_model_repo(Flux3Transformer2DModel.from_config(config.transformer_config()), config, repo)
    return repo


MOSS_BASE = "OpenMOSS-Team/MOSS-Image-Base"
MOSS_BASE_REVISION = "85975793d8b3d432f1b6e27f352abaa4c2a4c0d3"  # pinned: the tokenizer and chat template MOSS reads

# Qwen3.5 at width 16, a linear-attention layer then a full-attention one, its vision tower (never run) as small; the
# vocabulary in full, as MOSS's own tokenizer writes every id of it
TINY_QWEN3_5 = {
    "text_config": {
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
    },
    "vision_config": {
        "depth": 1,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_heads": 2,
        "out_hidden_size": 16,
        "num_position_embeddings": 64,
    },
}
# three blocks of width 32 over the tiny VAE's 4 channels packed 2x2 and the tiny Qwen3.5's 16
TINY_MOSS_DIT = {
    "dim": 32,
    "ffn_dim": 64,
    "num_heads": 2,
    "num_layers": 3,
    "text_dim": 16,
    "freq_dim": 16,
    "latent_channels": 16,
}


@pytest.fixture(scope="session")
def tiny_moss(tiny_flux3_snapshot: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny MOSS-Image repo, diffu-moss-init's, on a tiny Base snapshot: MOSS's own tokenizer (downloaded once), a
    tiny Qwen3.5 text encoder, a VAE of FLUX.2's class that downsamples 8x as MOSS's does (tiny FLUX.3's) and a tiny
    DiT whose unconditional pass skips its middle block, all random weights, so a prompt changes what it draws."""
    base = tmp_path_factory.mktemp("moss_base")
    snapshot = Path(snapshot_download(MOSS_BASE, revision=MOSS_BASE_REVISION, allow_patterns=["tokenizer/*"]))
    shutil.copytree(snapshot / "tokenizer", base / "tokenizer")
    torch.manual_seed(0)
    Qwen3_5Model(Qwen3_5Config(**TINY_QWEN3_5)).save_pretrained(base / "text_encoder")
    vae_config = {**AutoencoderKLFlux2.load_config(tiny_flux3_snapshot, subfolder="vae"), **TINY_FLUX3_VAE}
    AutoencoderKLFlux2.from_config(vae_config).save_pretrained(base / "vae")
    MossTransformer(**TINY_MOSS_DIT).save_pretrained(base / "transformer")
    repo = tmp_path_factory.mktemp("moss")
    write_moss_repo(repo, base=base, uncond_skip_layers=(1,))
    return repo


TINY_PAIRS = 11  # a style-pair set of 11 targets, each pair's reference the next target's crop
LEAKING_PAIR = 3  # its reference (pair 4's crop) also holds its first line: left out, so 10 pairs are kept
PAIR_TARGET_SIZE = (64, 32)  # px: a 4 x 2 latent grid
PAIR_REFERENCE_SIZE = (32, 32)  # px: shrunk, as a style pair's reference is


def _pair_texts(k: int) -> tuple[str, str]:
    """The two lines of pair ``k``'s target crop."""
    first = f"Anno 17{k:02d} den {k + 1} Maj" if k != LEAKING_PAIR + 1 else _pair_texts(LEAKING_PAIR)[0]
    return first, f"på Rådstugan, nummer {k}"


def _described_caption(texts: tuple[str, ...]) -> str:
    """A target's caption as style_pairs writes a described reference pair (v3d): its lines drawn in the hand, ink and
    paper of <ref_image_0>, described as measured, each line in the reference's hand."""
    lines = [
        {
            "id": f"Sv_Text_{k}",
            "from": "target",
            "src_bbox": None,
            "tgt_bbox": [100 + 450 * (k - 1), 50, 450 + 450 * (k - 1), 950],
            "desc": f'Handwritten line of text reading "{text}", right-slanted cursive in the reference\'s hand, in '
            "brown ink.",
        }
        for k, text in enumerate(texts, start=1)
    ]
    paper = {
        "id": "paper_1",
        "from": "ref_image_0",
        "src_bbox": [0, 0, 1000, 1000],
        "tgt_bbox": [0, 0, 1000, 1000],
        "desc": "Aged cream paper (#D8C8A8) of a bound archival volume.",
    }
    cited = " ".join(f"<Sv_Text_{k}>" for k in range(1, len(texts) + 1))
    paragraph = (
        "Create a flat, evenly lit archival scan of a new handwritten page in the hand, ink and paper of <ref_image_0>: "
        "right-slanted cursive, about 30 degrees, with fine pen strokes and small, closely packed letters, in brown ink "
        f"(#5A4632) on cream paper (#D8C8A8). The paper <paper_1> carries {len(texts)} lines of cursive handwriting in "
        f"brown ink; the main text runs from top to bottom: {cited}."
    )
    return f"{paragraph} {json.dumps([paper, *lines], ensure_ascii=False)}"


def _noise_image(size: tuple[int, int], seed: int) -> Image.Image:
    pixels = torch.rand(size[1], size[0], 3, generator=torch.Generator().manual_seed(seed))
    return Image.fromarray((pixels * 255).to(torch.uint8).numpy())


@pytest.fixture(scope="session")
def tiny_pairs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny style-pair set as diffu_page.data.style_pairs writes a described one: ``train.jsonl`` (the image-edit
    manifest) of 11 pairs ``vol_00`` ... ``vol_10``, its targets (64 x 32 px, two lines each: pair ``k``'s "Anno 17kk
    den k+1 Maj" and "på Rådstugan, nummer k", in brown ink #5A4632 on cream paper by its caption) and its references
    (32 x 32 px), noise images. Pair ``k``'s reference is pair ``k + 1``'s crop; pair 3's reference, pair 4's crop,
    holds pair 3's first line."""
    root = tmp_path_factory.mktemp("pairs")
    (root / "targets").mkdir()
    (root / "refs").mkdir()
    rows = []
    for k in range(TINY_PAIRS):
        crop, reference = f"vol_{k:02d}", f"vol_{(k + 1) % TINY_PAIRS:02d}"
        _noise_image(PAIR_TARGET_SIZE, k).save(root / "targets" / f"{crop}.png")
        _noise_image(PAIR_REFERENCE_SIZE, 100 + k).save(root / "refs" / f"{crop}.png")
        media = [
            {"role": "target", "file_name": f"targets/{crop}.png"},
            {"role": "context", "file_name": f"refs/{reference}.png"},
        ]
        row = {"id": crop, "prompt": _described_caption(_pair_texts(k)), "media": media}
        rows.append(json.dumps({**row, "metadata": {"row_index": k, "reference": f"{reference}.png"}}))
    (root / "train.jsonl").write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def tiny_pair_cache(tiny_moss: Path, tiny_pairs: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The tiny pairs cached by the tiny MOSS's encoders, each kept pair in the form the mix gives it."""
    cache = tmp_path_factory.mktemp("pair_cache") / "cache"
    write_cache(tiny_moss, tiny_pairs / "train.jsonl", cache, CacheSettings(device="cpu"))
    return cache


@pytest.fixture(scope="session")
def tiny_moss_dit(tiny_moss: Path) -> Path:
    """The tiny MOSS's DiT folder (its Base snapshot's ``transformer/``)."""
    return Path(ModelIndex.read(tiny_moss).source("transformer").repo)


@pytest.fixture(scope="session")
def tiny_reference_run(tiny_moss_dit: Path, tiny_pair_cache: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A reference training of the tiny MOSS's DiT on the tiny pairs' cache: 5 steps of 2 samples (one pass over its
    10 entries), a checkpoint at the end. Its run directory."""
    run = tmp_path_factory.mktemp("reference_run") / "run"
    train(tiny_moss_dit, tiny_pair_cache, run, TrainSettings(steps=5, save_every=5, val_every=5, device="cpu"))
    return run


@pytest.fixture(scope="session")
def tiny_reference_moss(
    tiny_moss_dit: Path, tiny_reference_run: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """The tiny reference training's last checkpoint merged into the tiny MOSS's DiT (diffu-moss-export-ref), as a MOSS
    repo on the tiny Base snapshot (diffu-moss-init --transformer): a model that takes a reference image."""
    export = tmp_path_factory.mktemp("reference_export") / "export"
    export_merged(tiny_moss_dit, tiny_reference_run / "checkpoints" / "step_000005", export)
    repo = tmp_path_factory.mktemp("reference_moss")
    write_moss_repo(repo, base=tiny_moss_dit.parent, transformer=export, uncond_skip_layers=(1,))
    return repo


class NemoLoraWriter(Protocol):
    """``write_nemo_lora``: a NeMo LoRA of one ``module``, down weights ``a`` and up weights ``b``, its ``model/`` dir."""

    def __call__(self, module: str, a: torch.Tensor, b: torch.Tensor, *, alpha: int) -> Path: ...


@pytest.fixture
def write_nemo_lora(tmp_path: Path) -> NemoLoraWriter:
    """Writes a NeMo AutoModel LoRA checkpoint (PEFT keys under ``base_model.model.``) for one module of a transformer:
    ``write_nemo_lora(module, a, b, alpha=...)`` returns its ``model/`` dir."""
    written: list[Path] = []

    def write(module: str, a: torch.Tensor, b: torch.Tensor, *, alpha: int) -> Path:
        lora_dir = tmp_path / f"lora_{len(written)}"
        lora_dir.mkdir()
        weights = {f"base_model.model.{module}.lora_A.weight": a, f"base_model.model.{module}.lora_B.weight": b}
        save_file(weights, lora_dir / "adapter_model.safetensors")
        config = {"r": a.shape[0], "lora_alpha": alpha, "target_modules": [module], "peft_type": "LORA"}
        (lora_dir / "adapter_config.json").write_text(json.dumps(config))
        written.append(lora_dir)
        return lora_dir

    return write
