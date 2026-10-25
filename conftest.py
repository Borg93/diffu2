"""Fixtures shared by every workspace member's tests: PAGE pages to plan and draw, a crop and its training record,
diffusers' tiny FLUX.3 test model (with a VAE that downsamples as FLUX.3's does), and a tiny diffu model repo (a tiny
Qwen3.5, a tiny Qwen-Image VAE and a tiny transformer, all random weights)."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from diffusers import AutoencoderKLFlux2, AutoencoderKLQwenImage, Flux3Transformer2DModel
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

from diffu.configs import DiffuConfig, DiffuSize, TextEncoder, Vae
from diffu.init import write_model_repo
from diffu_page.captions.look import ScanLook
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


ARJEPLOGS = "GT-RA_arjeplogs_tingslag_20260220"
ARJEPLOGS_PAGES = Path(ARJEPLOGS, "Arjeplogs tingslag (1690-1700)", "Arjeplogs tingslag (1690-1700)", "page")
YELLOWED = ScanLook(greyscale=False, paper="yellowed", ink="dark brown")

type RecordMaker = Callable[..., CropRecord]


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


type NemoLoraWriter = Callable[..., Path]


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
