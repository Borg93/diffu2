"""A run on a local Ray cluster (CPU): each page drawn whole by an actor holding the tiny MOSS-Image model that takes a
sample image, judged by a tiny TrOCR and committed to the run's Lance table; a page that fails kept with its error; a
run into a table resuming it; and everything that would stop a run checked before it starts."""

from __future__ import annotations

import io
import json
import os
import re
from collections.abc import Iterator
from pathlib import Path
from xml.etree import ElementTree

import lance
import pytest
import ray
import torch
from PIL import Image
from typer.testing import CliRunner

from diffu_page.captions.look import ScanLook
from diffu_page.captions.provenance import Provenance
from diffu_page.captions.restyle import ReferenceCaption
from diffu_page.draw.jobs import PageJob, crop_jobs, single_pages
from diffu_page.draw.pipeline import DrawSettings
from diffu_page.draw.plan import PageStyle, Reference
from diffu_page.page.layout import Layout, TextLine
from diffu_runner.export import export_pages
from diffu_runner.run import RUNS, RunConfig, RunMeta, RunRefused, app, read_draw_settings, read_pages, run
from diffu_runner.stages import DEVICE_FAULT_EXIT, Attention, DrawActor, DrawConfig, JudgeModel
from diffu_runner.table import FILE_FORMAT, PAGES, PageRow, page_table

STYLE = PageStyle(
    provenance=Provenance(title="the archive", years="1700"), look=ScanLook(greyscale=False, paper="cream", ink="brown")
)
TEXT = "Anno sextonhundra och tjugo den femte dagen i maj kom for ratten bonden Olof Jonsson ifran Rodon " * 4
PAGE_PIXELS = 250_000  # --page-megapixels 0.25, its least
SETTINGS = DrawSettings(steps=2, device="cpu")


@pytest.fixture(scope="module")
def local_ray() -> Iterator[None]:
    """A local Ray of one CPU, which the pages are planned on: its actors keep none (``cpus_per_actor`` 0). A run that
    should be refused runs here too, so that one which is not stays on this one CPU."""
    ray.init(address="local", num_cpus=1, include_dashboard=False, log_to_driver=False)
    yield
    ray.shutdown()


@pytest.fixture
def sample(tmp_path: Path) -> Reference:
    """A sample image to draw a page in the hand of, in the sample form of MOSS-Image's page pairs."""
    path = tmp_path / "sample.png"
    Image.new("RGB", (96, 128), (214, 200, 168)).save(path)
    return Reference(image=path, caption=ReferenceCaption.SAMPLE)


@pytest.fixture
def pages(opening_page: Path, unsplit_spread_page: Path, sample: Reference) -> list[PageJob]:
    """A run's three page jobs, each drawn from the sample: a two-page opening (``op``), a spread whose pages its
    lines cannot split (``spread``), which planning refuses, and a page given as a layout of boxes alone (``boxes``)."""
    boxes = Layout(
        width=1000,
        height=1200,
        lines=[TextLine(box=(100, 200 + 100 * i, 900, 270 + 100 * i), baseline_y=270 + 100 * i) for i in range(4)],
    )
    return [
        PageJob(id="op", template=opening_page, text=TEXT, style=STYLE, reference=sample, seed=3),
        PageJob(id="spread", template=unsplit_spread_page, text=TEXT, style=STYLE, reference=sample),
        PageJob(id="boxes", layout=boxes, text=TEXT, style=STYLE, reference=sample),
    ]


@pytest.fixture
def draw(tiny_reference_moss: Path, tiny_htr: Path) -> DrawConfig:
    return DrawConfig(
        model=tiny_reference_moss, settings=SETTINGS, page_pixels=PAGE_PIXELS, judge=JudgeModel(model=str(tiny_htr))
    )


def _on_the_cpu(draw: DrawConfig) -> RunConfig:
    return RunConfig(draw=draw, gpus=0, cpus_per_actor=0)


def _write_jobs(path: Path, pages: list[PageJob]) -> Path:
    path.write_text("".join(f"{page.model_dump_json()}\n" for page in pages), encoding="utf-8")
    return path


@pytest.fixture
def jobs(pages: list[PageJob], tmp_path: Path) -> Path:
    """The run file of ``pages``."""
    return _write_jobs(tmp_path / "run.jsonl", pages)


def _rows(out: Path) -> dict[str, dict]:
    """The table's rows by id, each id held once."""
    rows = lance.dataset(str(out / PAGES)).to_table(blob_handling="all_binary").to_pylist()
    by_id = {row["id"]: row for row in rows}
    assert len(by_id) == len(rows), f"ids held twice: {sorted(row['id'] for row in rows)}"
    return by_id


def _whole(page: PageJob) -> PageJob:
    return page.model_copy(update={"page_pixels": PAGE_PIXELS})


def _held(page_id: str, *, image: bytes | None = None, error: str | None = None, xml: str | None = None) -> PageRow:
    """A row as an earlier run left it in a table."""
    return PageRow(
        id=page_id,
        image=image,
        alto_xml=None if xml is None else f"<alto {xml}/>",
        page_xml=None if xml is None else f"<page {xml}/>",
        template="page.xml",
        job="{}",
        model="moss",
        run="earlier",
        seconds=1.0,
        error=error,
    )


def test_diffu_run_draws_each_page_whole_into_the_table_and_keeps_a_page_that_fails_with_its_error(
    local_ray: None, pages: list[PageJob], jobs: Path, tiny_reference_moss: Path, tiny_htr: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    draw_settings = tmp_path / "draw.json"
    draw_settings.write_text(json.dumps({"steps": 2, "device": "cpu"}), encoding="utf-8")
    options = ["--jobs", str(jobs), "--out", str(out), "--model", str(tiny_reference_moss)]
    options += ["--draw-settings", str(draw_settings), "--gpus", "0", "--cpus-per-actor", "0"]
    options += ["--page-megapixels", "0.25", "--htr-model", str(tiny_htr), "--htr-revision", ""]

    result = CliRunner().invoke(app, [*options, "--commit-every", "2"])

    assert result.exit_code == 0, result.output
    rows = _rows(out)
    [meta_file] = (out / RUNS).iterdir()
    meta = RunMeta.model_validate_json(meta_file.read_text(encoding="utf-8"))
    assert sorted(rows) == ["boxes", "op_left", "op_right", "spread"]
    assert (meta.drawn, meta.failed, meta.in_table, meta.retried) == (3, 1, 0, 0)
    assert [rows[page_id]["template"] for page_id in ("op_left", "boxes")] == [str(pages[0].template), None]
    for single in [*single_pages(pages[0]), pages[2]]:
        [job] = crop_jobs(_whole(single), 0)
        lines = [job.plan.lines[k].text for k in job.plan.reading_order]
        row = rows[str(single.id)]
        assert row["error"] is None
        assert Image.open(io.BytesIO(row["image"])).size == (job.plan.width, job.plan.height)
        assert (row["width"], row["height"], row["seed"]) == (job.plan.width, job.plan.height, job.seed)
        assert (row["text"], [line["text"] for line in row["lines"]]) == ("\n".join(lines), lines)
        assert ElementTree.fromstring(row["alto_xml"]).tag == "{http://www.loc.gov/standards/alto/ns-v4#}alto"
        page_xml = ElementTree.fromstring(row["page_xml"]).find("{*}Page")
        assert page_xml is not None
        assert page_xml.get("imageFilename") == f"{single.id}.jpg"
        assert PageJob.model_validate_json(row["job"]) == _whole(single)
        assert (row["model"], row["run"]) == (str(tiny_reference_moss), meta.run)
    assert rows["spread"]["image"] is None
    assert rows["spread"]["error"].startswith(
        f"ValueError: {pages[1].template} is a spread whose pages cannot be split"
    )
    dataset = lance.dataset(str(out / PAGES))
    assert dataset.data_storage_version == FILE_FORMAT
    assert dataset.schema.field("image").type.extension_name == "lance.blob.v2"


def test_run_into_a_table_draws_only_the_pages_it_lacks_and_those_that_failed(
    local_ray: None, pages: list[PageJob], jobs: Path, draw: DrawConfig, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    held = [_held("op_left", image=b"drawn"), _held("spread", error="ValueError: x")]
    lance.write_dataset(page_table(held), str(out / PAGES), data_storage_version=FILE_FORMAT)

    meta = run(pages, _on_the_cpu(draw), out, jobs=jobs)

    rows = _rows(out)
    assert sorted(rows) == ["boxes", "op_left", "op_right", "spread"]
    assert (rows["op_left"]["run"], rows["op_left"]["image"]) == ("earlier", b"drawn")
    assert (rows["op_right"]["run"], rows["op_right"]["error"]) == (meta.run, None)
    assert rows["spread"]["run"] == meta.run
    assert rows["spread"]["error"].startswith("ValueError: ")
    assert (meta.drawn, meta.failed, meta.in_table, meta.retried) == (2, 1, 1, 1)


@pytest.fixture(scope="module")
def tiny_moss_draw(tiny_moss: Path, tiny_htr: Path) -> DrawConfig:
    """Drawing with the tiny text-only MOSS-Image model on the CPU."""
    return DrawConfig(model=tiny_moss, settings=SETTINGS, judge=JudgeModel(model=str(tiny_htr)))


@pytest.fixture(scope="module")
def actor(tiny_moss_draw: DrawConfig) -> DrawActor:
    """A draw actor off Ray, holding the tiny text-only MOSS-Image model."""
    return DrawActor(tiny_moss_draw, run="r")


@pytest.fixture
def planned(opening_page: Path) -> dict[str, list[str]]:
    """A batch of one planned page, as the plan stage hands it to an actor: the opening's left page, planned whole."""
    page = _whole(PageJob(id="p", template=opening_page, text=TEXT, style=STYLE, side="left"))
    [job] = crop_jobs(page, 0)
    return {"page": [page.model_dump_json()], "job": [job.model_dump_json()], "error": [""]}


class ProcessExit(Exception):
    """What the patched ``os._exit`` raises instead of ending the test's process."""


def test_draw_actor_exits_its_process_when_the_device_fails(
    actor: DrawActor, planned: dict[str, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise torch.AcceleratorError("CUDA error: an illegal memory access was encountered")

    def exit_process(status: int) -> None:
        raise ProcessExit(status)

    monkeypatch.setattr("diffu_runner.stages.draw_crop", fail)
    monkeypatch.setattr(os, "_exit", exit_process)

    with pytest.raises(ProcessExit) as exited:
        actor(planned)

    assert exited.value.args == (DEVICE_FAULT_EXIT,)


def test_draw_actor_keeps_a_page_that_runs_out_of_memory_as_its_row_with_the_error(
    actor: DrawActor, planned: dict[str, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    def run_out(*args: object, **kwargs: object) -> None:
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr("diffu_runner.stages.draw_crop", run_out)

    [row] = actor(planned).to_pylist()

    assert (row["id"], row["image"]) == ("p", None)
    assert row["error"] == "OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"


def test_draw_actor_draws_on_the_attention_kernel_it_is_given(
    tiny_moss_draw: DrawConfig, planned: dict[str, list[str]]
) -> None:
    efficient_only = DrawActor(tiny_moss_draw, run="r", attention=Attention.EFFICIENT.value)  # a kernel no CPU has

    [row] = efficient_only(planned).to_pylist()

    assert row["image"] is None
    assert row["error"].startswith("RuntimeError: No viable backend for scaled_dot_product_attention was found")


@pytest.mark.parametrize(
    ("gpus", "actors_per_gpu", "actors", "share"),
    [pytest.param(4, 1, 4, 1.0, id="one-a-gpu"), pytest.param(4, 2, 8, 0.5, id="two-a-gpu")],
)
def test_run_config_draws_with_actors_per_gpu_on_every_gpu(
    tiny_moss: Path, gpus: int, actors_per_gpu: int, actors: int, share: float
) -> None:
    draw = DrawConfig(model=tiny_moss, settings=DrawSettings(device="cuda"))

    config = RunConfig(draw=draw, gpus=gpus, actors_per_gpu=actors_per_gpu)

    assert (config.actors, config.gpus_per_actor) == (actors, share)


@pytest.mark.parametrize(
    ("device", "gpus"), [pytest.param("cuda", 0, id="cuda-without-gpus"), pytest.param("cpu", 2, id="cpu-with-gpus")]
)
def test_run_config_refuses_a_device_its_gpus_do_not_fit(tiny_moss: Path, device: str, gpus: int) -> None:
    draw = DrawConfig(model=tiny_moss, settings=DrawSettings(device=device))

    with pytest.raises(ValueError, match="draw on the CPU with 0 GPUs, else on GPUs"):
        RunConfig(draw=draw, gpus=gpus)


def test_read_pages_refuses_a_line_that_is_not_a_page_job(tmp_path: Path) -> None:
    jobs = tmp_path / "run.jsonl"
    jobs.write_text('\n{"id": "a"}\n', encoding="utf-8")

    with pytest.raises(RunRefused, match="run.jsonl:2 is not a page job: style: Field required"):
        read_pages(jobs)


@pytest.mark.parametrize(
    ("ids", "error"),
    [
        pytest.param(["a", None], "run.jsonl:2 has no id", id="no-id"),
        pytest.param(["a", "b", "a"], "an id names one page, but 1 name several: a", id="shared-id"),
    ],
)
def test_read_pages_refuses_pages_it_could_not_resume_by_id(
    tmp_path: Path, opening_page: Path, ids: list[str | None], error: str
) -> None:
    lines = [PageJob(id=page_id, template=opening_page, text=TEXT, style=STYLE) for page_id in ids]
    jobs = _write_jobs(tmp_path / "run.jsonl", lines)

    with pytest.raises(RunRefused, match=error):
        read_pages(jobs)


def test_draw_config_holds_its_paths_absolute_for_actors_in_other_working_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "htr").mkdir()

    draw = DrawConfig(
        model=Path("moss"), loras={"hand": Path("lora")}, settings=SETTINGS, judge=JudgeModel(model="htr")
    )
    from_the_hub = JudgeModel(model="org/htr")

    assert (draw.model, draw.loras, draw.judge.model) == (
        tmp_path / "moss",
        {"hand": tmp_path / "lora"},
        str(tmp_path / "htr"),
    )
    assert from_the_hub.model == "org/htr"


def test_read_draw_settings_refuses_a_key_that_is_no_draw_setting(tmp_path: Path) -> None:
    draw_settings = tmp_path / "draw.json"
    draw_settings.write_text(json.dumps({"steps": 30, "step": 20, "guidance": 2.0}), encoding="utf-8")

    with pytest.raises(RunRefused, match="draw.json names draw settings that do not exist: guidance, step$"):
        read_draw_settings(draw_settings)


def test_run_refuses_repair_passes(
    local_ray: None, pages: list[PageJob], jobs: Path, draw: DrawConfig, tmp_path: Path
) -> None:
    repairing = draw.model_copy(update={"settings": SETTINGS.model_copy(update={"repair_passes": 1})})

    with pytest.raises(RunRefused, match="repair_passes is 1: a run draws each page once"):
        run(pages, _on_the_cpu(repairing), tmp_path / "out", jobs=jobs)


def test_run_refuses_a_sample_image_for_a_model_that_takes_none(
    local_ray: None, pages: list[PageJob], jobs: Path, tiny_moss_draw: DrawConfig, tmp_path: Path
) -> None:
    config = _on_the_cpu(tiny_moss_draw)

    with pytest.raises(RunRefused, match=re.escape(f"(the first: op), but {tiny_moss_draw.model} takes none")):
        run(pages, config, tmp_path / "out", jobs=jobs)


def test_run_refuses_a_model_directory_written_on_another_machine(
    local_ray: None, pages: list[PageJob], jobs: Path, draw: DrawConfig, tmp_path: Path
) -> None:
    model = tmp_path / "moss"
    model.mkdir()
    index = json.loads((draw.model / "modular_model_index.json").read_text(encoding="utf-8"))
    index["vae"][2]["pretrained_model_name_or_path"] = "/elsewhere/MOSS-Image-Base"
    (model / "modular_model_index.json").write_text(json.dumps(index), encoding="utf-8")
    config = _on_the_cpu(draw.model_copy(update={"model": model}))

    with pytest.raises(RunRefused, match=re.escape("not on this machine (vae at /elsewhere/MOSS-Image-Base/vae)")):
        run(pages, config, tmp_path / "out", jobs=jobs)


def test_run_refuses_a_judge_that_is_not_staged(
    local_ray: None, pages: list[PageJob], jobs: Path, draw: DrawConfig, tmp_path: Path
) -> None:
    config = _on_the_cpu(draw.model_copy(update={"judge": JudgeModel(model="nobody/no-such-htr")}))

    with pytest.raises(
        RunRefused, match="the HTR model nobody/no-such-htr at revision .* is neither a local directory"
    ):
        run(pages, config, tmp_path / "out", jobs=jobs)


def test_run_refuses_a_run_file_that_names_a_relative_path(
    local_ray: None, pages: list[PageJob], jobs: Path, draw: DrawConfig, tmp_path: Path
) -> None:
    relative = [pages[0].model_copy(update={"template": Path("templates/page.xml")}), pages[1]]

    with pytest.raises(RunRefused, match="1 templates or reference images are named by a relative path"):
        run(relative, _on_the_cpu(draw), tmp_path / "out", jobs=jobs)


def test_run_refuses_pages_whose_files_are_not_here(
    local_ray: None, pages: list[PageJob], jobs: Path, draw: DrawConfig, tmp_path: Path
) -> None:
    gone = tmp_path / "gone.xml"
    moved = [pages[0].model_copy(update={"template": gone}), pages[1]]

    with pytest.raises(RunRefused, match="1 templates or reference images are not here") as refused:
        run(moved, _on_the_cpu(draw), tmp_path / "out", jobs=jobs)

    assert str(refused.value).endswith(f": {gone}")


def test_run_refuses_an_out_whose_pages_were_drawn_otherwise(
    local_ray: None, pages: list[PageJob], jobs: Path, draw: DrawConfig, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    (out / RUNS).mkdir(parents=True)
    other = _on_the_cpu(draw.model_copy(update={"settings": SETTINGS.model_copy(update={"steps": 3})}))
    earlier = RunMeta(run="earlier", config=other, jobs="run.jsonl", pages=2, in_table=0, retried=0, started="")
    (out / RUNS / "earlier.json").write_text(earlier.model_dump_json(), encoding="utf-8")

    with pytest.raises(RunRefused, match=re.escape(f"{out} holds pages drawn otherwise (run earlier")):
        run(pages, _on_the_cpu(draw), out, jobs=jobs)


@pytest.mark.parametrize(
    ("ids", "written"), [pytest.param((), ["a", "b"], id="every-drawn-page"), pytest.param(("b",), ["b"], id="by-id")]
)
def test_export_pages_writes_each_drawn_page_as_its_image_and_labels(
    tmp_path: Path, ids: tuple[str, ...], written: list[str]
) -> None:
    held = [_held("a", image=b"jpeg a", xml="a"), _held("b", image=b"jpeg b", xml="b"), _held("c", error="ValueError")]
    lance.write_dataset(page_table(held), str(tmp_path / PAGES), data_storage_version=FILE_FORMAT)

    exported = export_pages(tmp_path / PAGES, tmp_path / "files", ids=ids)

    expected: dict[str, bytes] = {}
    for page_id in written:
        expected |= {
            f"{page_id}.jpg": f"jpeg {page_id}".encode(),
            f"{page_id}.alto.xml": f"<alto {page_id}/>".encode(),
            f"{page_id}.page.xml": f"<page {page_id}/>".encode(),
        }
    assert exported == written
    assert {path.name: path.read_bytes() for path in (tmp_path / "files").iterdir()} == expected
