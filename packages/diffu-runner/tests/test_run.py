"""A run's Ray Data stages on a local Ray cluster (CPU): pages planned into exactly their crop jobs, crops drawn by an
actor holding diffusers' tiny FLUX.3 model, a crop that fails kept with its error."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pytest
import ray
from PIL import Image

from diffu_gen.generator import GeneratorSettings
from diffu_page.captions.look import ScanLook
from diffu_page.captions.prompt import LineRecord
from diffu_page.captions.provenance import Provenance
from diffu_page.jobs import CropJob, PageJob, crop_jobs
from diffu_page.plan import CropPlan, PageStyle
from diffu_runner.run import RunConfig, draw_dataset, plan_dataset

STYLE = PageStyle(
    provenance=Provenance(title="the archive", years="1700"), look=ScanLook(greyscale=False, paper="cream", ink="brown")
)
TEXT = "Anno sextonhundra och tjugo den femte dagen i maj kom for ratten bonden Olof Jonsson ifran Rodon " * 4


@pytest.fixture(scope="module")
def local_ray() -> Iterator[None]:
    ray.init(address="local", num_cpus=2, include_dashboard=False, log_to_driver=False)
    yield
    ray.shutdown()


def _small_job(name: str, reference: Path | None = None) -> CropJob:
    plan = CropPlan(
        rect=(0, 0, 64, 32),
        width=64,
        height=32,
        caption="A handwritten page.",
        lines=[LineRecord(id="Sv_Text_1", text="Göteborgs poliskammare", role="", bbox=[0, 0, 1000, 1000])],
        reading_order=[0],
    )
    return CropJob(name=name, template=Path("page.xml"), plan=plan, seed=0, reference=reference)


def test_plan_dataset_plans_every_page_into_its_crop_jobs(local_ray: None, body_and_notes_page: Path) -> None:
    page = PageJob(template=body_and_notes_page, text=TEXT, style=STYLE)

    rows = plan_dataset([page, page]).take_all()

    planned = sorted((CropJob.model_validate_json(row["job"]) for row in rows), key=lambda job: job.name)
    assert planned == crop_jobs(page, 0) + crop_jobs(page, 1)


def test_draw_dataset_draws_every_crop_and_keeps_a_failed_one_with_its_error(
    local_ray: None, tiny_model: Path, tmp_path: Path
) -> None:
    jobs = [_small_job("drawn"), _small_job("failed", reference=tmp_path / "missing.jpg")]
    config = RunConfig(
        model=tiny_model,
        loras={},
        generator=GeneratorSettings(steps=2, text_layers=(1,), device="cpu"),
        gpus_per_drawer=0,
    )

    rows = draw_dataset(ray.data.from_items([{"job": job.model_dump_json()} for job in jobs]), config).take_all()

    by_name = {row["name"]: row for row in rows}
    assert Image.open(io.BytesIO(by_name["drawn"]["image"])).size == (64, 32)
    assert by_name["drawn"]["error"] == ""
    assert by_name["failed"]["image"] == b""
    assert by_name["failed"]["error"].startswith("FileNotFoundError: ")
