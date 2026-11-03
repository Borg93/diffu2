"""Where htr-train finds and writes things on this machine, read from ``HTR_TRAIN_*`` environment variables (a path
relative to the repository root it is run from)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

QUEUE = Path.home() / "diffu_queue"  # outside /tmp, which a reboot empties: a queue outlives a reboot


class Settings(BaseSettings):
    """htr-train's machine settings.

    Attributes:
        runs: Each run's folder ``<runs>/<name>/``: its spec, manifest, scripts and ms-swift's checkpoints.
        envs: The tool environments, each a uv project with its ``.venv`` (``<envs>/<env>/.venv/bin/...``).
        guard: The Spark's memory guard every GPU job runs under.
        floor_file: The file holding the guard's free-memory floor (GB) the queue scripts read; ``floor_gb`` when it
            is missing.
        floor_gb: The guard's floor when ``floor_file`` is missing.
        memory_gb: The Spark's memory, its GPU's included (unified).
        queue: Where ``htr-train launch --queue`` writes a run's script for the queue.
        trackio_project: The trackio project ms-swift's runs log to (local, never a Space).
        slurm_partition: The Slurm partition a run is submitted to (None: the site's default).
        slurm_account: The Slurm account a run is billed to (None: the user's default).
        jev_font: The font the Jev renderer draws candidate texts in; its digest is part of the renderer's version.
    """

    model_config = SettingsConfigDict(env_prefix="HTR_TRAIN_")

    runs: Path = Path("runs/htr")
    envs: Path = Path("packages/htr_train/envs")
    guard: Path = Path("jobs/spark_guard.sh")
    floor_file: Path = QUEUE / "guard_floor_gb"
    floor_gb: int = Field(default=20, ge=0)
    memory_gb: int = Field(default=119, gt=0)
    queue: Path = QUEUE
    trackio_project: str = "diffu"
    slurm_partition: str | None = None
    slurm_account: str | None = None
    jev_font: Path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

    def guard_floor(self) -> int:
        """The guard's floor now: the number in ``floor_file``, or ``floor_gb`` where there is no such file.

        Raises:
            ValueError: If the file holds no whole number of GB.
        """
        if not self.floor_file.is_file():
            return self.floor_gb
        text = self.floor_file.read_text(encoding="utf-8").strip()
        if not text.isdigit():
            raise ValueError(f"{self.floor_file} holds {text!r}, not the guard's floor in whole GB")
        return int(text)
