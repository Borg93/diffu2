"""Where the pools and the ground truth are on this machine, from the environment (``DIFFU_CURATE_*``)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


class Pool(BaseModel):
    """A pool of machine-labelled scans: its name in the tables, the folder of its volumes of ALTO files and the folder
    of its volumes of scans (a scan is ``<volume>/<stem>.jpg``, its labels ``<volume>/<stem>.xml``)."""

    model_config = ConfigDict(frozen=True)

    name: str
    labels: Path
    scans: Path

    def volumes(self) -> list[Path]:
        """The pool's volumes of label files, by name."""
        return sorted(folder for folder in self.labels.iterdir() if folder.is_dir())

    def label_file(self, stem: str) -> Path:
        """The label file of the scan named ``stem`` (``<volume>_<page>``)."""
        return self.labels / volume_of(stem) / f"{stem}.xml"

    def scan_file(self, stem: str) -> Path:
        """The image of the scan named ``stem``."""
        return self.scans / volume_of(stem) / f"{stem}.jpg"


COLLECTION_PREFIX = "GT-"  # how the ground truth names its collections' folders (GT-RA_..., GT-SSA_..., GT-FNA_...)


def collection_of(path: Path) -> str:
    """The ground-truth collection a file lies in: the folder nearest to it whose name starts with
    :data:`COLLECTION_PREFIX` (the ground truth's own root folder may start with it too); ``""`` if none."""
    return next((folder.name for folder in path.parents if folder.name.startswith(COLLECTION_PREFIX)), "")


def volume_of(stem: str) -> str:
    """The volume a scan's stem names: what stands before its first underscore."""
    return stem.split("_", 1)[0]


SVEA = Pool(
    name="svea",
    labels=Path("/mnt/work/haas/deliveries_old/svea-all"),
    scans=Path("/mnt/scratch/haas/input/svea"),
)
ANDRAARKIV = Pool(
    name="andraarkiv",
    labels=Path("/mnt/work/haas/output/andraarkiv"),
    scans=Path("/mnt/work/haas/input/andraarkiv"),
)


class CurateSettings(BaseSettings):
    """The pools by name, the ground truth's root (GT-Lejonet: ``GT-*/<volume>/<volume>/alto/*.xml``) and which of
    its test lists names its test scans."""

    model_config = SettingsConfigDict(env_prefix="DIFFU_CURATE_", env_nested_delimiter="__")

    pools: dict[str, Pool] = {SVEA.name: SVEA, ANDRAARKIV.name: ANDRAARKIV}
    ground_truth: Path = Path.home() / "Desktop/data/GT-Lejonet_HTR_202602"
    test_list: str = "test_0050"  # the ground truth's test list every set here is split by

    def pool(self, name: str) -> Pool:
        """The pool named ``name``.

        Raises:
            ValueError: If no pool has that name; the message lists those there are.
        """
        if name not in self.pools:
            raise ValueError(f"no pool named {name!r}: there are {', '.join(sorted(self.pools))}")
        return self.pools[name]
