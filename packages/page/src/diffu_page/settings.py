"""Where diffu-page finds its models: typed settings, read once from the environment or a ``.env`` file."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """The defaults the CLIs use, from the environment or a ``.env`` file in the working directory (gitignored).

    Attributes:
        flux3_model: The FLUX.3 Klein directory (a Modular Diffusers repo), env ``FLUX3_MODEL``. None when unset: a
            command that needs a model then asks for ``--model`` (``cli.options.required_model``).
        htr_model: The HTR model every drawn line is read back with (a Hub id or a local copy, for offline nodes), env
            ``DIFFU_HTR_MODEL``.
        htr_revision: Its pinned Hub revision, so every CER and unclear tag is read by the same weights; env
            ``DIFFU_HTR_REVISION``, empty for a local copy.
        print_hand_model: The print/hand classifier ``diffu-generate --mark-from-scan`` reads a template's lines with
            (``python -m diffu_page.marking.train``'s ``print_hand.pt``), env ``DIFFU_PRINT_HAND_MODEL``.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    flux3_model: Path | None = Field(default=None, alias="FLUX3_MODEL")
    htr_model: str = Field(default="Riksarkivet/trocr-base-handwritten-hist-swe-2", alias="DIFFU_HTR_MODEL")
    htr_revision: str | None = Field(default="aa79fcb1850bf3155ebc442570d6c6bfc0ac8100", alias="DIFFU_HTR_REVISION")
    print_hand_model: Path = Field(default=Path("runs/print_hand/print_hand.pt"), alias="DIFFU_PRINT_HAND_MODEL")


settings = Settings()
