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
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    flux3_model: Path | None = Field(default=None, alias="FLUX3_MODEL")


settings = Settings()
