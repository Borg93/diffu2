"""Where diffu-page finds its models: typed settings, read from the environment once."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """The defaults the CLIs use; each can be overridden in the environment.

    Attributes:
        flux3_model: The FLUX.3 Klein directory (a Modular Diffusers repo), env ``FLUX3_MODEL``.
    """

    flux3_model: Path = Field(
        default=Path.home() / "Desktop/repos/flux3-models/flux-3-image-klein-base", alias="FLUX3_MODEL"
    )


settings = Settings()
