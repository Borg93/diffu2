"""Eruku-backed renderer — a public autoregressive styled-text generator behind the same seam."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .matte import ink_from_pil

if TYPE_CHECKING:
    from PIL.Image import Image


class ErukuRenderer:
    """A :class:`LineRenderer` backed by Eruku — a public autoregressive styled-text generator.

    Lets the page pipeline run with REAL handwriting without a Diffu checkpoint (any line generator
    that satisfies :class:`LineRenderer` drops in here). Reuses the project's torch + transformers
    (lazy import); one style reference = one writer per page.

    Caveat: Eruku is English/modern (IAM/CVL/RIMES/FontSquare), NOT Swedish-historical — use it to
    validate the *pipeline*, not to produce domain-correct data; å ä ö may render poorly. A GPU is
    needed in practice (autoregressive generation per line).
    """

    def __init__(
        self,
        style_path: str,
        *,
        cfg_scale: float = 1.25,
        model_id: str = "blowing-up-groundhogs/eruku",
        device: str | None = None,
    ) -> None:
        self.style_path = style_path
        self.cfg_scale = cfg_scale
        self.model_id = model_id
        self.device = device
        self._model: Any = None
        self._style: Image | None = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch  # lazy
        from PIL import Image
        from transformers import AutoModel

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True).to(self.device).eval()
        self._style = Image.open(self.style_path).convert("RGB")

    def render(self, text: str, height: int) -> np.ndarray:
        import torch  # lazy

        self._ensure()
        if self._style is None:
            raise RuntimeError("ErukuRenderer not initialised")  # _ensure guarantees this
        with torch.inference_mode():
            pil = self._model.generate_handwriting(
                style_image=self._style, gen_text=text or " ", style_text="", cfg_scale=self.cfg_scale
            )
        return ink_from_pil(pil, height)
