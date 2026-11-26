"""diffu_sglang_moss, with SGLang's refusal of torch.compile for MOSS-Image lifted, to measure it (26 Nov).

SGLang 0.5.23's MOSS-Image config refuses ``enable_torch_compile`` ("support has not been validated"). This package
is given as SGLANG_EXTERNAL_MODEL_PACKAGE in place of the plugin: it imports the plugin (which registers itself) and
drops that one check, keeping the others (TP, autocast). Nothing of SGLang or of the plugin is changed on disk.
"""

import diffu_sglang_moss  # noqa: F401  registers the pipeline
from diffu_sglang_moss.pipeline import DiffuMOSSImagePipelineConfig


def _validate(self, server_args) -> None:
    if server_args.tp_size != 1:
        raise ValueError("MOSSImage supports sequence and CFG parallelism, not TP.")
    if not server_args.disable_autocast:
        raise ValueError("MOSSImage requires disabled autocast to preserve reference precision.")


DiffuMOSSImagePipelineConfig.validate_server_args = _validate


# --- 26 Nov, 22:10: the sample "above" the page -------------------------------------------------------------------
# diffu-moss-above was trained with its sample page's rows lying above the drawn page's (its config's
# ``reference_place: above``; diffu_moss.transformer.token_positions). The plugin's DiT puts every reference centre on
# centre ("over", MOSS-Image's own), so SGLang drew a copy of the sample (PSNR to our pipeline's page 16.6 dB). Here
# the references' rows are lifted as token_positions lifts them, until the plugin does it itself.
import os

import torch
from diffu_sglang_moss.dit import DiffuMOSSImageTransformer, _prepare_rope, reference_positions

_sample_of = DiffuMOSSImageTransformer._sample


def _place(self) -> str:
    return getattr(self.config, "reference_place", None) or os.environ.get("MOSS_REFERENCE_PLACE", "over")


def _sample(self, latents, *rest):
    self._drawn_rows = latents.shape[2]
    return _sample_of(self, latents, *rest)


def _reference_rope(self, grids, text_length, like):
    if self.rope.bands.dtype != torch.float32:
        self.rope.reset_bands()
    grids_positions, lifted = [], 0
    for k, (h, w) in enumerate(grids):
        positions = reference_positions(h, w, text_length, k + 1, like.device)
        if _place(self) == "above":
            positions[:, 2] -= (self._drawn_rows + h) / 2 + lifted
            lifted += h
        grids_positions.append(positions)
    phase = torch.cat(grids_positions).unsqueeze(-1) * self.rope.bands
    table = torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0).to(like.dtype)
    sin, cos = _prepare_rope(table)
    return sin.expand(like.shape[0], -1, -1).contiguous(), cos.expand(like.shape[0], -1, -1).contiguous()


DiffuMOSSImageTransformer._sample = _sample
DiffuMOSSImageTransformer._reference_rope = _reference_rope
