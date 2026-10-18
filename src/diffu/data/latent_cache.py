"""Offline VAE latent cache for Stage-1 training — skip the frozen f8 encoder every step.

Wired into training via ``diffu-train --cache-latents`` (or ``make train-* CACHE_LATENTS=1``): the flag
builds the cache once if missing, attaches :class:`LatentCache` to the train dataset, and swaps in a
latent-space collate so each step feeds the cached latent as ``x0`` instead of calling ``vae.encode``.
Build it standalone with ``diffu-cache-latents --data-dir <dir>`` (or ``python -m diffu.data.latent_cache``).

Profiling (docs/PERF_AUDIT.md §0) shows the frozen Qwen video-VAE ``conv3d`` encode is the single biggest
removable op: ~23% of the training step's CUDA time (bigger than the backbone forward). The encoder is frozen
and, with ``white_pad_prob=0``, each line's input is deterministic, so its normalized latent is a pure function
of the image — precompute it once and reuse it every epoch.

Storage: a fixed-width float16 memmap ``[N, C, h, w_max]`` (``w_max = max_line_width // downscale``) plus a
``latent_w`` index (the real per-line latent width; the rest of each row is zero and is sliced off on read), a
``white`` column ``[C, h, 1]`` = the encoder's steady-state white-paper latent (the correct value to right-pad
a batch with — the encoder is nonlinear/biased, so padding the latent with zeros would NOT match encoding a
white-padded image), plus a JSON header that STAMPS the encoder identity so a mismatched cache refuses to load.
The padded layout wastes ~45% disk vs a ragged blob but needs no offset arithmetic and is trivially worker-safe
(the memmap is reopened lazily per DataLoader worker).

Quality: the cache stores the posterior MODE (``sample=False``), not a per-step posterior sample — the f8
posterior std is tiny and flow matching injects its own noise, so this is behavior-safe (docs/PERF_AUDIT.md).
Like ``--bucket-width``, treat it as opt-in and gate a full-run gen_CER check before making it the default.

Constraints: Stage-1 only (Stage-0 fine-tunes the decoder and needs images); requires ``white_pad_prob=0`` (the
augmentation changes the input → invalidates the cache); REPA / ink-focal / fill-ratio still read the raw image,
so the dataset keeps loading images — the cache removes only the encode.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch
from pydantic import BaseModel

if TYPE_CHECKING:
    from ..config import Config

_DAT = "{split}.latents.f16.dat"
_WIDTHS = "{split}.latents.widths.npy"
_META = "{split}.latents.meta.json"
_WHITE = "{split}.latents.white.npy"  # [C, h, 1] encoder steady-state white-paper latent (batch right-pad value)


class LatentCacheMeta(BaseModel):
    """Header stamping the encoder identity so a stale/mismatched cache refuses to load."""

    n: int
    channels: int
    latent_h: int
    w_max: int
    pretrained: str
    subfolder: str
    line_height: int
    max_line_width: int
    downscale_factor: int


def _meta_for(cfg: Config, n: int) -> LatentCacheMeta:
    w_max = cfg.data.max_line_width // cfg.vae.downscale_factor
    return LatentCacheMeta(
        n=n,
        channels=cfg.vae.latent_channels,
        latent_h=cfg.data.line_height // cfg.vae.downscale_factor,
        w_max=w_max,
        pretrained=cfg.vae.pretrained,
        subfolder=cfg.vae.subfolder,
        line_height=cfg.data.line_height,
        max_line_width=cfg.data.max_line_width,
        downscale_factor=cfg.vae.downscale_factor,
    )


class LatentCache:
    """Read-only, worker-safe reader over a built latent cache (see module docstring)."""

    def __init__(self, cache_dir: str, split: str, expected: LatentCacheMeta) -> None:
        meta_path = os.path.join(cache_dir, _META.format(split=split))
        with open(meta_path) as fh:
            meta = LatentCacheMeta.model_validate_json(fh.read())
        if meta.model_dump() != expected.model_dump():
            raise ValueError(
                f"latent cache {meta_path} was built for a different encoder/geometry than the current config "
                f"({meta.model_dump()} != {expected.model_dump()}); rebuild with "
                f"`python -m diffu.data.latent_cache`."
            )
        self.meta = meta
        self._dat_path = os.path.join(cache_dir, _DAT.format(split=split))
        self._widths: np.ndarray = np.load(os.path.join(cache_dir, _WIDTHS.format(split=split)))
        self._mm: np.memmap | None = None  # opened lazily per process/worker (fork-safe)
        # Encoder white-paper latent column [C, h, 1] — the batch right-pad value (see module docstring).
        white = np.load(os.path.join(cache_dir, _WHITE.format(split=split)))
        self.white_column: torch.Tensor = torch.from_numpy(white.astype(np.float32))

    def _memmap(self) -> np.memmap:
        if self._mm is None:
            m = self.meta
            self._mm = np.memmap(
                self._dat_path, dtype=np.float16, mode="r", shape=(m.n, m.channels, m.latent_h, m.w_max)
            )
        return self._mm

    def __len__(self) -> int:
        return self.meta.n

    def __getitem__(self, i: int) -> torch.Tensor:
        w = int(self._widths[i])
        block = np.asarray(self._memmap()[i, :, :, :w])  # [C, h, w] — copy out of the memmap
        return torch.from_numpy(block.copy())


@torch.no_grad()
def build_latent_cache(
    cfg: Config, data_dir: str, split: str, *, device: str = "cuda", batch_size: int = 16
) -> str:
    """Pre-encode every line in ``{data_dir}/{split}.jsonl`` to a deterministic normalized latent.

    Encodes under bf16 autocast (matching the live training path) with the posterior MODE (deterministic), and
    writes the memmap + widths index + stamped header into ``data_dir``. Returns the memmap path.
    """
    from PIL import Image
    from torchvision.transforms import functional as TF

    from ..model.vae import VAEWrapper
    from .dataset import load_manifest

    rows = load_manifest(os.path.join(data_dir, f"{split}.jsonl"))
    meta = _meta_for(cfg, len(rows))
    vae = VAEWrapper(cfg.vae).to(device).eval()

    # White-paper latent column [C, h, 1]: encode a pure-white strip (1.0 = white in [-1,1]) and take a
    # steady-state interior column. This is the value the collate right-pads batched latents with — it
    # matches encoding a white-padded image; padding with zeros would not (the encoder is nonlinear/biased).
    white_w = min(cfg.data.max_line_width, 256)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        white_lat = vae.encode(torch.ones(1, 3, cfg.data.line_height, white_w, device=device), sample=False)
    mid = white_lat.shape[-1] // 2
    wcol = white_lat[0, :, :, mid : mid + 1].float().cpu().numpy().astype(np.float16)  # [C, h, 1]

    dat_path = os.path.join(data_dir, _DAT.format(split=split))
    mm = np.memmap(
        dat_path, dtype=np.float16, mode="w+", shape=(meta.n, meta.channels, meta.latent_h, meta.w_max)
    )
    widths = np.zeros(meta.n, dtype=np.int32)

    def load(path: str) -> torch.Tensor:
        with Image.open(path) as im:
            img = im.convert("RGB")
        if img.width > cfg.data.max_line_width:
            img = img.resize((cfg.data.max_line_width, cfg.data.line_height), Image.LANCZOS)
        return TF.normalize(TF.to_tensor(img), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    from .sampler import line_widths

    # Encode in WIDTH-SORTED batches so each batch is near-homogeneous -> minimal intra-batch white padding ->
    # the conv receptive field barely bleeds foreign padding into a line's boundary latent columns, so the
    # cached latent matches what bucketed training (which also pads to ~the line's own width) would encode.
    widths_px = line_widths(
        rows, max_width=cfg.data.max_line_width, cache_path=os.path.join(data_dir, f"{split}.widths.json")
    )
    order = sorted(range(meta.n), key=widths_px.__getitem__)
    for start in range(0, meta.n, batch_size):
        batch_idx = order[start : start + batch_size]
        imgs = [load(rows[k]["image"]) for k in batch_idx]
        wmax = max(t.shape[-1] for t in imgs)
        batch = torch.stack([torch.nn.functional.pad(t, (0, wmax - t.shape[-1]), value=1.0) for t in imgs])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z = vae.encode(batch.to(device), sample=False)  # [B, C, h, wmax//8] deterministic (mode)
        z = z.float().cpu().numpy().astype(np.float16)
        if z.shape[-2] != meta.latent_h:  # fail loudly on the line-height / data-height footgun
            raise ValueError(
                f"encoded latent height {z.shape[-2]} != expected {meta.latent_h}: the line images are "
                f"{z.shape[-2] * cfg.vae.downscale_factor}px tall, but cfg.data.line_height is "
                f"{cfg.data.line_height}px. Point --cache-latents at data prepared at the same height."
            )
        zw = z.shape[-1]  # the actual latent width the VAE produced for this padded (wmax) batch
        for j, k in enumerate(batch_idx):
            # the line's content is the first proportional slice (uniform stride-downscale conv); clamp in.
            lw = min(round(imgs[j].shape[-1] * zw / wmax), zw, meta.w_max)
            mm[k, :, :, :lw] = z[j, :, :, :lw]
            widths[k] = lw
        if start % (batch_size * 200) == 0:
            print(f"  encoded {start + len(batch_idx)}/{meta.n}", flush=True)

    mm.flush()
    np.save(os.path.join(data_dir, _WIDTHS.format(split=split)), widths)
    np.save(os.path.join(data_dir, _WHITE.format(split=split)), wcol)
    with open(os.path.join(data_dir, _META.format(split=split)), "w") as fh:
        fh.write(meta.model_dump_json())
    print(f"latent cache written: {dat_path} ({meta.n} lines)", flush=True)
    return dat_path


def open_cache(cfg: Config, data_dir: str, split: str) -> LatentCache:
    """Open an existing cache, validating it matches ``cfg`` (raises if missing/mismatched)."""
    return LatentCache(data_dir, split, _meta_for(cfg, _row_count(data_dir, split)))


def _row_count(data_dir: str, split: str) -> int:
    with open(os.path.join(data_dir, f"{split}.jsonl"), encoding="utf-8") as f:
        return sum(1 for _ in f)


def main() -> None:
    import argparse

    from ..config import Config

    ap = argparse.ArgumentParser(
        description="Pre-encode line images to a VAE latent cache (python -m diffu.data.latent_cache)."
    )
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    build_latent_cache(Config(), args.data_dir, args.split, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
