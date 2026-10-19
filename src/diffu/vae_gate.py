"""Stage-0 VAE diacritic gate — the Stage-0 acceptance check (run via `diffu-vae-gate` / `make vae-gate`).

Round-trips line images through the 16-ch VAE (encode -> decode) and checks that the HTR Character
Error Rate of the reconstruction stays within ``cfg.vae.recon_cer_gate`` of the raw-image CER — i.e.
the VAE preserves å ä ö well enough to train on. The single most important check before training.

Reuses ``VAEWrapper.reconstruction_cer_gate`` + the eval ``HTRRecognizer``.

    diffu-vae-gate --manifest data_out/val.jsonl --n 500       # on YOUR real lines
    diffu-vae-gate --synthetic --save-dir vae_check             # self-test on 6 sample words, no data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as TF

from .config import Config
from .data.dataset import load_manifest
from .eval import STANDARD_HTR, HTRRecognizer
from .model.vae import VAEWrapper

_SAMPLE_WORDS = ("Göteborg", "Smörgåsbord", "Råå församling", "Ängelholm", "väderöarna", "Åsa köper öl")


def _pad_batch(tensors: list[torch.Tensor], multiple: int = 16) -> torch.Tensor:
    """Right-pad ``[3,H,W]`` tensors to a common (rounded) width with white (1.0) and stack."""
    target = max(t.shape[-1] for t in tensors)
    target = ((target + multiple - 1) // multiple) * multiple
    return torch.stack([F.pad(t, (0, target - t.shape[-1]), value=1.0) for t in tensors])  # [B,3,H,W]


def load_real_lines(manifest: str, n: int, cfg: Config) -> tuple[torch.Tensor, list[str]]:
    """Load up to ``n`` lines from a manifest -> padded ``[B,3,H,W]`` batch in ``[-1,1]`` + texts."""
    rows = load_manifest(manifest)[:n]
    height, max_width = cfg.data.line_height, cfg.data.max_line_width
    tensors, texts = [], []
    for r in rows:
        with Image.open(r["image"]) as im:
            img = im.convert("RGB")
        w = min(max_width, max(8, round(img.width * height / img.height)))
        img = img.resize((w, height), Image.LANCZOS)
        tensors.append(TF.normalize(TF.to_tensor(img), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))  # [3,H,W] in [-1,1]
        texts.append(r["text"])
    return _pad_batch(tensors), texts


def render_synthetic(
    words: tuple[str, ...], height: int, font: str | None = None
) -> tuple[torch.Tensor, list[str]]:
    """Render sample Swedish words to ``[B,3,H,W]`` in ``[-1,1]`` (a data-free gate self-test)."""
    size = int(height * 0.6)
    fnt = ImageFont.truetype(font, size) if font else ImageFont.load_default(size)
    tensors = []
    for word in words:
        w = max(height, int(len(word) * size * 0.6))
        img = Image.new("RGB", (w, height), "white")
        ImageDraw.Draw(img).text((4, (height - size) // 2), word, fill="black", font=fnt)
        tensors.append(TF.normalize(TF.to_tensor(img), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    return _pad_batch(tensors), list(words)


def _save_pairs(originals: torch.Tensor, recon: torch.Tensor, save_dir: str, k: int = 8) -> None:
    """Write original vs round-trip PNGs (stacked vertically) for the first ``k`` lines."""
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(min(k, originals.shape[0])):
        pair = torch.cat([originals[i], recon[i]], dim=1)  # stack original over recon -> [3,2H,W]
        TF.to_pil_image(((pair.clamp(-1, 1) + 1) / 2).cpu()).save(out / f"pair_{i:02d}.png")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-0 VAE diacritic gate (round-trip recon CER).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="jsonl of {image, text} real lines")
    src.add_argument("--synthetic", action="store_true", help="render sample words instead (no data needed)")
    ap.add_argument("--n", type=int, default=16, help="number of lines to test")
    ap.add_argument("--htr", default=STANDARD_HTR, help="HF TrOCR id (default: the standard recognizer)")
    ap.add_argument("--font", default=None, help="TTF for --synthetic (default: PIL built-in)")
    ap.add_argument("--save-dir", default=None, help="write original-vs-recon PNG pairs here")
    ap.add_argument("--batch-size", type=int, default=8, help="VAE round-trip chunk size (bounds GPU memory)")
    ap.add_argument("--track", action="store_true", help="log the Stage-0 gate result to trackio")
    args = ap.parse_args()

    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.synthetic:
        images, texts = render_synthetic(
            _SAMPLE_WORDS[: args.n] or _SAMPLE_WORDS, cfg.data.line_height, args.font
        )
    else:
        images, texts = load_real_lines(args.manifest, args.n, cfg)
    images = images.to(device)

    vae = VAEWrapper(cfg.vae).to(device).eval()
    recognizer = HTRRecognizer(args.htr, device=device)
    result = vae.reconstruction_cer_gate(images, texts, recognizer, batch_size=args.batch_size)

    print(
        f"raw CER {result['raw_cer']:.3f}  |  recon CER {result['recon_cer']:.3f}  |  "
        f"gap {result['recon_cer'] - result['raw_cer']:+.3f}  (threshold +{cfg.vae.recon_cer_gate})"
    )
    print(
        "GATE:", "PASS — VAE preserves the text" if result["passed"] else "FAIL — fine-tune the VAE decoder"
    )
    if args.track:  # record the gate alongside Stage-1/2 runs so trackio holds all three stages
        import trackio

        trackio.init(project="diffu", config={"stage": "stage0", "n": args.n, "htr": args.htr})
        trackio.log(
            {
                "raw_cer": float(result["raw_cer"]),
                "recon_cer": float(result["recon_cer"]),
                "gap": float(result["recon_cer"]) - float(result["raw_cer"]),
                "passed": int(bool(result["passed"])),
            }
        )
        trackio.finish()
        print("logged Stage-0 gate to trackio (project diffu, stage=stage0)")
    if args.save_dir:  # only the first 8 lines are saved, so don't re-decode all of them (OOM)
        with torch.no_grad():
            recon = vae.decode(vae.encode(images[:8]))
        _save_pairs(images[:8], recon, args.save_dir)
        print(f"wrote visual pairs -> {args.save_dir}")


if __name__ == "__main__":
    main()
