"""``diffu-studio`` CLI — resolve the device, build the engine, launch the gradio.Server control panel.

Safety first (condition G): set the CUDA fragmentation guard and pin a single GPU BEFORE any torch import,
so the studio coexists with the training run and never OOMs it.
"""

from __future__ import annotations

import argparse
import os

from diffu_studio.device import configure_memory


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Diffu Studio — handwriting synthesis control panel.")
    ap.add_argument(
        "--ckpt", default=None, help="checkpoint to preselect (step_* dir or .safetensors); else newest"
    )
    ap.add_argument("--ckpt-root", default="checkpoints", help="root dir scanned for checkpoints")
    ap.add_argument("--gallery", default=None, help="manifest jsonl to sample style references from")
    ap.add_argument("--device", default=None, help="zerogpu | gpu | cpu (default: autodetect)")
    ap.add_argument(
        "--gpu-index", default="0", help="which CUDA device to pin (default 0 — the co-tenant GPU)"
    )
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a public gradio.live tunnel link")
    ap.add_argument(
        "--recognizer", default=None, help="TrOCR repo id for read-back (default: the gen_CER model)"
    )
    ap.add_argument("--no-recognizer", action="store_true", help="disable the TrOCR read-back panel")
    ap.add_argument(
        "--no-glyph-line", dest="glyph_line", action="store_false", help="checkpoint has per-char glyphs"
    )
    ap.add_argument(
        "--style-tokens", dest="style_in_context", action="store_true", help="checkpoint uses style tokens"
    )
    ap.add_argument("--glyph-concat", action="store_true", help="checkpoint channel-concats a glyph latent")
    ap.add_argument(
        "--line-height", type=int, default=64, help="crop height the checkpoint was trained at (64 or 128)"
    )
    ap.set_defaults(glyph_line=True, style_in_context=False)
    return ap.parse_args()


def _default_gallery() -> str | None:
    """The style gallery is a core feature, not a flag — auto-pick the first val manifest that exists."""
    for cand in ("data_out_lejonet/val.jsonl", "data_out_lejonet_128/val.jsonl"):
        if os.path.exists(cand):
            return cand
    return None


def main() -> None:
    configure_memory()  # MUST precede any torch import
    args = _parse_args()
    if args.device != "cpu":
        os.environ.setdefault(
            "CUDA_VISIBLE_DEVICES", args.gpu_index
        )  # pin one GPU; keep training's untouched
    # Every weight the studio needs is already cached on this box (training uses DINOv3 + the Qwen VAE; the
    # recognizer loads from cache too). Force HF offline so a flaky online revision check on the first
    # from_pretrained can never hang the generate queue — the "everything is just queued" failure where the
    # worker sits in huggingface_hub's retry-backoff and every click piles up behind it. Override with
    # HF_HUB_OFFLINE=0 to allow downloads. On a Hugging Face Space (SPACE_ID set) nothing is pre-cached,
    # so weights (PiD, Gemma-2, the recognizer) MUST download on first use — leave offline mode off there.
    if not os.environ.get("SPACE_ID"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # Imports that pull torch stay below the env setup above.
    from diffu.config import Config
    from diffu_studio.checkpoints import ArchConfig
    from diffu_studio.core import list_checkpoints, resolve_device
    from diffu_studio.engine import StudioEngine
    from diffu_studio.line import sample_gallery
    from diffu_studio.ui import build_blocks

    device = resolve_device(args.device)
    if device.kind.value == "zerogpu":
        # Pull PiD/Gemma weights on CPU at boot — the @spaces.GPU seconds budget must not pay for downloads.
        from diffu_studio.upscale import prefetch_weights

        prefetch_weights()
    arch = ArchConfig(
        glyph_line=args.glyph_line,
        style_in_context=args.style_in_context,
        glyph_concat=args.glyph_concat,
        line_height=args.line_height,
    )
    recognizer_name = None if args.no_recognizer else (args.recognizer or Config().aux.htr_recognizer_eval)
    engine = StudioEngine(device=device, ckpt_root=args.ckpt_root, arch=arch, recognizer_name=recognizer_name)

    found = list_checkpoints(args.ckpt_root)
    default_ckpt = args.ckpt or (found[0].path if found else None)  # found is sorted newest-first
    gallery_manifest = args.gallery or _default_gallery()  # the style gallery is core, not a flag
    gallery = sample_gallery(gallery_manifest) if gallery_manifest else None

    print(f"[studio] device={device.label}  checkpoints={len(found)}  default={default_ckpt}")
    # Launch the Blocks UI directly with its queue running. NOT via gr.Server + mount_gradio_app: gr.Server.launch
    # builds a fresh server-Blocks and re-launches (gradio server.py:284), clobbering the lifespan wrapper that
    # mount_gradio_app installs to start the mounted UI's queue (gradio routes.py:2606) — so every UI event
    # enqueued with no worker to run it ("everything just gets queued"). The SSE @app.api layer in server.py is
    # for a future custom frontend with no consumer yet; re-add it later by mounting this queued Blocks onto a
    # FastAPI app (gr.mount_gradio_app) instead of wrapping it in gr.Server.
    blocks = build_blocks(engine, gallery_paths=gallery, default_ckpt=default_ckpt)
    blocks.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
