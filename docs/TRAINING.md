# Training Diffu on RTX PRO 6000 Blackwell

End-to-end: clone → install → smoke test → prep data → train. Stage-1 = flow-matching on real
Swedish lines. Success metric is **downstream HTR CER on held-out real Swedish**, not loss/FID.

This box (`dmlpai01`) has 3× RTX PRO 6000 Blackwell (97 GB each). **GPU 0 is busy** (a foreign
`rerun_plus_olof` job — leave it alone); GPUs **1 + 2 are the free pair**, so every target below pins
`CUDA_VISIBLE_DEVICES` to GPU 1 (single) or GPU 1,2 (dual). Override with `GPUS_1=` / `GPUS_2=` if
that changes.

## 0. Get the code
```bash
git clone https://github.com/AI-Riksarkivet/diffu.git && cd diffu
```

## 1. Install
One step — `uv sync` pulls everything, and `torch`/`torchvision` come from the CUDA 13.0 (cu130) wheel
index automatically (wired into `pyproject.toml`), so the Blackwell GPUs get the right kernels:
```bash
make install        # uv sync  (add `make install-dev` for ruff / pyrefly / pytest)
```
Check the GPUs are visible:
```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.device_count())"   # -> 2.14.1+cu130 3
```

> **DINOv3 is gated.** Accept the model licence on the Hub and run `hf auth login`; without it the
> `StyleEncoder` refuses to load (there is no fallback encoder).

## 2. Smoke test (no dataset needed)
Runs a few real training steps on synthetic data — verifies the full model + the GPU path before you
have data. First run downloads the VAE + style encoder; then it's cached.
```bash
make smoke-1gpu     # GPU 1, single process
make smoke-2gpu     # GPU 1 + 2, accelerate --multi_gpu  (exercises DDP)
```
`smoke OK — 3 step(s) ...` with a falling-ish loss = good.

## 3. Prep data (your PAGE-XML / ALTO-XML export)
```bash
make data XML_DIR=/path/to/xml IMAGE_DIR=/path/to/page_images OUT_DIR=data_out
```
Writes `data_out/lines/*.png`, `data_out/manifest.jsonl`, and writer-disjoint `train/val/test.jsonl`.
It also prints the å ä ö Å Ä Ö coverage — eyeball that the diacritics are well represented.

## 4. Train
```bash
make train-1gpu DATA_DIR=data_out STEPS=100000 BATCH=8          # GPU 1
make train-2gpu DATA_DIR=data_out STEPS=100000 BATCH=8          # GPU 1 + 2
```
`train-2gpu` runs `accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16
-m diffu.train ...`. Checkpoints land in `data_out/checkpoints/` (resumable `accelerate` state
every 10000 steps (`--save-every`) + a `final/`). bf16 + gradient checkpointing + EMA are on; FlashAttention is used
automatically via PyTorch SDPA (SD3's `JointAttnProcessor2_0`). Add `--compile` for `torch.compile`
once line widths are stable.

**Tuning knobs:** `BATCH` is per GPU (raise it on the 97 GB cards — try 16–32), `LR`, `STEPS`.
Lines are width-bucketed and right-padded per batch (cap `data.max_line_width=1024`).

**The settled recipe** (what the released `exp_sd35_128` run used, plus the measured perf config):
`--glyph-line` (cross-attention content — beats concat), `--cond-dropout 0.1 --text-dropout 0.1`
(the CFG-null fix: trains BOTH the full null and the text-only null that inference CFG uses), and the
~1.9× perf trio `--bucket-width --no-grad-checkpoint --compile` (accuracy-safe; GPU idle 50%→6%).
Measurements and caveats: `docs/PERF_AUDIT.md` §0e.

Add `--cache-latents` (opt-in, default off) to skip the frozen VAE encode — it pre-encodes each line
once and feeds the cached latent as `x0`, measured **1.36× faster** (the encode was ~23% of the step).
Pair it with `--bucket-width` and `--white-pad-prob 0`. Like `--bucket-width`, it's opt-in: it stores the
posterior mode (not a per-step sample), so validate gen_CER on a full run before making it default.

> `train.jsonl` must have **at least `BATCH × num_processes` lines** (≥16 for `BATCH=8` on 2 GPUs) —
> `drop_last=True` means a smaller set yields 0 batches. The trainer fails fast with a clear message
> instead of hanging; lower `BATCH` for a tiny bootstrap set.

## 5. Optional modes & next steps
- **REPA alignment** — **on by default** (aligns a mid-block DiT hidden state to DINO features of the
  target); `diffu-train --no-repa` disables it.
- **Evaluate** (the success metric) — `diffu-eval --manifest data_out/val.jsonl` reports HTR CER.
- **Generate** — `diffu-generate --text "Smörgåsbord" --style ref.png --ckpt data_out/checkpoints/final/...`.
- **Stage-0 VAE recon-CER gate** on real Riksarkivet ink (`make vae-gate` / `diffu-vae-gate`) — confirm
  å ä ö survive encode→decode on *degraded* ink before trusting the latent.
- **Tests** — `make test` runs the pytest suite (data pipeline, flow math, CER, eval).

## 6. Stage-3 — LoRA adapters (adapt a base to a new hand/collection)
`--lora` freezes the base and trains only a low-rank adapter on the backbone's joint-attention
projections (peft `LoraConfig` on the SD3 MMDiT — a `PeftAdapterMixin`). A run writes a **few-MB
`pytorch_lora_weights.safetensors`** instead of a 17 GB checkpoint, so per-collection adapters are cheap
to store and swap. Pair `--lora` with `--init-from <base>` — LoRA on a random base learns nothing useful.

You can train straight off a **Hub dataset** (image + text) with `--hf-dataset` — no local manifest, no
`make data`. Needs the extra: `uv sync --extra hf`.

```bash
# adapt the released base to the IAM handwriting set (English), rank-16 adapter
CUDA_VISIBLE_DEVICES=1 uv run diffu-train \
    --init-from data_out/checkpoints/final/model.safetensors \
    --hf-dataset Teklia/IAM-line \
    --lora --lora-rank 16 --lora-alpha 32 \
    --steps 3000 --batch-size 8 --out-dir runs/iam_lora
# -> runs/iam_lora/final/pytorch_lora_weights.safetensors  (~12 MB)
```

Inference loads the adapter onto the same base (`--lora` is a dir or a Hub id; `--lora-scale` in `0..1+`):

```bash
diffu-generate --ckpt data_out/checkpoints/final/model.safetensors \
    --lora runs/iam_lora/final --lora-scale 1.0 \
    --text "the quick brown fox" --style ref_line.png --line-height 128 --out iam.png
```

or in Python: `DiffuPipeline.from_pretrained("Gabriel/diffu_sd35_128", lora="runs/iam_lora/final")`.
Knobs: `--lora-rank` (capacity), `--lora-alpha` (scale = alpha/r), `--lora-dropout`. Adapters save/resume
as safetensors; a full-state resume (`--resume`) is for non-LoRA runs.

## Gotchas
- Qwen VAE is video-derived (5D) — handled internally (T=1 frame dim); no action needed.
- If `accelerate` asks to configure, the Makefile passes flags explicitly so you can skip
  `accelerate config`.
- GPU 0 busy? That's expected (foreign job) — the targets avoid it. To use other GPUs: `make train-2gpu GPUS_2=1,2`.
