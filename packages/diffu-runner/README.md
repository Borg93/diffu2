# diffu-runner

Run **Diffu at scale** to forge synthetic HTR datasets — a uv **workspace member** that isolates the heavy,
optional **Ray** dependency from the training (`diffu`) and page-composition (`diffu-page`) packages.

## Line pipeline

It turns the single-line model into a **million-line, leakage-safe, quality-gated** training set by
streaming two stateful Ray actors:

```
work-list (train-only text × style)
    → DiffuActor   (GPU actor — generate a line per (text, style))     concurrency = generator GPUs
    → ink gate     (CPU flag — blanks / doubled-render smears)          free
    → TrOCRActor   (fractional-GPU actor — read back, score CER)        co-located readers
    → ink + CER filter (keep CER ≤ cer_max)
    → Lance dataset (clean rows only: png · text · cer [· style_path · doc_id · line_idx])
```

**Why Ray Data here:** streaming with **backpressure** (the cheap TrOCR gate never bottlenecks the slow
generator), **stateful actors** (each model loaded once per worker), and **multi-node** scaling by bumping
`concurrency`. TrOCR is ~10–20× cheaper than Diffu per line, so it co-locates on the generator GPUs — the
gate is a small % of pipeline cost, folded inline instead of a separate pass over a million files.

**Why a Lance dataset (not parquet shards):** downstream consumers — the training loader and the page
compositor's line bank — random-access single rows (`take`) out of a ~44 GB dataset; Lance is built for
that, parquet is slow at random access.

**Leakage-safe by construction:** text *and* writer-style crops are sampled only from the **train** split;
the held-out **`val.jsonl`** real lines are never read — they're the real test set for the augmentation
experiment (fine-tune TrOCR on *N* real vs *N* real + synthetic; judge CER on the real test set).

```bash
# the benchmarked fleet shape: co-locate one TrOCR reader with each generator (0.75 + 0.25 per GPU)
uv run diffu-runner --ckpt checkpoints/exp_sd35_128/step_120000/model.safetensors \
    --train-jsonl data_out_lejonet_128/train.jsonl --out synth_shards --n 1_000_000 \
    --gen-gpus 2 --gen-gpu-frac 0.75 --trocr-replicas 2 --trocr-gpu-frac 0.25 \
    --num-steps 12 --coherent --cer-max 0.30
```

### CLI

| flag | meaning |
|---|---|
| `--ckpt` / `--train-jsonl` / `--out` | Diffu checkpoint (`.safetensors`) · TRAIN split (never `val.jsonl`) · Lance output dir |
| `--n` | candidate count before gating (`--stream`), or exact clean target (`--exact-target`) |
| `--num-steps` | denoise steps per line (default 24); **12 is benchmark-validated equal quality** on the 128px model (paired ΔCER ≈ 0 vs 24) for ~1.5× generation speed — 8 visibly degrades letterforms |
| `--bucket-px` | width-bucket coarseness; **0 = off (the default, fixed-width canvas)** — bucketing is blocked by the current checkpoint's batch-composition glyph conditioning; set only for a future conditioning-fixed checkpoint |
| `--coherent` / `--random` | `--coherent` takes WHOLE documents — contiguous lines keyed `(doc_id, line_idx)`, one writer per doc — so the line bank can replay coherent prose runs; `--random` (default) pairs independent text × style. Only `--stream` honors it |
| `--exact-target` / `--stream` | `--exact-target` regenerates rejects with a fresh writer style until EXACTLY `n` clean lines exist; `--stream` (default) is one pass over `n` candidates and may undershoot if the reject rate is high |
| `--compile` / `--no-compile` | torch.compile the DiffuActor's MMDiT blocks (regional compile — the same win used at train/inference; on by default) |
| `--gen-dtype` | DiffuActor inference dtype (default `bfloat16`, the validated setting) |
| `--gen-gpus` / `--gen-gpu-frac` | DiffuActor replicas · GPU fraction per replica |
| `--trocr-replicas` / `--trocr-gpu-frac` | TrOCR gate replicas · GPU fraction per replica |
| `--cer-max` | drop lines read back worse than this CER (default 0.30) |

**Fleet shape (benchmarked 2026-07-13, 2× Blackwell):** co-locate readers WITH generators — every GPU at
`--gen-gpu-frac 0.75` plus one TrOCR reader at `--trocr-gpu-frac 0.25` beats dedicating a whole GPU to
gating by +28–39% throughput per GPU (the readers barely tax the generator).

## Page pipeline (bank-backed)

`diffu_runner.pages.run_pages` composes quality-gated, **coherent** document pages: one document's
consecutive text in one writer's hand, corpus-proportion templates + scan profiles from `diffu-page`
(the `page` extra: `diffu-runner[page]`). Two modes:

- **Render mode** (default): each `PageActor` pins a GPU, renders every line with Diffu, and reads a
  sample of the composed page back with TrOCR — the page is kept only while the fraction of bad lines
  (line CER > `line_cer_max=0.35`) stays under `max_bad_frac=0.12`.
- **Bank mode** (`PageConfig.bank` = a Lance line-bank path): every handwriting line comes
  **pre-rendered and pre-CER-gated** from the bank (`bank.LineBank`), so composition is **pure CPU**
  (`num_gpus=0`; `gpus` degrades to a plain worker count) — the 1M-page lever that turns ~79 GPU-days of
  re-rendering into about a CPU-day of seating. Each seated line keeps its own bank text as the label, the
  template mix is **prose-only** (a bank row is a whole sentence — seated in a register/ledger number cell
  it would shrink to microtext, so tabular pages stay on the render path), and the page gate is pure math
  over the rows' **stored** generation-time CERs. Build the bank with the line pipeline's `--coherent` so
  pages replay contiguous `(doc_id, line_idx)` runs; a keyless bank still composes, grouped by writer.

Entry point: `scripts/launch_pages.py` (pins ckpt / train split / paper dir; set `bank=` in its
`PageConfig` for bank mode), with env overrides `OUT`, `TARGET`, `GPUS`, `BATCH`, `SEED`:

```bash
OUT=synth_pages/pages TARGET=10000 GPUS=2 BATCH=24 SEED=1 uv run python scripts/launch_pages.py
```

Output is `img/page_*.png` + an append-mode `manifest.jsonl` (template, page CER, line count); a rerun
into the same `OUT` resumes from the pages already on disk. `PageConfig.upscale` adds a FINAL PiD 4×
super-resolution stage that reuses the GPUs freed after generation.

---

`DiffuActor`/`TrOCRActor` wrap the `diffu` package; `gate.py` and `bank.py` are pure logic (unit-tested
off-cluster, no ray imports).
