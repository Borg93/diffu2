# Handoff — continuing on the DGX Spark (GB10), 2026-10-19

State of `main` and what to run next on the DGX Spark. First written from a CPU-only cloud session; updated
the same day after the first GPU session on the GB10 and the cleanup refactor (section 1b).

## 1. What changed since the original diffu (`2b7ab3f`)

| Change | Status |
|---|---|
| `--style-canvas`: reference line left of the target on one canvas (extra input channels), its transcription in the glyph tokens, loss on target columns only, `--ref-dropout` + `--style-cfg-scale` | Implemented + unit-tested. **Untrained** — style quality unproven. |
| `--glyph-pad`: glyph tokens cover only the real ink width | Implemented + tested. Alignment error 212 → 4 px (measured). Quality untrained. |
| diffusers 0.41 / accelerate 1.15 / torch 2.14.1 / transformers 5.20.1 / peft 0.21.2 / trackio 0.42 | Done. |
| Attention through `dispatch_attention_fn` (`--attention-backend`) | Done. Benchmarked on the GB10: the default SDPA is fastest (section 3). |
| Fused QKV (inference), one QKV matmul under compile (training), bf16 Q/K after qk-norm | Done, outputs identical (re-checked on the GB10). |
| VAE single-frame fast path (3× less conv work) | Done. On the GB10: same output to ~1e-6 (fp32), encode+decode 1.65× faster. |
| Compile modes + SD3 Inductor flags + `--compile-vae` | Done. 0 graph breaks (`fullgraph=True`). A handful of recompiles remain (the last block's `context_pre_only`, optional args, the token-count `% 8` guard as widths vary — `PERF_AUDIT.md` §0b), within the raised `recompile_limit` of 256. |
| Losses read only at log steps (no per-step host sync) | Done. |
| DINOv2 fallback removed; FLUX.2 backbone experiment reverted | Done. |

Reports (private artifacts, share from the page): before/after measurements
<https://claude.ai/artifact/XAWWbmTauyQAkW84cDMRkf>, architecture review vs Qwen-Image 2.1 / FLUX.3
<https://claude.ai/artifact/6jZok7ZFYpTrPCatm3uLfM>.

### 1b. The 2026-10-19 refactor

- **One content encoder:** the Unifont glyph-line encoder (column-aligned tokens + shared-column RoPE) is the
  only content path; the `--glyph-line` flag is gone because it is always on. Removed: per-character content,
  `--glyph-concat`, style tokens in the attention context (`--no-style-tokens` / `--style-tokens`; dino style
  is pooled-only), `rope=False`, the pre-Unifont font path (pre-Unifont / SD3.0-era checkpoints no longer
  load), and `vae.finetune_decoder_only` (a no-op: the VAE is frozen whole).
- **Two style modes:** `dino` (default; every model trained so far) and `canvas` (`--style-canvas`).
  `--glyph-pad` puts the glyph tokens over each line's own ink extent (a length control).
- **Canvas fixes:** a dropped reference (`--ref-dropout` / `--cond-dropout`) now leaves its canvas columns pure
  noise in training. Sampling denoises only the target and rebuilds the reference columns every step as the
  reference noised to the current level. The style-guidance null branch zeroes the pooled vector (the
  `--cond-dropout` mode). With `--cfg-scale 0`, style guidance keeps the text (`v_text + s·(v_full − v_text)`)
  and skips the null pass. A single reference is VAE-encoded once (posterior mode) at inference.
- **REPA target:** DINOv3 ViT-L/16 sees the target line at its native size — its 16-px patches are the DiT
  token grid, no 224² squash or interpolation. Canvas mode builds only the DINOv3 features REPA needs.
- **Defaults:** the ink-focal loss is on (`--no-ink-focal`; config fields `aux.ink_focal` / `aux.ink_weight`).
  One standard HTR recognizer for the live gen_CER, `diffu-eval` and `diffu-vae-gate`:
  `Riksarkivet/trocr-large-handwritten-hist-swe-3-char` (`cfg.aux.htr_recognizer_eval`). `diffu-data --height`
  and the dataset default to 128.
- **Shared architecture flags** (`config.add_arch_args`), the same on `diffu-train`, `diffu-generate`,
  `diffu-synth` and `diffu-smoke`: `--line-height`, `--style-canvas`, `--glyph-pad`, `--no-fill-ratio`,
  `--max-line-width`, `--max-ref-width`, `--max-chars`.
- **Widths:** `--max-line-width` (default 1024) is a memory cap, the model has no width limit;
  `--drop-wide-lines` skips wider lines instead of squashing them; canvas references prefer same-writer lines
  that fit `--max-ref-width`; `diffu-generate --width` is the canvas width for dino models and the written
  length (ink extent) for canvas models.
- **Checkpoints:** every checkpoint dir has `config.json` (the full `Config`). `step_N/` = resumable accelerate
  state + `ema_model.safetensors` (written by training itself). `final/model.safetensors` = the EMA-applied
  export without the frozen VAE / DINOv3 weights (they load from the Hub). Every loader reads `config.json`
  (`diffu.generate.load_config` + `build_model`); arch flags override it; a checkpoint without one needs its
  arch flags passed.
- **Training:** `--compile` works with gradient checkpointing too: Dynamo's config is per-thread, and the
  checkpoint recompute on autograd's thread once hit the default recompile limit of 8 and hard-failed a few
  steps in, so the backbone now re-applies the compile settings on whatever thread runs a compiled block
  (verified: 80 IAM steps with val + samples every 20). Without checkpointing is still faster. trackio
  `>=0.42` (0.27 broke against huggingface_hub 1.33): `--track`, alerts, `--track-space` (private Space).
  `--profile-steps` writes `profile_step_table.txt` + `trace.json`; `--profile-stack` replaces the
  `DIFFU_PROFILE_STACK` env var; the profile table no longer double-counts GPU time. img/s includes
  `--grad-accum`. LoRA runs can't `--resume` (rejected up front). `--resume auto` continues from the newest
  complete `step_N` in `--out-dir`, or starts fresh.
- **Data:** `scripts/prepare_iam_writers.py` builds `data_iam/` (IAM with real writer ids, `TRAINING.md` §3);
  `crop_line` clamps to the page (no black borders); the width cache stores raw widths.
- **Removed:** `--cache-latents` / `diffu-cache-latents` / `data/latent_cache.py` (never used in a recorded
  run), `src/diffu/rl`,
  `src/diffu/quantize.py` + `scripts/bench_quant.py`, and the one-off experiment scripts (`diag_*`, `exp_*`,
  `harmonize_synth`, `sample_iam_texts`, `targeted_synth`, `prepare_iam_full`, `profile_tensorcore`,
  `get_unifont`, `gallery.py`) — all in git history at `c53d30e`.
- **Model release:** `Gabriel/diffu_sd35_128` is no longer on the Hub, and its `step_120000` checkpoint stayed
  on the old box. There is no pretrained model to start from: the canvas run below trains from scratch.

## 2. Setup on the GB10

The GB10 is aarch64 with one Blackwell GPU (index 0, sm_121) and 119 GB of unified memory. torch comes from
the cu130 index in `pyproject.toml`. The Makefile defaults are GPU 0 and one process, so no overrides are needed
here (`make train` adds `--multi_gpu` only when `NPROC>1`; new target: `make smoke-canvas`).

```bash
git clone https://github.com/Borg93/diffu2.git && cd diffu2
uv sync --extra dev --extra track --extra hf     # a plain `uv sync` drops the extras
uv run python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
# -> 2.14.1+cu130 NVIDIA GB10 (12, 1)
hf auth login                                   # DINOv3 (REPA target / dino mode) is gated
hf download microsoft/trocr-base-handwritten    # the gen_CER gauge loads its recognizer from the local cache only
```

## 3. Run in this order

**Done on the GB10 (2026-10-19):**
1. Tests: `make test`.
2. Smoke: `make smoke-1gpu`, then `make smoke-compile` (`smoke OK`, REPA on).
3. Attention backends, `scripts/bench_attention.py --canvas --compile --width 1024` (batch 8, bf16): the
   default SDPA (`native`) is the fastest correct backend for inference (165 ms) and training (523 ms);
   `_native_flash` / `_native_cudnn` / `_native_efficient` are correct but 2–11 % slower; the flash, sage and
   Hub-kernel backends aren't installed. → Keep the default (no `--attention-backend`).

   Also measured: compiled training forward+backward 1.41× faster than eager (batch 4 at 1024 px, 36 s cold
   compile); the VAE single-frame path 1.65× (encode+decode). max-autotune compile modes can't autotune the
   GEMMs on the GB10 (Inductor: "Not enough SMs to use max_autotune_gemm mode"), so use plain `--compile`.

**Next:**

4. The canvas retrain on IAM with real writer ids (`data_iam/`, `TRAINING.md` §3). IAM is English, so the live
   gauge needs an English recognizer (downloaded in section 2):
   ```bash
   uv run diffu-train --data-dir data_iam --out-dir runs/canvas --style-canvas --glyph-pad \
       --text-dropout 0.1 --ref-dropout 0.2 --cond-dropout 0.05 \
       --max-line-width 3072 --max-ref-width 2048 --drop-wide-lines \
       --bucket-width --no-grad-checkpoint --compile --batch-size 8 \
       --htr-recognizer-eval microsoft/trocr-base-handwritten --track
   ```
   `--drop-wide-lines` skips the 245 train / 34 val lines wider than 3072 px. Monitor with
   `uv run trackio show --project diffu`, `uv run trackio list alerts --project diffu --json --since <ts>` and
   `uv run trackio get metric --project diffu --run canvas --metric val_loss --json`; the sample sheets land in
   `runs/canvas/samples/`. Defaults: 100000 steps, a `step_N/` every 10000 (`--save-every`).
5. Sample (the checkpoint's `config.json` supplies the arch flags):
   ```bash
   uv run diffu-generate --ckpt runs/canvas/step_N/ema_model.safetensors \
       --style ref.png --style-text "what ref.png says" --text "..." \
       --cfg-scale 5 --style-cfg-scale 2
   ```
   For a canvas model `--width` sets the written length (ink extent); without it the length comes from the
   reference's px-per-char.
6. Optional: profile a few real steps and keep the trace for the `profile-model` skill — add
   `--profile-steps 20` to the step-4 command with another `--out-dir` (it writes `profile_step_table.txt` +
   `trace.json` there and exits before training).

PyTorch agent skills used in the review: `npx skills add meta-pytorch/skills` (profile-model,
debug-graph-breaks, analyze-memory-snapshot).

## 4. Open risks

- **`writer_id` is the page id** in the Swedish PAGE-XML data: the reference shares the target's paper, so the
  model can learn to copy the paper instead of the hand. Lines from other pages of the same writer, or
  writer-labelled data, fix this. For IAM, `data_iam/` now carries real IAM writer ids, so its references are
  lines by the same writer from any of their forms; the Swedish page data still has the problem.
- **Copying the reference's letters** instead of writing the target text. Check with a swap test: same
  reference, different texts → the text must change; same text, different references → the hand must change.
- **No style metric yet.** Without one, style can't be compared between runs.

## 5. Why the backbone has no shared modulation

`SD3Transformer2DModel` hard-codes an `AdaLayerNormZero` per block and per stream (DiT's adaLN-Zero, which SD3
kept). We use the diffusers class as-is, so we inherited it: 24 blocks × 2 streams × (1024 → 6·1024) ≈ 300 M
of the 900 M parameters, all mapping the same timestep + fill-ratio vector. It is not configurable in that
class. Newer models compute modulation once and share it (FLUX.2, Qwen-Image 2.1, FLUX.3: ~1–3 % of params).
At our width and token count, a `Flux3Transformer2DModel` stack with the same FLOPs (821 vs 825 GFLOP per
forward) has 378 M parameters — measured; whether it writes better is untested.

## 6. Next decisions (not implemented)

No retrain needed:
- Sampling shift sweep: `FlowMatchEulerDiscreteScheduler()` in `model/diffu.py` uses shift 1. Try 1.5 / 2 / 3
  on CER (SD3's `calculate_shift` gives ≈1.9 at 1,024 image tokens, e.g. a 2048-px canvas at 128 px).
- `UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True)` at 12–16 steps vs Euler 24.
- CFG-Zero* / APG guidance maths (diffusers guiders), computed on the target columns only.
- Skip diffusers' cache hooks: FirstBlockCache crashes on SD3's last block; MagCache / TaylorSeer break our
  `fullgraph=True` regional compile.

With a retrain:
- Reference as clean tokens at timestep 0 (FLUX.3 / Qwen-Edit) instead of noised canvas columns.
- Equal-FLOPs A/B: SD3 24 dual blocks vs `Flux3Transformer2DModel` (2 per-mode + 20 joint, width 1024),
  same data, judged on held-out-writer CER + a style metric. `Flux3Transformer2DModel` is in diffusers main
  only until 0.42 ships.
- The external baseline worth beating: FLUX.3 Klein base + LoRA (diffusers ships the DreamBooth scripts) on
  the same lines, judged the same way.
