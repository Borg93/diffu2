# Diffu Performance Audit — Stage-1 Training Hot Path

> **UPDATE 2026-06-29 — the audit was executed and acted on.** GPU profiling, A/B micro-benchmarks, and a
> `torch.compile` graph-break/recompile study were run on a live GPU; the safe, behavior-neutral wins are
> implemented and `exp_sd35_inkfocal` was relaunched on them. **Read §0 (what shipped + measured) and §0b
> (the compile study) first.** §1–§5 are the original static-analysis pass, kept for reference; their ranked
> table is still valid, with current status tracked in §0 / the note atop §2.

> **Note (2026-10-19):** (1) Until 2026-10-19 the `profiling.py` report summed every profiler row with self
> GPU time — the aten ops *and* the kernels they launched (plus the step annotations) — so it double-counted
> GPU time. Shares below that came from `profile_step_table.txt` (the 14.6 % Tensor-Core share, `aten::copy_`
> 20.5 %, VAE `conv3d` 13.4 %, the ~13 % VAE-encode share in §0e) are suspect; the numbers are kept as
> recorded. Wall-clock and img/s measurements are unaffected. (2) Removed since: `--cache-latents` /
> `data/latent_cache.py`, `diffu.quantize` + `scripts/bench_quant.py`, and `scripts/profile_tensorcore.py` — all
> in git history at `c53d30e`. The `DIFFU_PROFILE_STACK=1` env var is now `diffu-train --profile-stack`, and
> `--compile` now requires `--no-grad-checkpoint`. (3) Measurements on the DGX Spark (GB10) are in
> [`HANDOFF.md`](HANDOFF.md) §3.

## 0. IMPLEMENTED & MEASURED (2026-06-29) — what shipped

**The bottleneck, confirmed twice.** The original DCGM read (TENSO ~0.25) and a fresh `torch.profiler` run
agree: the Stage-1 step is **overhead / launch-bound, not math-bound**. In the profiler kernel table only
**14.6%** of CUDA time is Tensor-Core-eligible GEMM/conv; the rest is `aten::copy_` (20.5%, ~45k tiny calls),
the frozen VAE `conv3d` (13.4%), and elementwise (mul/add/LayerNorm/RoPE). **Tensor cores are already in use**
(bf16 cutlass kernels in the trace); 14.6% is *dilution* by overhead + small GEMMs, not TC being off. So every
win below **removes non-math work per step — it does not change the model, the math, or the predictions.**
*(2026-10-19: these percentages come from the old, double-counting profile report — see the note at the top.)*

All items are **behavior-neutral** (bit-identical, or within-bf16 noise) unless noted; validated with ruff +
pyrefly + a smoke run. The inference ones are identical-by-construction.

| # | Change (file) | What it does (pedagogical) | Improves | Measured | Status |
|---|---|---|---|---|---|
| T1 | **Glyph-render cache** — `content_encoder.py` `UnifontRenderer.render_char` → instance dict | Each glyph bitmap is a pure fn of (char,font,size). Before: PIL re-rasterized every char of every line **every step** on the main GPU process (stalls kernel launch). After: rasterize once, dict-lookup after. | throughput (un-stalls GPU); small CPU | **1.8×** on the call (4.00→2.26 ms) | ✅ live |
| T2 | **fused AdamW** — `train.py` `fused=cuda` | One fused kernel for the optimizer step vs many tiny per-tensor kernels → fewer launches. | throughput (opt step) | **1.97×** (8.41→4.27 ms) | ✅ live |
| T3 | **foreach EMA** — `train.py` `EMAModel(foreach=True)` | EMA update vectorized (`torch._foreach_`) vs a Python per-tensor loop. | throughput (EMA), small | math-identical (Δ1.2e-7) | ✅ live |
| T4 | **DDP bucket-view + no buffer broadcast** — `train.py` | Grads alias DDP reduction buckets (1 fewer copy + lower peak mem); skip per-step buffer broadcast (safe: BN uses local minibatch, seed-identical init). | mem (peak) + throughput (multi-GPU) | — | ✅ live |
| T5 | **Async H2D** — `train.py` `DataLoaderConfiguration(non_blocking=True)` | Next batch's host→device copy overlaps current compute (pin_memory already on). | latency (small) | — | ✅ live |
| T6 | **Frozen-DINO eval-pin** — `conditioning.py` `StyleEncoder.train()` | Frozen teacher stays `eval()` regardless of `model.train()` → deterministic style/REPA features. | correctness hygiene | inert today | ✅ live |
| **T7** | **`--no-grad-checkpoint`** (launch flag; now the run default) | Grad checkpointing trades ~20% recompute to save activation memory we don't need at 96 GB. Off = no recompute, **identical gradients**. | **throughput** ↑ for **mem** ↑ (we have it) | **+15%** (54→63 img/s, isolated A/B) | ✅ live |
| I1 | **Batched CFG** — `diffu.py` `_denoise_latents` | CFG ran the denoiser **twice/step** (cond, uncond). Now one 2×B forward — RoPE/SDPA are per-sample → numerically identical. | inference latency/throughput + GPU fill | **1.87×** (39.7→21.3 ms/step) | ✅ live |
| I2 | **`inference_mode`** — `diffu.py`, `generate.py` | Skips autograd bookkeeping on the pure-inference sampling path. | inference latency (small) | — | ✅ live |
| E1 | **`--resume`** — `train.py` (`accel.load_state`) | True resume of model+optimizer+LR+EMA+step (vs `--init-from` = weights only). Found+fixed an EMA-shadow-on-CPU device crash. | enabler (correct continuation) | verified (loss 0.057, LR correct) | ✅ live |
| E2 | **Regional `torch.compile`** — `train.py` behind `--compile` (OFF) | Compiles 1 repeated `JointTransformerBlock` (not 24); `fullgraph=True` (0 graph breaks); `recompile_limit` raised. | throughput (fuses copy/elementwise) | +22% single-GPU; **recompile-thrashes on the multi-width corpus → deferred** (§0b) | ⚠️ experiment |

**Net:** the live `exp_sd35_inkfocal` resume runs T1–T7 (+ the I-paths drive the live-gen_CER sampling / deploy).
Guaranteed behavior-neutral gain ≈ **+15%** (`--no-grad-checkpoint`) on top of the T1–T6 launch-overhead cuts.
**Bigger batch was deliberately NOT used** — it changes effective-batch = a *training-behavior* change, which
violates the "latency/throughput only" constraint. (Live 3-GPU img/s is currently confounded by a co-resident
second session on GPUs 0 & 2; the isolated single-GPU A/B numbers above are the clean measurements.)

## 0a. Width bucketing (`--bucket-width`, `data/sampler.py`) — implemented + measured 2026-06-29

The **#1 lever** (PERF_AUDIT §2 #1). `WidthBucketBatchSampler` groups similar-width lines into each batch
("smart batching": shuffle → megabatch → sort-by-width → batch → shuffle batch order). **DDP-safe**:
accelerate's `BatchSamplerShard` splits the (identically-seeded) global batch list across ranks, and DDP
all-reduces param-shaped grads, so ranks may hold different-width batches on the same step. **Opt-in** — it
makes batches width-correlated (not i.i.d.), a small but real training-dynamics change, so validate gen_CER
before defaulting. Covered by `tests/test_sampler.py` (6 tests: homogeneity, DDP determinism, coverage, reshuffle).

- **Padding waste (offline sim, B32): 54.5% → 1.3%** — reclaims ~half the wasted VAE+DiT width-compute. THIS is
  bucketing's value, and it's independent of compile.
- **Measured throughput: +20%** (61 → 74 img/s, single-GPU 20k-subset A/B, no compile, same config ± `--bucket-width`).
  The full corpus (wider width spread) should gain more. Stacks with `--no-grad-checkpoint` (T7, +15%).
- **DDP-correctness: VERIFIED** (adversarial workflow check vs the committed code). accelerate's
  `BatchSamplerShard` shards the identically-seeded global batch list `idx % world == rank`; `drop_last` →
  equal per-rank counts (no hang); the in-`__iter__` epoch increment is the lockstep driver (accelerate's
  `set_epoch` doesn't reach a custom `batch_sampler`, so this is the sole, consistent reshuffle).
- **Compile-unlock: NO (verified, corrects my earlier optimism).** The megabatch-sort pads each batch to its
  OWN max → ~50 distinct collate widths (round-up-16), not ~8 → `--compile` still cold-recompiles ~50× (it
  survives under `recompile_limit=256`, but there's no steady-state-zero). Truly unlocking compile needs a
  coarser **fixed-px-bucket** variant (bucket key == collate pad target, e.g. `round_up(w,128)` → ~8 shapes) —
  but that RE-adds ~10–20% padding for narrow lines, trading back most of the win. Net: keep `--bucket-width`
  for the padding reclaim; treat `--compile` as separately deferred.
- **Caveats:** (1) `--white-pad-prob > 0` pads lines AFTER bucketing → fragments shapes; keep it 0 when
  bucketing (CLI default 0.0; the inkfocal run uses 0). (2) `--resume` restarts the sampler epoch at 0 (not in
  `accelerate save_state`) → data order differs after a resume; cross-rank correctness is unaffected.

## 0b. torch.compile study (2026-06-29) — graph breaks vs recompiles

Run with the `meta-pytorch/skills` **`debug-graph-breaks`** skill + `TORCH_LOGS="graph_breaks,recompiles"`.

- **Graph breaks: 0.** `fullgraph=True` *passes* — the custom `RoPEJointAttnProcessor` (apply_rotary_emb on a
  slice + `torch.cat`, qk RMSNorm, SDPA) traces into one graph. Nothing to fix on the graph-break axis.
- **Recompiles: the real problem (~39+, even with `dynamic=True`).** `TORCH_LOGS=recompiles` named 4 guards:
  1. **`(encoder_hidden_states.size(1) + image_rotary_emb[0].size(0)) % 8 != 0`** — a Tensor-Core **seq-length
     alignment guard**; as line **width** varies the image-token count varies and crosses 8-multiples →
     recompile. **This is the width-driven recompiler.**
  2. **`context_pre_only == True` / `KeyError 'to_add_out'`** — SD3's **last** block has `context_pre_only=True`
     (no `to_add_out`); the other 23 don't → 2 bounded block variants (expected).
  3. **`'NoneType'.stride`** — an optional rotary/mask arg toggling None vs tensor.
- **Why it crashed first:** default `recompile_limit=8` < corpus width spread → Dynamo aborts. We raised
  `recompile_limit`/`accumulated_recompile_limit` in the `--compile` branch — that *survives* the thrash but
  doesn't *stop* it (each new width still recompiles).
- **The fix = width bucketing (§2 #1):** quantize widths to a few buckets → the `%8` guard + token count
  stabilize → recompiles converge to ≈(num_buckets × 2 context variants) and go quiet. It *also* deletes the
  20–40% padding-FLOP waste. **One change unlocks both compile and a big throughput chunk.** Until then,
  `--compile` stays OFF for the long run.
- **Tooling (answer to "do we have the right tooling?"):** yes — `TORCH_LOGS="graph_breaks,recompiles"`, the
  `debug-graph-breaks` skill, `torch._dynamo.config.recompile_limit`, `torch._dynamo.explain()`,
  `TORCH_COMPILE_DEBUG=1`; plus the repo's `--profile-steps` (torch.profiler) and the `profile-model` skill.

## 0c. Free inference sweep (2026-06-29) — cfg_scale & steps (quality/legibility, not perf)

Measured gen_CER on 12 val lines (step_160000, Swedish-Lion recognizer):

| | cfg=3 | cfg=5 | cfg=7 |
|---|---|---|---|
| 24 steps | 0.328 (old default) | **0.253** | 0.264 |
| 48 steps | 0.290 | 0.270 | 0.285 |

- **`cfg_scale` 3 → 5 = ~23% relative gen_CER win** (0.328 → 0.253), free, inference-only. cfg=7 over-saturates.
  **Applied:** `generate.py --cfg-scale` default and the train live-eval scale are now 5.0.
- **More steps do NOT help** — 48 ≥ 24 in CER at every cfg → keep 24 (don't pay for more). The lever here is CFG,
  not step count.
- 100%-safe (no retrain) — raises legibility of what ships without touching weights.

## 0d. HTA deep-dive (2026-06-29) — the optimized step is 50% GPU-idle, host-bound

Full Holistic Trace Analysis on a chrome-trace of the optimized config (`--bucket-width --no-grad-checkpoint`,
batch 32, 1 GPU; `profiling.py` now exports the trace):
- **Temporal: GPU idle 50.8% · compute 48.1% · non-compute 1.1%** — half the wall-clock the GPU does nothing.
- **Idle breakdown: 91% host_wait · 8% kernel_wait · 1% other** — the idle is the GPU waiting on the CPU (host),
  confirming **host/launch-bound** (not memory- or compute-bound).
- **Kernel mix when busy: 97.6% COMPUTATION · 2.4% MEMORY** — the GPU work itself is efficient (real GEMM); the
  problem is the GPU being STARVED, not wasteful kernels.
- **Implication:** remaining throughput is gated by HOST overhead (kernel launches + Python dispatch). Levers:
  (a) `torch.compile` (fuse → fewer launches; deferred, recompile-bound — §0b), (b) **bigger batch** (more GPU work
  per launch → better overlap → fills the idle; but changes eff-batch/LR). Both are trade-offs; the proven recipe
  (eff-batch 96, lr 1e-4) + banked wins is the safe launch config, with batch-bump/compile as post-launch A/Bs.

## 0e. torch.compile VALIDATED (2026-06-29) — the fix for the host-bound storm; GPU idle 50%→6%

The §0d host-bound diagnosis was confirmed by three independent trace analyses: (1) kernel histogram — ~13.3k
kernels/step, 67% < 10us, 58,971 `cudaLaunchKernel`; (2) cast-parent attribution (interval-nesting on the trace) —
of ~64.7k casts, 25.5k rooted under `aten::linear` (autocast fp32→bf16 per layer), 19.1k under `ToCopyBackward0`;
(3) `with_stack` — the casts have NO Python line → inserted by **autocast at the C++ dispatcher**, not our code. So
the storm is intrinsic eager-mode mixed precision + per-layer RoPE/norm glue (≈96 `apply_rotary_emb` fp32 casts/fwd),
**not a deletable bug**. The cure is operator fusion.

Enabling the existing regional-compile branch (`--compile`: `compile_repeated_blocks` on `JointTransformerBlock`,
`dynamic=True`, `fullgraph=True`) WITH `--bucket-width` (widths → a handful of shapes → recompiles bounded) ran
clean (0 graph breaks) and measured (batch 32, 1 GPU, data_out_lejonet_sub):

| per step                | eager | compile | Δ      |
| ----------------------- | ----- | ------- | ------ |
| cudaLaunchKernel (host) | 5,897 | 1,750   | −70%   |
| aten cast ops           | 6,466 | 2,048   | −68%   |
| fused Triton kernels    | 0     | 14,832  | (37%)  |

- **GPU idle 50% → ~6%** — steady-state `nvidia-smi dmon` SM = **94%** (was ~48% compute). NB: the compile *trace*
  reports 86% idle — a **profiling artifact** (Dynamo compiles during the profiled steps); trust dmon/throughput.
- **End-to-end A/B:** OLD (grad-ckpt on, no bucket, no compile) ~61 img/s → NEW (`--no-grad-checkpoint
  --bucket-width --compile`) ~116 img/s = **~1.9×** (and this undercounts vs the true weekend run — the code wins
  are in both sides).

Mechanism: Inductor fuses the cast/RoPE/norm elementwise glue into Triton kernels (HF "Profiling in PyTorch" Part 2),
cutting HOST launch work 70% — exactly the host_wait that starved the GPU. Same math (fused), no training-dynamics
change. **REGIME SHIFT: the step is now compute-bound (94% busy)** — host-side micro-opts (cast reduction, the
`backbone.py` `_rope` memoize shipped here) no longer move the needle; the next lever is cutting COMPUTE (latent
caching `latent_cache.py` to skip VAE encode ~13% CUDA — needs gen_CER validation; DINO feature caching; bigger
batch). **RECOMMENDATION: add `--compile` to the from-scratch run.** Supersedes §0b's "compile deferred". Profiler
now exports a chrome trace + opt-in `DIFFU_PROFILE_STACK=1` with_stack attribution table (`profiling.py`).
*(2026-10-19: `latent_cache.py` / `--cache-latents` removed — never used in a recorded run; in git history at
`c53d30e`. `DIFFU_PROFILE_STACK=1` is now `--profile-stack`.)*

## 0f. torchao quantization VERDICT (2026-07-02) — measured LOSS at inference; ship compiled bf16

CER-gated bench (`scripts/bench_quant.py` — removed; in git history at `c53d30e`; step_280000, 6 val lines, cfg=5, Blackwell CC 12.0,
torchao 0.17, regional compile on all variants):

| variant | ms/line | speed | CER |
|---|---|---|---|
| **bf16 compiled** | **367** | **1.00×** | 0.047 |
| int8 weight-only | 2605 | 0.14× | 0.015 |
| fp8dq | 672 | 0.55× | 0.030 |

- **Accuracy cost ≈ zero** (CER spread is 6-line sampling noise) — quantization does not hurt legibility.
- **Speed is a loss**: our GEMMs are small (1.3B model, short line canvases) → not memory-bound → weight-
  quant overhead dominates; int8's 0.17-default config also takes a deprecated legacy kernel path.
- **Decision: ship compiled bf16** (~0.37 s/line vs ~1 s eager). No quantized artifacts are stored or
  published; `diffu.quantize` (int8 v2 / fp8, `set_inductor_config=False`) remains a one-line opt-in for
  consumers on hardware where it pays. *(2026-10-19: `diffu.quantize` removed; in git history at `c53d30e`.)*
- Methodology notes (audit-driven): per-variant reseed (ΔCER measures quantization, not seeds), globals
  set once up front (torchao configs no longer flip TF32/inductor at quantize time), warmup covers every
  width before timing.

## 1. Live measurement context

While the unattended overnight run was active, DCGM reported, at **100% GPU utilization**:

| Counter | Value | Meaning |
|---|---|---|
| **TENSO** (tensor-active) | **~0.25** | Tensor Cores doing HMMA math only ~25% of the time. |
| **SMACT** (SM-active) | **~0.80** | SMs are busy ~80% of the time — kernels *are* running. |
| **DRAMA** (DRAM-active) | **~0.44** | Memory pipe ~44% busy. |
| Util | 100% | The "anything launched" counter — uninformative on its own. |
| Power / thermal | **capped** (~83 GB/97 GB, 87 °C) | The cards are pinned at their power+thermal limit. |

**What this means.** 100% util with **high SMACT (0.80) but low TENSO (0.25)** is the textbook *"the GPU is busy, but not doing the math you care about"* signature from the HF *Profiling in PyTorch* posts. The SMs are saturated by **memory-bound / launch-bound work** — elementwise, LayerNorm/SiLU/GELU, softmax, glyph rasterization, H2D copies, padding — not by Tensor-Core GEMMs. DRAMA 0.44 confirms a meaningful memory-traffic component. Because the cards are already power/thermally capped, **the only way to more throughput is to do less non-math work per step**: stop repeating frozen/constant work, cut wasted padding FLOPs, and fuse/stabilize kernels so a larger share of the (capped) power budget lands on the tensor pipe. Every ranked item below targets that gap.

## 2. Ranked actions (original audit; statuses updated 2026-06-29)

Sorted by corrected impact, then by lowest effort. All impacts are the **adversarially-corrected** values.

> **Status (2026-06-29):** #2 glyph render (dict cache = T1 ✅; GPU atlas still pending) · #5 TF32 ✅ (present) ·
> #6 `torch.compile` ✅ implemented behind `--compile` but **recompile-bound until #1 lands** (§0b) ·
> #11 non-blocking H2D ✅ (T5) · **`--no-grad-checkpoint` ✅ (T7)** · **#1 width bucketing ✅ (§0a, `--bucket-width`,
> measured +20%, DDP-verified)**. **#3 VAE latent cache ✅ IMPLEMENTED 2026-07-15** (`--cache-latents`, opt-in;
> measured **1.36×** / 32.0→43.6 img/s, the encode was ~23% of the step; default path byte-identical; gate gen_CER
> before defaulting). **Top REMAINING win is now #2 (GPU glyph atlas)**, then #4/#8 (DINOv3 feature cache — REPA is
> now ON by default, so #8's REPA-target cache applies). NB: #1 does NOT unlock `torch.compile` (~50 shapes, not ~8 — §0a/§0b).
> *(2026-10-19: #3's `--cache-latents` was removed — never used in a recorded run; in git history at `c53d30e`.
> The per-character content tokens and the style tokens that #6/#7 describe no longer exist: the glyph-line
> encoder makes one content token per patch column, and style is pooled-only.)*
>
> **Empirical (2026-06-29, 3000-line sample of data_out_lejonet):** width p50=556 / p90=881 / max=2500 px →
> **random-batch padding waste @B32 = 54.5%** of width-FLOPs vs **1.3% bucketed** → width bucketing roughly
> **HALVES wasted VAE+DiT width-compute** (the single biggest lever, bigger than T7's +15%) AND collapses the
> ~hundreds of distinct shapes into ~9 buckets (unlocks `torch.compile`, §0b). Frozen VAE encode =
> 65/134/202 ms/batch at W=256/512/768 → #3 latent caching removes ~145 ms/batch at the mean width.

| # | Title | Code location | Article lesson | Concrete change | Impact | Effort | Risk | Caveat |
|---|---|---|---|---|---|---|---|---|
| 1 | **Bucket collate by width** (kills padding waste + shape churn) | `data/collate.py:33`; `train.py` `build_dataloader` (~448-457, `shuffle=True`) | "Profile first, then stop wasting FLOPs": padding to batch-max is pure wasted VAE+DiT compute, and per-step shape churn defeats `torch.compile`. | Add a `width` field in `build_manifest` (crops are fixed-height=64, width is free at crop time); use a DDP/accelerate-aware `batch_sampler` drawing each batch from one width bucket, shuffling bucket order per epoch. Cuts intra-batch pad (often 20-40% of width) and collapses distinct latent shapes from hundreds to ~num_buckets. | **High** | Med | Med | Sampler must be accelerate/DDP-aware (or `split_batches`); re-apply the `len(ds) ≥ batch*world` guard per bucket with a drop/merge policy for small buckets. |
| 2 | **Glyph PIL render on the GPU main thread** every step | `model/content_encoder.py:37-48, 86-91`; `model/diffu.py:65,89` | "Stop overhead-bound starvation": hundreds of Python/PIL `Image.new`+`ImageDraw.text`+`np.asarray` calls per step sit on the synchronous forward path (NOT the dataloader — `num_workers` can't help), feeding a blocking H2D copy before the CNN. | Precompute a GPU-resident glyph atlas `[num_codepoints,1,S,S]` once at init and index it by codepoint (zero PIL, no per-step H2D). Minimum: `lru_cache`/dict on `render_char`. | **Med** | Med | Low | Deterministic per char (fixed font+size, no aug) → numerically identical. The GlyphCNN ResNet18 over B×96 glyphs still runs; this removes the CPU stall + copy, not the CNN. |
| 3 | **VAE encoder runs every step** on un-augmented images | `model/diffu.py:84`; `model/vae.py:84-100`; `data/dataset.py:236-244` | "Don't make the GPU repeat constant work": a frozen f8 conv encoder over deterministic inputs is pure overhead — precompute once. | Pre-encode every line once with the frozen VAE, store normalized latent sidecars (bf16) keyed to manifest rows; dataset/collate load the latent instead of calling `vae.encode`. Encoder frozen + `@no_grad`, Stage-1 never decodes → loss-equivalent. | **Med** | Med | Med | Cache the posterior **mean** or a fixed-seed `sample()` (freezes one reparam draw). Pad latents in latent space with the encoder's output **for white paper (nonzero)** — zero-padding is wrong (encoder is nonlinear+biased), or cache at fixed max width. **Gate behind a flag**: enabling image aug later invalidates it. |
| 4 | **DINOv3 style ViT-L runs frozen** every step | `model/conditioning.py:121,125-133`; `model/diffu.py:66,89`; `data/dataset.py:223-234,245` | Same lesson: a frozen ViT-scale feature extractor re-running over a finite input set is redundant. Cache the constant part, keep the trainable Perceiver in the graph. | Precompute DINOv3 `last_hidden_state` per **unique** image (per-step `random.choice` of style peer only changes *which* cached tensor is looked up); feed cached patch tokens into the still-trainable resampler. | **Med** | Med | Med | Cache is **~120-150 GB** fp16 memmap (~299k lines × ~196-256 tokens × 1024). **Key the cache by `cfg.cond.style_encoder`**, so a changed encoder never serves stale features. |
| 5 | **TF32 for residual fp32 matmuls** | none in `src/` (add at process start, ~`train.py:295` after `Accelerator`) | "Move stray fp32 GEMMs onto Tensor Cores": fp32 matmuls fall back to FP32 CUDA cores (no HMMA) unless TF32 is on. | `torch.set_float32_matmul_precision("high")` (+`allow_tf32` flags). | **Low** | Low | V.low | Most of the model's GEMMs are *already* bf16 under accelerate's autocast (which wraps `forward` through `no_grad`), so the rescued surface is small — mainly non-autocast eval/generate/vae-gate paths. Safe one-liner; negligible recon-CER risk. |
| 6 | **`torch.compile` the SD3 MMDiT, shape-robust** | `train.py:171` (bare `torch.compile`); `data/collate.py:33` | "compile fuses pointwise chains (AdaLN/SiLU/GELU/residual) into Triton and removes dispatch gaps — but specializes per shape": directly attacks low TENSO. | `dynamic=True` or `mark_dynamic(latent, 3)` on width only (content len fixed at 96, style tokens fixed — encoder seq dim need *not* be dynamic); combine with the width bucketing in #1. | **Med** | Med | Med | Without dynamic/bucketing, compile recompiles per padded width → thrash, possibly slower than eager. The REPA forward-hook (`diffu.py:125-141`) breaks graphs **if `--repa` is on** (off by default). Compile *after* `enable_gradient_checkpointing` (already true). |
| 7 | **Glyph ResNet18 over B×96 tiny 32×32 glyphs** is low arithmetic-intensity | `model/content_encoder.py:50-66, 86-91` | "Many small low-AI ops keep SMs busy but starve the tensor pipe": dedup redundant compute, right-size layers. | Within-a-step dedup unique chars + `index_select`; consider a purpose-built 3-4 block stem instead of ImageNet ResNet18 for 32×32. | **Low** | Med | Low | Feature cache must be **intra-step only** (`glyph_content` is trainable, `train.py:52`). Bitmap caching always safe but is a CPU win (doesn't raise TENSO). Stem swap re-parameterizes → fresh run, not a drop-in. |
| 8 | **REPA target: frozen DINO re-encodes target image** (only when `--repa`) | `model/diffu.py:143-150` | Another frozen-encoder-on-constant-input redundancy; precompute the alignment target. | Cache pooled `[B,dino_dim]` target vector keyed by image path; bundle with #3's offline pass. | **Low** | Med | Low | **REPA is off by default** (`config.py:67`) → zero saving on the current run. Target is genuinely constant (style randomness doesn't touch it). |
| 9 | **EMA + `.item()` sync run every micro-step** under grad-accum | `train.py` (EMA `.step` / `loss.item()` outside the `accel.sync_gradients` gate) | "Don't sync/copy on accumulation-only steps." | Guard `ema.step` and `loss.item()` with `if accel.sync_gradients:` — also fixes EMA decay-horizon bias. | **Low** | Low | Low | **grad_accum defaults to 1** and the Makefile never overrides it → the live run is a **pure no-op**. EMA `.step` is an on-device foreach lerp (not a host copy as the finding claimed). |
| 10 | **DataLoader `prefetch_factor` left at 2** | `train.py:448-457` | "Deepen the host queue to hide decode latency" (each `__getitem__` does two PIL decode+resize passes). | `prefetch_factor=4`, expose as CLI flag. | **Low** | Low | Low | Glyph render is in the **forward**, not workers, so the step is likely GPU-bound → near-zero gain until profiling proves dataloader-bound. Watch RSS on the shared box. |
| 11 | **Non-blocking H2D** on val/sample/glyph copies | `train.py:108,137`; `content_encoder.py:89` | "Pinned memory only pays off with `non_blocking=True`." | Add `non_blocking=True` to those `.to(device)` calls. | **~None** | Low | V.low | Glyph copy is consumed by `self.cnn(...)` on the next line → sync forces zero overlap; val/sample run only every N steps. Subsumed by #2's atlas. |

## 3. Do tomorrow (watched) — ordered

**These all require restarting the run, so apply them together in one new run and watch DCGM TENSO + images/sec move.**

1. **Add width to the manifest + bucketed `batch_sampler`** (#1) — biggest single win: removes 20-40% wasted VAE+DiT pad FLOPs *and* stabilizes shapes so compile/pos-embed reuse pay off. Foundation for #6.
2. **GPU glyph atlas** (#2) — removes the per-step PIL stall + H2D copy on the forward's critical path; this is the most direct hit on the SMACT-high/TENSO-low overhead signature.
3. **Offline VAE latent cache** (#3) — gated behind a `--cache-latents` flag (disable it the moment image augmentation is added). Run the pre-encode sweep once, then every step skips the frozen encoder. *(Implemented 2026-07-15, removed 2026-10-19.)*
4. **`torch.compile` with `dynamic` width** (#6) — only meaningful *after* #1 buckets the widths; otherwise it thrashes. Mark width dynamic, leave batch/channels/height/seq static.
5. **TF32 one-liner** (#5) — free, safe, do it in the same restart.
6. *(Optional, large disk)* **DINOv3 feature cache** (#4) — only if the ~120 GB memmap is acceptable; key by the actually-loaded encoder.
7. *(Only if `--repa` ever turned on)* bundle the REPA target cache (#8) into #3's sweep.

After restart, watch for: TENSO climbing above ~0.25, images/sec up, and (if compiled) `TORCH_LOGS=recompiles` staying quiet.

## 4. Rejected by verification (honest)

- **Fused norm/activation kernels (Liger/apex) as a "shape-robust alternative to compile"** — **REJECTED (valid=false).** The load-bearing premise is wrong: the normalized feature dim is **fixed** (1024); only line **width / sequence length** varies, which eager LayerNorm already handles — it was never the recompile trigger. The benefit overlaps the existing `torch.compile` path; highest-value targets are vendored in diffusers (brittle monkeypatching); apex/Liger must support Blackwell sm_120/cu128 where wheels lag. Self-rated effort=high/risk=high, "do only if compile is ruled out." Not adopted.
- **`--profile-steps N` torch.profiler flag (design)** — **REJECTED (already implemented).** The scaffold already shipped it: `src/diffu/profiling.py` is the shared classifier, `scripts/profile_tensorcore.py` (removed; in git history at `c53d30e`) imports it, and `train.py` has `--profile-steps` (default 0) wired through `_run_profile()`, which profiles N real dataloader steps via `accel.backward`, writes the kernel table + TENSO% to `<out_dir>/profile_step_table.txt`, and returns **before** the main loop. Remaining gaps (chrome-trace export, `profile/` subdir, `--profile-then-exit`) are cosmetic and were themselves marked optional.

### Verified non-issues (correctly scoped out, no change)
- **SD3 attention is already on the fused SDPA/Flash path** (`JointAttnProcessor2_0` → `F.scaled_dot_product_attention`, head_dim=64, no mask, bf16) — confirmed by construction; it is **not** the cause of low TENSO. Don't chase an attention math-fallback.
- **`channels_last` does NOT apply to the VAE** — it's a frozen 3D-conv (5D-tensor) module off the Stage-1 hot path; NHWC doesn't even apply (would be NDHWC, narrow cuDNN coverage). Applying it would cause format thrash. Steer effort to the DiT where gradients flow.
- **CANINE/`TextEncoder` is never instantiated** (`use_canine=False` dead flag); **guidance_image / discriminators are Stage-2-only** (`--stage2`, single-GPU). Zero Stage-1 cost. Re-audit only if either is enabled.

## 5. Profiling instrumentation (new)

Scaffolded by the prior pass — importable, CPU-safe (`torch` imported lazily; importing the module never touches CUDA).

**Created** `src/diffu/profiling.py`:
- `TENSOR_CORE_RE` — regex tagging GEMM/conv kernels (`gemm|tensorop|wmma|s16816|s161616|cutlass|implicit|dgrad|wgrad|fprop|cublas`).
- `KernelRow`, `ProfileReport` (pydantic) — `ProfileReport.render()` prints the profiler table + Tensor-Core breakdown + top-15 kernels tagged `TC`.
- `tensor_core_fraction(prof)` — share (0-1) of self-CUDA time in TC-eligible kernels.
- `profile_callable(step, *, warmup=2, active=3)` — runs `torch.profiler` (CPU+CUDA, `schedule(wait=0, warmup, active, repeat=1)`), syncs CUDA, returns the report.
- `read_dcgm(fields=(1002,1004,1005))` — shells `dcgmi dmon -c 1 -e <fields>`, parses SMACT/TENSO/DRAMA defensively by header; returns `None` if `dcgmi` absent; **no GPU memory cost**.

**Edited** `scripts/profile_tensorcore.py` (removed; in git history at `c53d30e`) (now imports from `diffu.profiling`, DRY) and `train.py` (`--profile-steps N`, default 0 → off; at N==0 the only delta is a guarded early-return branch that's never entered, so the training path is behaviorally unchanged).

**Commands** (do NOT run now — training owns all GPUs; use later on a free/least-busy card):
```bash
# (a) standalone synthetic-step profiler (scripts/profile_tensorcore.py — removed; in git history at c53d30e)
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python scripts/profile_tensorcore.py

# (b) profile N REAL training steps (writes <out-dir>/profile_step_table.txt, then exits before the loop)
CUDA_VISIBLE_DEVICES=1 uv run --no-sync -m diffu.train \
  --data-dir data_out --out-dir checkpoints --profile-steps 5
```

**Not yet wired (recommended next):** continuous in-run logging of `read_dcgm()` TENSO/SMACT/DRAMA into `tracker.log` on the `step % log_every == 0` cadence (the sampler exists; ~5 lines to wire), plus windowed `steps/sec` + `images/sec` (pure `perf_counter` arithmetic, main-process-only, zero device syncs). Both are observability-only — the headline number to watch when validating #1-#3.
