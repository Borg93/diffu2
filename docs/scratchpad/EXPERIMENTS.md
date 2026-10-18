# Experiments

## 2026-07-13 — Closing entry: this log stops at 06-22; what happened after

- **`exp_stage2_wordcritic` went flat** (like the PatchGAN before it) → discriminators dropped entirely.
- **DOG guidance and the recog-loss removed** (tried + rejected); classic CFG won, shipped default `cfg_scale=5`.
- **`exp_sd35_fast` (64 px, SD3.5 qk-norm)** → released as `Gabriel/test_sd35` (~0–5 % guided CER).
- **`exp_sd35_128` (128 px + the CFG-null fix `--cond-dropout 0.1 --text-dropout 0.1`)** converged ~100k,
  killed 07-05 at 216k; `step_120000` released 07-05 as **`Gabriel/diffu_sd35_128`** (~2 % guided CER).
- **07-13 discovery:** batch composition changes output quality — glyph conditioning is not pad-invariant
  (work-lists must shuffle for mixed lengths; width bucketing blocked until a conditioning fix + retrain).

**The log continues in `docs/STATUS.md`.**

---

## 2026-06-22 · night — Classic CFG breakthrough + warm-started discriminator

**Classic CFG works dramatically** (replaces the broken earlier guidance method). N=64 honest sweep on the coherent base
`exp_base_weekend/step_360000` (EMA, glyph-line, no-style-tokens), eval recognizer trocr-base-hist-swe-3,
24 steps, foreign style refs:

| CFG | gen_cer | ctrl_gap |
|---|---|---|
| 0.0 | 0.507 | +0.896 |
| 1.5 | 0.301 | +1.096 |
| **3.0** | **0.163** | +1.231 |
| **4.5** | **0.157** | +1.272 |
| 6.0 | 0.160 | +1.261 |
| 8.0 | 0.141 | +1.305 |

- **gen_cer 0.51 → 0.16 (cfg 3), plateau ~0.15–0.16 for cfg 3–4.5** — on the single-domain Swedish model,
  far below the old ~0.60 plateau. The conditioning pathway WORKS; it needed
  real guidance. The old guidance method was 0.95 ≫ raw (broken) and masked this all along.
- Implemented as `v = v_uncond + cfg·(v_cond − v_uncond)` with the content tokens + concat glyph dropped,
  style/pooled kept (uses the trained `--cond-dropout`/`--text-dropout`). Sampler: `diffu.py _denoise_latents`.
  Exposed in `app.py` + `generate` CLI (default 3.0); the old guidance method relegated. Training eval logs **`gen_cer_cfg`**
  (scale 3.0) replacing the dead `gen_cer_dog`.

**Running: `exp_stage2_disc`** — warm-started content discriminator on the coherent base:
`--init-from .../step_360000/ema_model.safetensors --glyph-line --no-style-tokens --ink-focal --stage2`,
2-GPU DDP, batch 16, lr 5e-5, 100k steps, content_weight 0.05. **Watch: does the *unguided* gen_cer bend
below ~0.59 by ~step 40k?** (the discriminator is a bonus lever; CFG already won).

**UPDATE (later 06-22):** that run used the **placeholder PatchGAN** word-critic and went **flat** (unguided
gen_cer 0.59→0.67 — the PatchGAN rewards realism, not the right word). **Built the real attention word-critic**
(`discriminators.py`: conv stem → per-column word-region tokens → **RoPE self-attention** reusing
`build_2d_rope`'s shared-column scheme → per-region real/fake logits, conditioned on the guidance). Smoked
(unit + 2-GPU DDP/bf16) and relaunched as **`exp_stage2_wordcritic`** (line 3D-CNN + RoPE word-critic,
warm-started from `step_360000`, logs `gen_cer` + `gen_cer_cfg`). Verdict ~tomorrow. Stage-2 confirmed DDP-OK
on a 2-GPU smoke (the "single-GPU only" docstrings were stale — fixed).

**Verified refs (cloned):** a **position-controlled** glyph-image concat (bbox-fit + ControlNet);
DiffInk = channel-concat but **NOT** aligned (1-D strokes + self-attention) — so our 2-D uniform-monospace
concat was the weak design (corrects the old "DiffInk = aligned concat" claim). → concat dropped.

---

## 2026-06-22 — Weekend base run RESULT + content-path A/B (concat vs attention)

**`exp_base_weekend` (`--glyph-line --no-style-tokens --ink-focal`, DDP×3, batch 96) ran to ~414k/500k,
stopped 06-22.** Verdict: **coherent but illegible, and mildly overfitting** — NOT usable for HTR
augmentation.

| metric | early | late (350–415k) | 200k→410k | reading |
|---|---|---|---|---|
| flow_loss (train) | 0.056 (50–100k) | **0.049** (min 0.0338 @410k) | −5.0% | still falling at the end |
| total loss (train) | 0.088 | 0.076 | −7.1% | still falling |
| **val_loss** | **0.079 (min @134k)** | **0.087** | **+1.9% (UP)** | bottomed early, drifts up |
| gen_cer | 0.68 (50–100k) | 0.60 | +1.1% | plateau ~0.60 (noisy min 0.49 @348k) |

- **Train keeps improving, val does not** → the train/val gap widens ~0 → ~0.011 = a real (mild)
  generalization gap. Best val @134k, best gen_cer @348k → the **final checkpoint generalises worse** than
  ~320–360k. Corrects the earlier "stable / no overfit" note.
- **Conclusion:** scaling *steps* on 1.36M Swedish lines is exhausted (counterproductive past ~150–350k).
  The plateau at ~0.60 is the documented trigger to escalate the content path from
  attention (`--glyph-line`) to channel-concat (`--glyph-concat`) AND to scale DATA diversity.
- The old guidance path still broken at inference (`gen_cer_dog` ~0.95 ≫ raw 0.60) — sample with dog_scale 0.

**Incoming commits verified before launch (all green):** pyrefly 0 · pytest 33 · `tests/test_grad_accum_gating.py`
(new) proves the grad_accum>1 EMA/step fix · smokes for default/`--glyph-concat`/`--glyph-line` · a real
2-GPU `--grad-accum 2` run passed val+sample with no DDP hang.

**Running A/B (matched, single-GPU each, `data_out_lejonet`, batch 64, everything else held constant):**

| Arm | trackio run | GPU | content path | flags (beyond shared) |
|---|---|---|---|---|
| **A** | `exp_concat_cfg` | 0 | channel-concat (DiffInk) | `--glyph-concat --cond-dropout 0.1 --text-dropout 0.1` |
| **B** | `exp_line_cfg` | 1 | cross-attention | `--glyph-line --cond-dropout 0.1 --text-dropout 0.1` |

Shared: `--no-style-tokens --ink-focal`, lr 1e-4, REPA on, logit-normal 0.0. **A vs B isolates concat vs
attention at matched CFG + batch**; the weekend run (line, no CFG, batch 96) is a loose 3rd reference for the
CFG effect. **Decisive metric: `ctrl_gap` > 0** (output follows the requested text, not a copy/collapse),
with `gen_cer` as the legibility gauge. GPU 2 free (slot for a 3rd arm = concat WITHOUT CFG → completes the
2×2). If concat clearly beats attention on ctrl_gap → scale that recipe on a larger, more-diverse corpus
(non-Swedish pretrain → Swedish fine-tune).

---

## 2026-06-18 — GT-Lejonet: style-token leak fix + grounding attempts → weekend base run

> **Corrects the "architecture is fully capable" claim in the 06-17 section below.** That rested on the
> overfit `brave-forest-1` (exp C), which we found used **val ⊆ train + a style-ref content leak** — it
> measured MEMORIZATION, not capability. So C is NOT proof the architecture can follow text.

**Setup.** New corpus **GT-Lejonet** (`data_out_lejonet`, 1.36M lines, **volume-disjoint** split) · leak
fixed (`_style_index` excludes the target line) · honest different-text eval (style ref shows text B, we ask
for A; a copy reads as B → CER-vs-A spikes) · `--glyph-line` · batch 32 (×3 DDP for the weekend run).

| Arm | trackio run | config | result | verdict |
|---|---|---|---|---|
| **A** | `exp_A_control` | style tokens IN context | gen_CER **flat 0.76→0.81** (1k–3k), loss floored | marginal collapse on clean data |
| **B / C** | `exp_B_pooled` / `exp_C_pooled_lownoise` | `--no-style-tokens` (C +`--logit-normal-mean -0.5`) | flat ~0.78 (5 pts) | style-token fix **necessary** (stops copying) but **not sufficient** |
| **D** | `exp_D_control` | base, no grounding | flat ~0.77 | clean baseline + warm-start source |
| **E / F** | `exp_E_recog0p25` / `exp_F_recog1p0` | frozen-TrOCR recog CE, weight 0.25 / 1.0 | `recog_loss` **flat ~3.4** (can't be reduced), images **degraded to white/speckle**, gen_CER **WORSE**, F loss climbed 3.9→4.5 | recognizer grounding **FAILED + harmful** |

### Conclusions
1. **The model COPIES the style reference.** The 64 spatial DINO style tokens carried glyph content
   (`diffu.py` cat content+style_tokens). Confirmed: hold the style ref, swap the text → output unchanged.
   `--no-style-tokens` (style as a pooled AdaLN vector only) stops the copying.
2. **Killing the copy-shortcut is necessary but not sufficient** — the model still collapses to the marginal
   (generic scribbles) because nothing in the loss rewards legibility.
3. **Recognizer-CE grounding fails from scratch.** The OCR-CE gradient through a noisy decoded `x0_pred` is
   useless/harmful: the CTC/OCR loss flattens style and is worse than a discriminator, and grounding
   only helps as a late-stage polish on an **already-coherent generator** —
   grounding is a *polish, not a from-scratch teacher*.
4. **→ Weekend base run `exp_base_weekend`** (DDP × 3, `--glyph-line --no-style-tokens --ink-focal`,
   `--steps 500000` ≈ ~30 epochs by Mon, NO grounding): does the base reach coherence (gen_CER off ~0.77)?
   Monday → add Stage-2 (warm-started discriminator) if coherent, else a prior (conv / pretrained SD3 init).

### Queued — `--glyph-concat` (channel-concat content)
`--glyph-line` is content via **cross-attention** — the glyph line is injected as attention context.
If attention + scale plateaus, escalate to **channel-concat** — stack a spatial glyph
latent onto the noisy latent so content is present at every cell (impossible to ignore). Verified by code:
**DiffInk** (closest *DiT* peer: content `torch.cat` to
input **+ CFG**) does this. Implemented behind `--glyph-concat` (default off, untested). **Planned arm:**
`--glyph-concat --cond-dropout 0.1 --no-style-tokens` (concat+CFG) vs the `--glyph-line` arm,
compared on `gen_CER`. Smoke first: `diffu-smoke --glyph-concat`.

---

## 2026-06-17 → 06-18 — Overnight ablation: where does content-following break?

**Question.** From-scratch SD3 MMDiT produces realistic handwriting *texture* but illegible *content* —
the generator ignores the text instruction ("paints handwriting, can't spell"). Localize the failure
with controlled single-GPU ablations on 3× RTX PRO 6000.

**Shared config.** From scratch · batch 64 (E: 32) · lr 1e-4 · `--ink-focal` ON · `--no-repa` ·
`--no-fill-ratio` · cond-dropout 0 (content always-on) · RoPE ON · micro-averaged gen_CER ·
one GPU each · ran ~14.8 h ≈ 43k steps. trackio project `diffu`. Logs + want/HTR-read sample grids in
`data_out/checkpoints/exp_*/`.

| Arm | GPU | trackio run | Differs by | Result | Verdict |
|---|---|---|---|---|---|
| **A — stripped baseline** | 0 | `dainty-sunset-0` | full 299k data | loss 0.09 (converged) · **gen_CER ~2.0 · illegible** | confounds were NOT the cause; still marginal-collapse on full data |
| **C — overfit / capacity** | 2 | `brave-forest-1` | **256 fixed lines** | loss **0.0003** · **gen_CER ≈ 0 — PERFECT legible reproduction** (HTR reads the exact target text) | ✅ architecture + rectified flow + conditioning + RoPE + VAE all WORK and CAN follow text |
| **E — + Stage-2 discriminators** | 1 | `calm-river-2` | `--stage2`, batch 32 | base flow loss **stuck ~1.93** · d-loss → 0.01 (saturated) · output = **RGB noise** | ❌ GAN-from-scratch destabilized the generator |

### Conclusions
1. **The architecture is fully capable.** Exp C generates legible, correct-text Swedish handwriting
   *from noise* (gen_CER ≈ 0; HTR reads `Rapportbok`, `vid Polisens detektiva afdelning`, `År 1880.`,
   `Rapportbok år 1880.` correctly). SD3 MMDiT + rectified flow + glyph/DINO conditioning + RoPE + the
   16-ch VAE are **not** the problem, and the text instruction **can** control the output.
2. **The real problem is GENERALIZATION of content-following, not capability.** On 256 lines it learns
   the text→image mapping perfectly; on 299k diverse lines it cannot generalize that mapping and falls
   back to the marginal (generic plausible handwriting). gen_CER ~2.0 on full data (A) vs ~0 memorized (C).
3. **Stripping the 5 confounds helped only marginally.** A (~2.0 micro) is a bit better than the
   previous full-confound run `silly-flower-15` (~3.2 micro-equiv) but still illegible — confounds were
   not the root cause.
4. **Discriminators from scratch fail.** They need a warm-started (already-legible) generator —
   multi-scale content learning works as an *auxiliary* on a trained generator, not a from-scratch GAN.

### Next experiments (queued)
- Make content-following **generalize few→many**: stronger content conditioning; a content-controllability
  ("control score": is the requested text read closer than a random text?) metric to track it directly.
- **Warm-start + Stage-2**: apply discriminators only after the generator is legible (not from scratch).
- Flow **#3** low-noise timesteps (`--logit-normal-mean -0.5`) — was swapped out for E; re-run.
- Flow **#1** length-aware-shift — wire carefully (train/inference matched) and A/B on wide lines.
