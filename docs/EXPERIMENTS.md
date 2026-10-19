# Experiments

This document records the experiments logged for Diffu — a diffusers-first Swedish handwriting-line
generator for HTR augmentation (SD3 MMDiT + rectified flow + 16-channel VAE + glyph-line content
conditioning + DINOv3 style). Every training run is tracked with [trackio](https://github.com/gradio-app/trackio)
in two projects:

- **`diffu`** — the generator itself: the from-scratch base, the released 128px model, the ablations,
  and the IAM adaptation arms.
- **`diffu-htr`** — the downstream question: do generated lines work as *training data* for a real HTR
  recognizer (TrOCR)?

View either dashboard with:

```bash
uv run trackio show --project diffu       # generator runs
uv run trackio show --project diffu-htr   # downstream TrOCR A/B
```

> **Note (2026-10-19):** a record of past runs, kept as written. Since then: `Gabriel/diffu_sd35_128` is no
> longer on the Hub; `--glyph-line` and `--no-style-tokens` were removed (the glyph-line encoder and pooled-only
> style are now the only options — drop those flags from the commands below); the ink-focal loss is now on by
> default; the latent cache was removed; the experiment drivers (`scripts/diag_*`, `scripts/exp_*`) are in git
> history at `c53d30e`. IAM with real writer ids is now built by `scripts/prepare_iam_writers.py`
> (`TRAINING.md` §3).

A caveat runs through everything below and is worth stating once up front: the generator's `val_loss` is
**flow-matching MSE, not legibility**. It measures how well the model predicts the rectified-flow velocity
field, not whether a human or a recognizer can read the output. Where we have a legibility signal we log
`gen_cer` (unguided) and `gen_cer_cfg` (CFG-guided); where we do not, we say so.

## IAM adaptation: LoRA vs full vs from-scratch

Three arms adapt the released 128px Swedish base (`exp_sd35_128/step_120000`) onto the English
`Teklia/IAM-line` dataset, to compare the *step and parameter efficiency* of three adaptation strategies.
All three share the base architecture flags (`--glyph-line --no-style-tokens`; both removed 2026-10-19, their
behaviour is now the only option), batch 8, `line_height=128`,
single GPU, warmup 2000 steps + cosine decay to ~1e-6.

| Arm | Key config | Final val_loss | Min-ever val_loss (step) | Last-5-mean val_loss | Throughput / wall | Takeaway |
|-----|-----------|---------------|--------------------------|----------------------|-------------------|----------|
| `iam_lora` | rank-32 attn-only LoRA, `--init-from` step_120000, 4k steps, lr 3e-4 | 0.0948 | **0.0726** (3000) | **0.0787** | 22.4 img/s / ~24 min | Matches full fine-tune on flow-loss with a **12.5M-param / 48 MB** adapter. Best step/param efficiency. |
| `iam_full` | full backbone FT, `--init-from` step_120000, 6k steps, lr 1e-5 | 0.0884 | **0.0724** (3000) | 0.0843 | 21.6 img/s / ~38 min | No measurable gain over LoRA; converged by step 3000 — the extra 3k steps bought nothing. |
| `iam_scratch` | random init + REPA, 60k steps, lr 1e-4 | 0.0730 | 0.0664 (53250) | 0.0864 | 21.8 img/s / ~6.3 h | Needs ~26k steps (~2.8 h, ≈9×) just to reach fine-tune parity; lower min-ever, but inside the noise. |

### The efficiency story

**LoRA is statistically indistinguishable from a full backbone fine-tune on flow-loss.** Over the 15
shared eval steps (250–3750) the two curves co-move at Pearson **r = 0.994**, with a mean absolute gap of
just **0.0007** (max 0.0037). Both bottom out at **~0.0725 at step 3000** (LoRA 0.0726, full 0.0724 — a
0.0002 difference). The rank-32 attention-only adapter is **12.5M trainable parameters** — a **48 MB**
file (382 tensors: 8 attention projections × 24 blocks) against the **5.2 GB / 1.38B-param** base
checkpoint — and buys the same adaptation as fine-tuning the entire backbone, at ~3.6% higher throughput.

**Both fine-tune arms converge in ~3000 steps (~18 min wall).** `iam_full`'s minimum val_loss occurs at
step 3000, *not* at its final step 6000; steps 3000→6000 just wander in the noisy 0.075–0.095 band. The
extra 3k steps of the full run were wasted compute.

**From-scratch is ≈9× more expensive to reach parity.** `iam_scratch` + REPA first crosses below the
fine-tune minimum (0.0726) at **step 26000 (~2.8 h)**, versus ~18 min for the fine-tune arms, and its
train loss has to fall 2.37→0.0625 from random init (the fine-tune arms start already at ~0.091). It does
keep drifting down to a **min-ever 0.0664 at step 53k** — lower than either fine-tune arm — but see the
caveat below.

### Read the ranking carefully

The one-number "final val_loss" (full 0.0884 < LoRA 0.0948) **mis-ranks the arms**, and it is worth being
explicit about why:

- The per-eval noise floor is **peak-to-peak ~0.03**, which is *larger than every inter-arm gap* in the
  table. The scratch tail alone oscillates 0.0664→0.1016; LoRA bounces 0.0726→0.0948 between steps 3000
  and 3750.
- "Final" compares *different steps*: full ran to 5750, LoRA stopped at a high-variance step 3750. At the
  same step they are tied (full 0.0942 vs LoRA 0.0948).
- The robust **last-5-mean flips the order to LoRA 0.0787 < full 0.0843 < scratch 0.0864** — i.e., by a
  noise-averaged metric LoRA looks *best*, the opposite of the last-point ranking.

Likewise, scratch's lower min-ever (0.0664) is a favorable draw, not a stable level: its mean over steps
≥45k is 0.0835, and the 0.006 gap to the fine-tune minimum is well inside the ~0.03 single-run eval variance.

**Scope limit.** None of these arms logged legibility (`log_cer=False` in all three configs), so this
experiment can only conclude on **step and parameter efficiency**, not output quality. The bottom line —
*LoRA gets you the same adaptation as a full fine-tune, ~9× cheaper than from-scratch* — is a statement
about flow-loss and wall-clock, nothing more. The definitive quality verdict comes from the downstream
HTR A/B below.

## Does synthetic data work as HTR training data? (the augmentation A/B)

The efficiency numbers above say nothing about whether the generated lines are *useful*. This experiment
asks the only question that ultimately matters for an augmentation tool: **if you fine-tune a real HTR
recognizer on generator output, does its accuracy on real held-out text go up or down?**

Setup: fine-tune two `microsoft/trocr-base-handwritten` recognizers for 5 epochs (batch 8, lr 5e-5),
evaluate both on **1000 held-out real IAM test lines**. The only difference is the training set:

| Run | Training data | Epoch-1 CER | Epoch-5 CER | Standalone CER (1000 real test) |
|-----|--------------|-------------|-------------|---------------------------------|
| `htr_real` | 2000 real IAM | 32.05% | 9.67% | **9.50%** |
| `htr_realsynth` | 2000 real + 4000 diffu-LoRA synth (2:1) | 23.14% | 12.04% | **11.91%** |

### The headline: naive 2:1 augmentation HURT the recognizer

> **✅ Resolved — this −2.4-pt result was a bug, not a verdict.** The synth here was rendered through a
> **canvas-width bug** (garbled images) *and* generated by an **undertrained adapter** (two separate defects
> — see *Root cause* and *Two bugs, then a win* below). With both fixed and a sensible ratio, clean synth
> actually **HELPS**: at 0.25:1 it *beats* the real-only baseline (**9.01% vs 9.50%**). The broken run is
> kept because the failure — and the fact that the downstream CER, not the flow val_loss, is what caught it
> — is the instructive part.

Adding 4000 synthetic IAM lines (from the `iam_lora` adapter) at a 2:1 synth:real ratio **raised**
standalone CER from **9.50% to 11.91%** — a **+2.41-point absolute / ~25% relative regression**. The
in-training `eval_cer` and the independent standalone re-eval agree in both magnitude (~2.4 pts) and
direction, so the result is robust for this configuration.

### The per-epoch curve is the diagnostic

The two curves *cross over*, which is the tell:

```
htr_real       0.3205 → 0.1838 → 0.1417 → 0.1176 → 0.0967   (epochs 1–5)
htr_realsynth  0.2314 → 0.2536 → 0.1686 → 0.1356 → 0.1204   (epochs 1–5)
```

- **Epoch 1: real+synth is far *better*** (23.1% vs 32.1%) — 3× the data gives a strong head start.
- **Epoch 5: real+synth is clearly *worse*** (12.0% vs 9.7%) — real overtakes from epoch 2 on and finishes ahead.
- There is even a **non-monotonic bump** in real+synth from epoch 1→2 (0.2314→0.2536).

The shape *looks* like the classic signature of fitting a mismatched distribution — but the real cause
turned out to be a **rendering bug**, and it took looking at the actual synth images to find it.

### Root cause: a canvas-width bug (not the model)

Inspected at full resolution, the `iam_lora` synth lines were **garbled** — the model fell into
**character-looping / stuttering**, repeating strokes instead of committing to the next glyph
(`"Sentence Database"` → `Semewtentenacccee DDatatatababasase`). Chasing it down through the `scripts/diag_*`
sweeps (removed; in git history at `c53d30e`):

1. A **CFG sweep** (`diag_cfg_sweep.py`) ruled out sampling — the looping is present at *every* guidance
   scale (1.0–5.0).
2. A **model sweep** (`diag_model_compare.py`) ruled out the adapter — base, full-FT, and the 60k
   from-scratch arm all loop the same way, and the **base loops on native Swedish too**
   (`diag_swedish_control.py`).
3. A **width sweep** (`diag_width_sweep.py`) found it: every generated line was **2–3× wider than the real
   crop**, and at the tight/real width the model rendered **cleanly**.

The model has **no length control** — it *fills* whatever canvas width it is given, so an over-wide canvas
is filled by looping characters. The width came from `natural_width`, whose raw estimate scales with
line-height; its `_WIDTH_BIAS = 0.55` was tuned for the original 64px model (~15 px/char) but at today's
**128px** it yields ~29 px/char — squarely in the documented catastrophic-looping regime (the code's own
CER-vs-density sweep: 30 px/char → 0.41 CER, 15 → 0.037). Worse, **`synth.py` bypassed the bias entirely**,
using the raw ~53 px/char width — the worst case.

**The fix (verified):** `_WIDTH_BIAS 0.55 → 0.45` (the empirically clean ~24 px/char at 128px, via
`diag_bias_pick.py`, removed; in git history at `c53d30e`) and make `synth.py` use the biased width + `ink_crop`
like the deploy path. Regenerated
synth is now clean and legible (`Any mental confusion can cause this and`, `She went back to Philip. But`).

### Two bugs, then the honest verdict

The −2.4-pt regression was two bugs stacked, neither of which is "synth doesn't work":

1. **Canvas-width bug** (above): the synth was garbled by looping. Fixing `_WIDTH_BIAS` alone cut the harm
   from **+2.4 → +0.6 pts** (real+synth 11.91% → 10.13% at 2:1).
2. **Undertrained adapter**: `iam_lora` spent 50% of its 4k steps in LR warmup (the default `warmup_steps=2000`
   left on a short run), so it barely trained — and `log_cer=False` meant nobody saw it. Retrained with a
   short warmup + a *meaningful* gauge (the **English** recognizer, not the Swedish default — the IAM renders
   are English), the adapter reaches **~1% guided gen_CER** and renders legible lines the English recognizer
   reads verbatim.

**The verdict — regime-dependent, and don't over-read the small numbers.** Two regimes tell opposite stories.

**Data-rich (2000 real lines).** A first single-seed ratio sweep *appeared* to show synth helping only at a
tiny ratio and hurting beyond (standalone CER; provisional baseline real-only = 9.50%):

| synth : real | CER | vs 9.50% |
|-------------:|:---:|:-----------:|
| 0.25 : 1 | 9.01% | −0.49 |
| 0.5 : 1 | 9.60% | +0.10 |
| 1.0 : 1 | 9.80% | +0.30 |
| 2.0 : 1 | 10.13% | +0.63 |

**This table is an artifact — do not trust it.** It has two confounds (an adversarial 6-finder/3-judge audit
flagged both): it's **single-seed**, and arms are matched on **epochs not steps**, so the synth-heavy arms
ran *proportionally more optimizer steps* (2:1 = 3× real-only) and overfit toward the synth distribution. Both
confounds cut the same way — they **hid a large positive effect**. The provisional 9.50% baseline was itself a
lucky-low single draw (true 5-seed mean **9.95%**). When we redo the sweep **step-matched and 5-seeded** (the
*settle*, two sections down), the ranking **inverts**: more synth is monotonically *better*, 2:1 reaching
**8.45%** (−1.50 pt vs real-only, 5/5 seeds). The only thing this single-seed table got right is that
*synth-only* / *curriculum* pretraining (no real mixed in) backfires — that one is real (below).

**Low-resource (200 real lines) — the regime diffu actually targets.** Here synth is a decisive win:

| low-resource arm | CER | vs 200-real-only |
|---|:---:|:---:|
| 200 real only | 25.41% | — |
| 200 real + 800 synth | 16.27% | **−9.14 (−36%)** |
| 200 real + 2000 synth | 15.47% | **−9.94 (−39%)** |

When real data is scarce, clean synth cuts CER by ~**40% relative**, and here *more* synth keeps helping
(dose-response runs opposite to the data-rich case). **This is the headline: diffu synth helps most exactly
where you have little labeled data** — Swedish archives, new hands, low-resource scripts.
(Reproduce: `scripts/exp_htr_ratio_sweep.sh`, `exp_htr_lowres.sh` — removed; in git history at `c53d30e`; arms
in trackio `diffu-htr`.)

### Is the synth broken? No — it's label-clean but off-manifold

The natural worry is that "more synth hurts" means the synth is garbage. It isn't. **Fidelity probe:** the
strong real-only recognizer (9.50% on real test) reads our 4000 synth images against their *own* labels at
**2.90% CER** — *cleaner* than it reads real scans, with near-verbatim reads. The synth is **legible and
correctly labeled**; the hurt is not label noise. What it *is* is an **image covariate shift**: measured on
CPU, synth is ~2× denser ink (0.16 vs 0.07), ~1.9× contrast (std 84 vs 44), and aspect-capped — a too-clean,
too-dense rendering that sits off the real-scan manifold. That is a **fixable generator artifact** (the
earlier width-broken→width-fixed regeneration alone moved the 2:1 mix by **1.78 pt**), so we pulled the lever:
a CPU per-image standardization of synth grayscale to the real-IAM (mean, std) — no regeneration — that pulls
synth ink 0.17→0.06, contrast std 85→44, matching real (0.08 / 45). Whether that harmonization *matters* is
answered by the settle below.

### The settle: step-matched + 5-seed — more synth is *better*, and harmonization is unnecessary

Removing both audit confounds at once — `--max-steps 1250` (identical optimizer budget + LR decay) × **5
seeds** — flips the data-rich story entirely. CER on 1000 real IAM test lines, mean ± sd over seeds 1–5, with
the **paired** (per-seed) delta vs real-only (`scripts/exp_htr_harmonize_sweep.sh` + `exp_htr_harm_hiratio.sh`,
removed; in git history at `c53d30e` → `results_harmonize.txt`, `results_harm_hiratio.txt`):

| arm (step-matched) | CER (5 seeds) | paired Δ vs real-only | seeds better |
|---|:---:|:---:|:---:|
| real-only | 9.95% ± 0.24 | — | — |
| raw **0.25:1** | 9.33% ± 0.48 | **−0.62 pt** (t=−4.0) | **5 / 5** |
| raw **2:1** | **8.45% ± 0.18** | **−1.50 pt** (t=−10.3) | **5 / 5** |
| harmonized 0.25:1 | 9.24% ± 0.30 | −0.71 pt | 5 / 5 |
| harmonized 2:1 | 8.57% ± 0.36 | −1.38 pt | 5 / 5 |

Three conclusions, each **overturning an earlier claim**:
1. **More synth is *better*, monotonically** — the exact opposite of the single-seed table above. 2:1 beats
   real-only by **−1.50 pt** (~15% relative, 5/5 seeds) and beats 0.25:1 by −0.88 (t=−3.3, 5/5). Diffu synth
   is a strong, *dose-dependent* augmentation even against a saturated 2000-real baseline.
2. **The "more synth hurts" curve was an epochs-not-steps artifact.** At fixed epochs the 2:1 arm (6000 lines)
   trained **3× more steps** than real-only and overfit *toward* the synth distribution → the old 10.13%. Hold
   steps constant and the sign flips to a big win. This confound was corrupting the whole ratio sweep, not just
   the 0.25:1 point — the single most important methodological lesson here.
3. **Harmonization does nothing — at any ratio.** harm ≈ raw at 0.25:1 (−0.71 vs −0.62) *and* at 2:1 (paired
   harm−raw = +0.12, t=+0.7, harm better in only 1/5). So the synth's "too clean / too dense" photometrics were
   never the limiter; its value is the **data diversity / regularization** it injects, and *raw* synth delivers
   it. The covariate-shift hypothesis is falsified for this task — no need to harmonize.

**Where does "more synth" peak?** Pushing the ratio further (fresh disjoint fakes, pooled to 14k; step-matched
1250 × 5 seeds; `exp_htr_moreratio.sh`, removed; in git history at `c53d30e` → `results_moreratio.txt`) finds
the turnover at **~2:1**:

| ratio | synth | CER (5 seeds) | paired Δ vs real-only |
|---|---|:---:|:---:|
| **2:1** | 4000 | **8.54% ± 0.37** | **−1.41** (peak) |
| 4:1 | 8000 | 8.82% ± 0.10 | −1.13 |
| 7:1 | 14000 | 9.45% ± 0.18 | −0.50 |

All still beat real-only (5/5 each), but the gain *erodes* past 2:1. The mechanism is the **fixed step budget**:
at 1250 steps, more synth means the 2000 real lines are sampled *less often per step*, so the real signal gets
diluted. **This is the key argument for a curriculum at scale** — to exploit ~100k fakes you must *not* dilute
the real-data fit; pretrain on the fakes (absorb them over many steps) and *then* finetune on real, which is
the scale-up experiment (`docs/STATUS.md`). Mixing 100k at a fixed budget would just be a worse 7:1.

### Mixing vs the curriculum — and why the curriculum backfired

Mixing is the *naive* way to use synth; the textbook-stronger recipe is a **curriculum** — pretrain on synth,
then fine-tune on real (how TrOCR itself was built). We tested both, plus a LoRA-adapt variant (`diffu-htr-train
--lora`, PEFT on the recognizer's decoder attention). Standalone CER on real IAM test:

| method | CER | vs 9.50% baseline |
|--------|:---:|:-----------------:|
| mix 0.25 : 1 (within noise — see caveat above) | 9.01% | −0.49 |
| curriculum LoRA (synth-pretrain → LoRA on real) | 11.74% | +2.24 |
| curriculum full-FT (synth-pretrain → real FT) | 12.16% | +2.66 |
| synth-pretrain only (no real) | 14.15% | — |

**The curriculum *hurt*** — counterintuitively — and the reason is the base: `trocr-base-handwritten` is
already massively pretrained on ~700M synthetic + real lines *including IAM*, i.e. a domain expert before we
touch it. A narrow 4000-line diffu-synth pretrain then **degrades** that strong representation (catastrophic
forgetting toward the synth distribution), and 2000 real lines can't fully pull it back. Two sub-findings:
(a) a TrOCR trained on **synth alone** still reads real IAM at ~14% — the synth carries genuine signal; and
(b) **LoRA-adapt forgot less than full-FT** (11.74% vs 12.16%) precisely because it trains only ~1% of the
params, preserving more of the base.

Takeaway: the curriculum pays off from a *weak / from-scratch* base (where synth is the model's first real
exposure), **not** from a base that already owns the domain — synth-heavy training just overwrites the
expert init (textbook catastrophic forgetting; iso-step, so it's not a budget artifact). For a strong base
with ample real data the upside is structurally small; **the decisive win is the low-resource regime, now
measured (−40% at 200 real, above)** — diffu's actual use case.
(Drivers: `scripts/exp_htr_curriculum.sh`, `exp_htr_curriculum_lora.sh`, `exp_htr_lowres.sh` — removed; in git
history at `c53d30e`.)

### Caveats (still honest about scope)

Both regimes are now wins: **low-resource −40%**, and **data-rich −1.50 pt at 2:1** (~15% relative), each
5/5 seeds after the step-matched settle. Scope notes:

- **Always step-match and seed.** The one hard-won lesson: comparing arms at fixed *epochs* let synth-heavy
  arms train more and overfit, which *inverted* the ratio curve (fake "2:1 hurts"). And single-seed point
  estimates carry a ±0.3 pt SE — the provisional 9.50% baseline was a lucky-low draw. Never rank without bands.
- **A strong, already-IAM-expert base** still gains ~15% relative at 2:1 — the upside is real, just smaller in
  absolute terms than the starved low-resource regime.
- **Harmonization is unnecessary.** Fidelity probe: the real recognizer reads synth at 2.90% (cleaner than
  real); photometric harmonization ≈ raw at *both* 0.25:1 and 2:1. The synth's value is data diversity, not
  photometric realism — the covariate-shift hypothesis is falsified for this task.
- **Not yet swept past 2:1** (where dilution must eventually bite), **no confidence-filtering**, and
  **out-of-domain by construction** (a *Swedish* model on *English* IAM via LoRA) — so IAM is a lower bound;
  diffu's native Swedish domain reads far cleaner (below).

### Contrast: native-Swedish synth reads near-real

Diffu's actual use case is its **native domain**, where it is strongest: generated Swedish lines read at
**2–5% CER through the real Riksarkivet HTRflow pipeline** — essentially indistinguishable from real
documents (see the HTRflow end-to-end validation). So the IAM result is a lower bound on what in-domain
augmentation should deliver. The plain conclusion: **clean diffu synth is a strong HTR augmentation when
real data is scarce (−40% at 200 real)**; against a strong, near-saturated recognizer with ample in-domain
real data, the upside is structurally small and adding *too much* synth hurts (image covariate shift, not
label noise). Getting more from it means staying in-domain, **harmonizing synth photometrics to real**, and
confidence-filtering — future work.

*(Data-hygiene note: the `diffu-htr` project registers `htr_real` under two config rows — a
logging/registration artifact. The metrics form a single clean 5-epoch series, so the numbers above are
unaffected.)*

## The experimental arc

The generator's development, in one table. The through-line: a long from-scratch 64px base that plateaued
*legible-but-smudged with no guidance path*, then a jump from the **CFG-null dropout** that unlocked
classifier-free guidance and **glyph-LINE** conditioning beating glyph-CONCAT, producing the released
model at ~2% *guided* CER. Two side-paths (Stage-2 GAN, recognizer-loss-from-scratch) were tried and dropped.

| Run | Key config delta vs base | final / val_loss / gen_cer (guided) | Takeaway |
|-----|-------------------------|-------------------------------------|----------|
| `exp_base_weekend` | 64px, cond_dropout 0, 3-GPU, 415k steps | 0.080 / 0.085 / gen_cer 0.61 (min 0.49; no guided) | Long from-scratch base; plateaus ~50–60% **unguided** CER, no CFG-null → no guided path. The legible-smudge baseline. |
| **`exp_sd35_128`** | line_height 128, cond_dropout 0.1, text_dropout 0.1, batch 48, 2-GPU | 0.109 / 0.077 / gen_cer 0.34, **gen_cer_cfg 0.02 @120k** | **RELEASED** as HF `Gabriel/diffu_sd35_128` step_120000. 128px + CFG-null fix → ctrl_gap ~1.0–1.2, ~2% **guided** CER. The project win. |
| `exp_sd35_fast` | config unlogged; ~64px full recipe, 145 img/s, 310k | 0.092 / 0.091 / gen_cer 0.49, gen_cer_cfg 0.02 | 64px "fast" variant: ~2% guided CER, but unguided stays ~0.49 (resolution-limited). |
| `exp_line_cfg` | glyph **LINE** cond, cond/text_dropout 0.1, 68k | 0.093 / 0.085 / gen_cer 0.674, ctrl_gap 0.101 | Glyph LINE: lower CER **and** ~5× higher control-gap than concat → adopted in sd35. |
| `exp_concat_cfg` | glyph **CONCAT**, glyph_line off, 68k | 0.105 / 0.085 / gen_cer 0.752, ctrl_gap 0.021 | Glyph CONCAT: higher CER, near-zero control gap → rejected. |
| `exp_recog_ON` | init base 360k, recog loss ON (warm), lr 5e-5 | 0.148 / 0.126 / gen_cer 0.30 (min), gen_cer_cfg 0.35 | Recognizer loss from a **warm** start reaches markedly lower CER than OFF (short run). |
| `exp_recog_OFF` | init base 360k, recog loss OFF, 14.5k | 0.072 / 0.084 / gen_cer 0.48 (min), gen_cer_cfg 0.67 | Control for the recog ablation; higher CER on both guided and unguided. |
| `exp_stage2_wordcritic` | Stage-2 GAN word-critic, init 360k, 51.8k | 0.067 / 0.090 / gen_cer 0.56 (min), gen_cer_cfg 0.51 | Adversarial finetune did **not** beat base CER (~0.49); GAN path abandoned. |
| `exp_stage2_disc` | Stage-2 GAN discriminator, init 360k, 17.2k | 0.088 / 0.087 / gen_cer 0.57 (min) | Discriminator GAN also failed to improve CER; short. |
| `exp_stage2_v2` | Stage-2 GAN, 450 steps (aborted) | 0.077 / — / — | Aborted — inconclusive. |
| `exp_whitepad` | init 360k, white_pad_prob 0.4, lr 5e-5, 60k | 0.068 / 0.084 / gen_cer 0.62, gen_cer_cfg 0.25 (min) | White-pad latent finetune: modest guided-CER gain vs base, but ~10× worse than sd35_128. |
| `exp_A/B/C/D` | from-scratch, eff_batch 32, ~5–6k steps, style-pooling variants | ~0.11–0.13 / ~0.11 / gen_cer 0.75–0.79 | A–F sweep too short to rank style pooling (all ~0.75–0.79 @5k). |
| `exp_E_recog0p25` / `exp_F_recog1p0` | from-scratch + recog 0.25/1.0, ~2.8k | E 1.22 / F 3.28 loss / gen_cer 0.80–0.83 | Recog-from-scratch destabilizes (loss 1.2–3.3) → recog needs a **warm** init. |
| `dainty-sunset-0` | repa off, fill_ratio_cond off, 44.5k | 0.081 / 0.080 / gen_cer 1.55 (garbage) | Disabling repa+fill_ratio collapses generation (CER > 100%). |
| `brave-forest-1` | 256-fixed-line overfit probe (config logs as dainty's twin) | 0.001 / 0.0003 / gen_cer 0.014 | A deliberate **memorization** run (256 lines, val⊆train + style-ref leak) — measures memorization, not generation. **Not a quality signal.** |
| `calm-river-2` | Stage-2 GAN + repa/fill off, 29k | 1.92 / 1.89 / gen_cer 4.97 | Early Stage-2 smoke, diverged — failed run. |
| `eager-mountain-3` | eff_batch 32, 50 steps | 1.62 / — / — | 50-step smoke, inconclusive. |
| `iam_scratch` / `iam_full` / `iam_lora` | Teklia/IAM-line, 128px, `log_cer=False` | val_loss 0.073 / 0.088 / 0.095 (no CER) | IAM adaptation arms — see the IAM section; no legibility logged. |

### What actually moved the needle

- **CFG-null dropout unlocked guidance.** The base (`cond_dropout=0`) had no classifier-free-guidance path
  at all. Adding `cond_dropout=0.1 + text_dropout=0.1` is what makes `gen_cer_cfg` meaningful; the released
  model reaches a control gap (`ctrl_gap`) of ~1.0–1.2 and ~2% guided CER. **This is the single decisive lever.**
- **Glyph LINE > glyph CONCAT** on both CER and control-gap, so LINE was adopted.
- **Resolution is not the CER lever.** Both 64px (`exp_sd35_fast`) and 128px (`exp_sd35_128`) reach ~2%
  *guided* CER; unguided CER is ~0.49 at 64px and ~0.34 at 128px — 128px helps the unguided smudge modestly
  but does **not** fix it. 128px was chosen for the released model for stroke detail, not a CER jump.
  (Do not read the transient single-eval draws — the released model's unguided CER is noisy across ~0.08–0.85.)
- **Recognizer loss helps only warm.** From a warm start it lowers CER; from scratch it explodes training
  (loss 1.2–3.3 vs ~0.11 for the non-recog controls).
- **Stage-2 adversarial never beat the base CER (~0.49)** and was abandoned.

### Throughput notes

Median throughput across the 24 runs spans ~17–145 img/s. A few honest observations:

- **The recognizer loss is the cleanest throughput lever in the data**: a same-batch, same-init A/B
  (`exp_recog_OFF` 49.0 vs `exp_recog_ON` 26.8 img/s) shows a **1.83× slowdown**, corroborated by the A–F controls.
- **128px is ~2× slower** than the 64px fast path (`exp_sd35_128` 70 img/s vs `exp_sd35_fast` 145; the IAM
  128px fine-tunes run ~22 img/s at batch 8).
- **DDP utilization is high but scaling efficiency is not logged.** Run-owned GPUs sit compute-bound at
  **97–100% utilization**; a naive per-GPU throughput calc suggests ~83% scaling at 3 GPUs (`exp_base_weekend`)
  and ~92% at 2 GPUs (`exp_whitepad`) — treat these as estimates, not measured scaling. Because the box is
  shared, some per-GPU system readings reflect co-tenant jobs, not the run.
- **Config caveat:** every run that *logs* `compile`/`grad_checkpoint` (22 of 24) has `compile=False` and
  `grad_checkpoint=True`; `exp_sd35_128` and `exp_sd35_fast` use a different, near-empty config schema and
  log no system metrics, so their throughput cannot be attributed from trackio. The known `torch.compile`
  (~1.9×) and latent-cache (~1.36×) wins live in `docs/PERF_AUDIT.md`, not in these logged configs. (The latent
  cache was removed on 2026-10-19.)

## Reproduce

> **Note (2026-10-19):** the drivers below were removed; recover them from git history, e.g.
> `git show c53d30e:scripts/exp_iam.sh`. Drop `--glyph-line --no-style-tokens` from their commands (both flags
> were removed; their behaviour is now the only option) and add `--no-ink-focal` (these runs trained without
> the ink-focal loss, which is now on by default). `exp_sd35_128/step_120000` is no longer available.

### IAM adaptation (3 arms, project `diffu`)

`scripts/exp_iam.sh` (removed; in git history at `c53d30e`) runs all three arms against the released base
(`checkpoints/exp_sd35_128/step_120000/model.safetensors`), logging to trackio project `diffu` under run
names `iam_lora` / `iam_full` / `iam_scratch`:

```bash
scripts/exp_iam.sh launch          # all arms detached: GPU1 = lora→full (sequential), GPU2 = scratch
# or one arm in the foreground on a chosen GPU:
scripts/exp_iam.sh lora    1
scripts/exp_iam.sh full    1
scripts/exp_iam.sh scratch 2
```

All arms share `--hf-dataset Teklia/IAM-line --hf-val-split validation --glyph-line --no-style-tokens
--batch-size 8 --no-cer --track --val-every 250 --sample-every 500`. The LoRA arm adds
`--no-repa --lora --lora-rank 32 --lora-alpha 64 --lr 3e-4 --steps 4000` (→ a ~48 MB adapter); full uses
`--no-repa --lr 1e-5 --steps 6000`; scratch uses `--lr 1e-4 --warmup-steps 2000 --steps 60000` (random
init, REPA on).

### Downstream HTR A/B (project `diffu-htr`)

Two scripts (`scripts/exp_htr_ab.sh`, `scripts/exp_htr_ab_train.sh` — removed; in git history at `c53d30e`),
run in order. The first materializes real IAM, generates synthetic lines from the diffu LoRA
adapter, and builds the mixed manifest; the second fine-tunes the two TrOCR recognizers in parallel and
evaluates both on held-out real test:

```bash
scripts/exp_htr_ab.sh [adapter_dir]   # default runs/iam_lora/final; waits for the adapter + a free GPU,
                                      #   builds runs/htr_ab/{real,test,real_plus_synth.jsonl,synth}
scripts/exp_htr_ab_train.sh           # TrOCR(real) on GPU1, TrOCR(real+synth) on GPU2, then CER on real test
```

Knobs via env vars: `N_REAL=2000 N_SYNTH=4000 N_TEST=1000 EPOCHS=5 HTRBASE=microsoft/trocr-base-handwritten`.
Final CER numbers land in `runs/htr_ab/results.txt`; per-epoch curves land in trackio project `diffu-htr`
(runs `htr_real`, `htr_realsynth`).

### Viewing

```bash
uv run trackio show --project diffu       # generator: base, released 128px, ablations, IAM arms
uv run trackio show --project diffu-htr   # downstream TrOCR augmentation A/B
```
