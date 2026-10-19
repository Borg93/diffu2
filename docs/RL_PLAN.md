# Legibility via recognizer-reward + Flow-OPD — post-base finetuning (Stage 3)

**Status:** plan, not implemented. The scaffolding that lived in `src/diffu/rl/` (reward/advantage/sde_logprob
implemented + tested; sampler/sft documented stubs) was **removed on 2026-10-19** — it is in git history at
`c53d30e`. The principled replacement for the differentiable recog-loss we
removed. The Stage-1 gate — a base that already follows the text — is **PASSED** by the released
[`Gabriel/diffu_sd35_128`](https://huggingface.co/Gabriel/diffu_sd35_128) (~2 % guided CER), so Stage 2 is
unblocked — RL/distillation cannot teach a model to spell from nothing, it only sharpens a
model that already mostly can.

> **Note (2026-10-19):** `Gabriel/diffu_sd35_128` is no longer on the Hub and its `step_120000` checkpoint
> stayed on the old training box, so a Stage 2 would start from a newly trained base.

> **This note now reflects the FULL Flow-OPD repo** (`CostaliyA/Flow-OPD`, read in full: `scripts/train_sd3.py`,
> `scripts/train_sd3_sft.py`, `scripts/train_sd3_opd_mix.py`, `flow_grpo/`, `config/`). The earlier version was
> based on a partial paste (only the SFT trainer) and got the headline method wrong — see "Corrections" below.

---

## TL;DR — where does this sit for us?

**It is a post-training / finetuning method, never a from-scratch trainer.** Every config loads a *pretrained*
SD3.5 base and trains a **LoRA** on top. It assumes the base already produces plausible images and only nudges
the distribution toward a reward. So for us the dependency chain is:

```
Stage 1  Base flow-matching pretrain on the 1M Swedish line pairs   ← the hard part, the prerequisite
            → a model that already (mostly) spells the requested text
Stage 2  Vanilla flow-GRPO with HTR (1−CER) reward   (reward_mode="task_only")
            → an "OCR/legibility expert" LoRA = a TEACHER
Stage 3  Flow-OPD: distill the teacher's dense per-step velocity field into a student
            (reward_mode="kl_only"/"gkd")   ← THIS is the actual "Flow-OPD" contribution
```

So to answer the question directly:

- **Not the beginning.** OPD does nothing for a model that hasn't learned to spell — there is no teacher yet
  and no good distribution to anchor to. Stage 1 (the base run) is non-negotiable first.
- **The immediate RL step is Stage 2 (`task_only` GRPO), not OPD.** OPD (Stage 3) needs a teacher, and the
  teacher is produced *by* Stage-2 GRPO. You cannot download a teacher — ours has custom conditioning (glyph +
  DINOv3 + 16-ch Qwen VAE), so an off-the-shelf SD3.5 LoRA won't load into it.
- **OPD's +8pt over GRPO comes from MULTI-teacher merging** (OCR + GenEval + DeQA + PickScore unified into one
  student). We have **one** axis that matters (legibility). With a single teacher, OPD ≈ distilling our own
  GRPO model into itself — largely redundant with just running GRPO longer. **Build Stage 2 first; only reach
  for Stage 3 if/when we have multiple reward axes to unify (e.g. legibility + style-fidelity + ink-realism).**

---

## What Flow-OPD actually is (full-repo reading)

It is `flow-grpo` (yifan123) extended with **On-Policy Distillation (OPD)**. One trainer, `scripts/train_sd3.py`,
parameterised by `config.train.reward_mode`. All four modes share the same on-policy spine:

**The spine (every mode).** Per outer epoch: the **student** samples `K=16` trajectories per prompt with an
**SDE sampler** (`sde_step_with_logprob`, `noise_level=0.7`, `num_steps=10`). The SDE noise is what makes the K
samples differ (a deterministic ODE step would give identical rollouts) and gives per-step `log_prob`s. Then it
re-evaluates each stored trajectory under the current policy and applies a **PPO clipped surrogate** on the
log-prob ratio:
```python
ratio       = exp(log_prob_now − log_prob_old)               # importance ratio (train_sd3.py:1193)
policy_loss = mean(max(−adv·ratio, −adv·clip(ratio, 1−ε, 1+ε)))
loss        = policy_loss + beta · KL_to_anchor               # beta=0.04
```
So the "GRPO loop my old note said wasn't in the paste" **is fully present** — `compute_log_prob` +
`sde_step_with_logprob` + ratio/clip. What changes between modes is **where `adv` (the advantage) comes from**:

| `reward_mode` | advantage source | what it is |
|---|---|---|
| `task_only` | group-normalised **scalar reward** (`PerPromptStatTracker`, OCR `1−CER`) | **vanilla flow-GRPO** — RL from the recognizer reward |
| `kl_only` | `kl_scale · (−KL(student_mean ‖ teacher_mean))` per step | **pure OPD** — push policy toward a **teacher's dense velocity field**, no scalar reward used |
| `mixed` | `task_advantage + kl_scale · kl_reward` | scalar reward AND teacher distillation |
| `gkd` | — (no ratio) — loss is directly `mean(KL(student_mean ‖ teacher_mean))` | **Generalized KD** — simplest on-policy distillation |

**Two distinct reference models** (this is the key structure the old note missed):

1. **`kl_ref_lora_path` — the TEACHER** (e.g. `jieliu/SD3.5M-FlowGRPO-Text`, itself made by `task_only` GRPO on
   the OCR reward). Forwarded on the **student's own latents** (→ on-policy) to produce `prev_sample_mean_ref_lora`.
   The OPD advantage is the **negative per-step KL** between student and teacher velocity:
   ```python
   kl_reward = ((prev_sample_mean − prev_sample_mean_ref_lora)**2).mean((1,2,3)) / (2·std_dev_t**2)
   advantages = kl_scale · kl_reward        # kl_scale = −1  →  advantage = −KL  →  move toward teacher
   ```
   This is "dense, trajectory-level, multi-teacher vector-field supervision" — a **full velocity field per
   step** instead of **one scalar per image**. That density is the paper's whole pitch (+18pt over base).

2. **base model (`disable_adapter()`) OR `mar_lora` — the ANCHOR** (MAR = Manifold Anchor Regularization). The
   `beta·KL_to_anchor` term keeps the student near a trusted distribution so the RL/distillation can't drift into
   illegible-but-high-reward artifacts. MAR generalises "anchor to base" to "anchor to a frozen **quality**
   teacher" (e.g. an aesthetic LoRA) so chasing OCR doesn't degrade visual quality.

**Cold start.** The student can be initialised by SFT or by **merging** expert LoRAs (`scripts/merge.py`) before
OPD — `train.lora_path`. Multi-teacher OPD (`train_sd3_opd_mix.py`, `mix_opd_8gpu`) **alternates datasets**, each
with its own `kl_ref_lora_path`, so one student inherits every expert's field.

---

## Corrections to the previous version of this note

The old note was written off the partial paste (only `train_sd3_sft.py`) and is wrong in three places:

1. **"We weren't given the GRPO/importance-sampling trainer."** ❌ — `train_sd3.py` **is** that trainer (PPO
   ratio/clip on SDE log-probs). The log-prob loop is present and portable (`sde_step_with_logprob` is
   scheduler-only).
2. **"What Flow-OPD does = best-of-group advantage-weighted FM + KL-to-base."** ❌ — that describes the *separate,
   simpler* `train_sd3_sft.py` (OnlineSFT). The **actual OPD method** is `reward_mode="kl_only"/"gkd"` in
   `train_sd3.py`: distil a **teacher LoRA's velocity field**. The old "Variant A vs Variant B" framing
   (advantage-weighted-FM vs importance-GRPO) conflated two orthogonal axes; the real axes are
   **{what the advantage is: scalar-reward vs teacher-KL}** × **{how it's applied: PPO-ratio vs direct-KD}**.
3. **"Build A first = SDE sampler + OCR reward + advantage-weighted FM."** Half-right but **mislabelled**: that
   first step is `task_only` GRPO (Stage 2), **not** Flow-OPD. OPD (Stage 3) additionally needs a teacher.

Still correct from the old note: recog-loss had to die (below); sample-cost makes this a finetuning phase, never
from-scratch; Goodhart guard on the reward recognizer; the architecture mapping table.

---

## Why the differentiable recog-loss had to go (kept — still the motivation)

Flow-MSE rewards *"ink that looks like handwriting,"* not *"spells the requested letters"* — a model sits at
gen_CER ~0.5 with beautiful texture. The previous fix differentiated a frozen TrOCR's CE **through**
`vae.decode(x0_pred)`. That **collapsed to white/speckle** (`exp_F_recog1p0`): backprop through a frozen
recognizer is adversarial — the cheapest way to lower its CE is a degenerate input. **The principled fix:** use
the recognizer as a **black-box reward** (forward only) and improve the generator via a reward-weighted policy
update — which is exactly Stage 2 (`task_only` GRPO). OPD then replaces even that sparse scalar with the
teacher's dense field.

---

## Mapping to OUR architecture (+ the teacher dependency)

| flow-grpo / Flow-OPD | Diffu (ours) |
|---|---|
| `StableDiffusion3Pipeline` (T5/CLIP, 4-ch VAE) | custom `Diffu` (MMDiT + glyph content + DINOv3 style + **16-ch Qwen VAE** + 2D-RoPE) |
| `sde_step_with_logprob` (scheduler-only) | **directly portable** — wrap `FlowMatchEulerDiscreteScheduler`; inject SDE noise into `Diffu.denoise_steps` |
| `sd3_pipeline_with_logprob` (full pipe) | **not reusable** — reimplement the sampling loop on our pipeline |
| PaddleOCR English, `1−dist/len`, target = `prompt.split('"')[1]` | **our Swedish HTR** (`read_lines`, `1−CER`), target = the **whole line** (no quote-split) |
| prompt = caption | `(text, style)` pair |
| `kl_ref_lora_path` = downloadable SD3.5 OCR expert | **must be OUR own Stage-2 GRPO LoRA** — custom conditioning means no off-the-shelf teacher loads |
| KL anchor via `disable_adapter()` | LoRA on MMDiT attn → same trick; full-finetune would need a frozen base copy (extra memory) |
| MAR = frozen aesthetic teacher | for us: anchor to the **base handwriting model** so legibility-RL doesn't kill ink-realism |

**Hyperparameters worth copying** (from `config/grpo.py`): LoRA + EMA(decay 0.9, every 8 steps); `noise_level=0.7`;
sampling `num_steps=10` but `eval_num_steps=40`; `K=num_image_per_prompt=16`; `timestep_fraction=0.99`;
`global_std=True`; PPO `clip_range=1e-4`; `beta=0.04`; `kl_scale=−1`; `lr=1e-4` (mix) / `3e-4` (single).

---

## Staged build plan for us

1. **Land Stage 1 first — DONE.** The gate (a base that *follows the text*, positive `ctrl_gap`) is passed
   by the released `Gabriel/diffu_sd35_128` (~2 % guided CER) — Stage 2 is unblocked.
2. **Stage 2 — `task_only` GRPO (do this before any OPD):**
   - SDE sampler on `Diffu.denoise_steps` (port `sde_step_with_logprob`; `noise_level≈0.7`, `K≈16`).
   - Reward = `1 − HTR_CER` from a **HELD-OUT** recognizer (Goodhart: the reward recognizer MUST differ from
     `cfg.aux.htr_recognizer_eval` — re-introduce the guard for the *reward*).
   - Advantage via `PerPromptStatTracker` (group-normalised); PPO clipped surrogate; `beta·KL_to_base` anchor;
     LoRA on MMDiT attention + EMA.
   - **Gate:** does Swedish `gen_cer_cfg` beat the base? If not, stop — not worth the compute.
   - Output: an **OCR-expert LoRA = our first teacher.**
3. **Stage 3 — Flow-OPD (only if Stage 2's gate passes AND we have ≥2 reward axes to unify):**
   - `reward_mode="kl_only"` (or `"gkd"` to start — simplest): teacher = the Stage-2 LoRA; advantage = `−KL` to
     its per-step velocity; anchor (`mar_lora`) = the base handwriting model.
   - Multi-teacher only if we actually train separate experts (legibility / style-fidelity / ink-realism) —
     `alternate_datasets`, one `kl_ref_lora_path` each, optionally merge-init.
   - With a **single** legibility teacher, expect little gain over Stage 2 — don't build it just to build it.

---

## Caveats

- **Sample-expensive:** `K` full denoises per prompt per step → a *finetuning* phase, never from-scratch.
- **Goodhart:** held-out reward recognizer, distinct from the eval gauge `cfg.aux.htr_recognizer_eval`.
- **Reward hacking:** the `beta·KL_to_anchor` (MAR) term is load-bearing — without it the model finds
  illegible-but-high-reward artifacts.
- **No free teacher:** our custom conditioning means the OPD teacher must be our own Stage-2 GRPO LoRA.
- **Diminishing returns single-axis:** OPD's headline win is multi-teacher unification; one legibility axis
  doesn't exercise it.
- **Our VAE/conditioning:** reuse only the scheduler-level SDE math (`sde_step_with_logprob`); the rest is ours.
