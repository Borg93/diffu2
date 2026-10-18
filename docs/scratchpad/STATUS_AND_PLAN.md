> ⚠ **FROZEN mid-June 2026 — historical record.** Current state: [`docs/STATUS.md`](../STATUS.md).
> Subsystems described below (DOG guidance, Stage-2 discriminators, CANINE, recog-loss) were later removed.

# Diffu — Where we are, how it works, and the plan

Plain-language status + plan. Read top to bottom; no ML background assumed.

---

## 1. The goal (one sentence)

Turn typed text into a **realistic, legible Swedish handwriting line-image**, so we can mass-produce
synthetic handwriting to train better handwriting-recognition (HTR) models.

Success is **not** "pretty pictures" — it's **legibility**: a recognizer must be able to read our
output back as the correct text.

---

## 2. The pieces of the model (what each part does)

When we ask for a line of handwriting, five parts cooperate:

| Part | Plain job | Trained? |
|---|---|---|
| **Glyph encoder** (Unifont) | Renders each letter to a tiny bitmap and encodes it → tells the model **WHAT** letters to draw, in order | trained |
| **Style encoder** (DINOv3) | Looks at a reference handwriting image → tells the model **HOW** it should look (slant, thickness, ink) | partly trained |
| **VAE** (Qwen-Image) | A compressor/decompressor. Squashes a 64×W image into a small "latent" grid the model works in, and expands it back | **frozen** (not trained) |
| **Backbone** (SD3 MMDiT, ~400M) | The brain. Starts from noise and, guided by glyph+style, removes the noise until a handwriting latent appears | **trained** (this is the hard part) |
| **Content guidance** | A sampling-time trick to sharpen output toward the requested text | n/a |

Flow of one generation: `text + style → glyph/style encoders → backbone denoises noise into a latent → VAE decodes → handwriting image`.

---

## 3. How a diffusion model generates (the intuition)

Think **"sculpting handwriting out of TV static."**

The backbone has learned, at every noise level, *"which direction makes this blob look a bit more like
the target handwriting."* It starts from pure static and nudges the image that way ~24 times until
handwriting appears. The **glyph** tells it which letters; the **style** tells it whose hand.

---

## 4. How training works — one step at a time

Training is just doing this, millions of times, on **real** handwriting lines:

1. Take a real handwriting line + its known text.
2. VAE-encode it into a latent.
3. Add a **random amount of noise** to that latent.
4. Show the model the noisy latent + the glyph + the style, and ask: *"which way is the clean version?"*
5. **Compare** the model's answer to the truth (the noise we actually added). The mismatch = the **loss**.
6. Nudge the model's weights to be slightly more right.

Repeat ~hundreds of thousands of times → the model learns to turn noise into handwriting.

### The catch (why our output is a smudge)
Step 5 only checks **pixels** — *"did your image match?"* It **never checks** *"are these actually
readable letters?"* So the model learns the **look** of handwriting (texture, slant, parchment) but not
real **spelling**. Result: realistic cursive that doesn't spell anything — a **smudge**.

---

## 5. Where we are right now (honest status)

- ✅ **Style is right** — realistic Swedish cursive on aged paper.
- ❌ **Content is wrong** — the words are an illegible smudge (confirmed by generating "göteborgs
  poliskammare" and looking; at best a word half-emerges).
- **Why:**
  1. **Trained from scratch** — the backbone started from random and must learn letter shapes from zero.
  2. **No legibility signal** in the loss (section 4's catch).
  3. A **learning-rate bug** wasted ~2/3 of the only real training run — **now fixed**.

---

## 6. Why we need an HTR model (and where it plugs in)

**HTR = Handwriting Text Recognition** = a model that **reads** handwriting → text (the opposite of what
we generate). We downloaded the Swedish National Archives' 2026 recognizer (**"Swedish Lion"**,
`Riksarkivet/trocr-large-handwritten-hist-swe-3-char`, now cached locally).

We need it for **two separate jobs** — note it is **not currently used in training**:

1. **MEASURE us (a number, not a vibe).** Run the recognizer on our generated line. If it reads back the
   right text → good; if gibberish → bad. This gives **CER** (Character Error Rate). Today we only
   eyeball; CER lets us say "0.9 → 0.4" and know if a change helped.
2. **FIX us (legibility signal).** The *differentiable* recognizer-loss (backprop CE through the recognizer)
   was tried and **collapsed** (`exp_F_recog1p0` → white/speckle); removed. The principled replacement is a
   recognizer **reward** via GRPO (no backprop through the recognizer) — see `RL_PLAN.md`, a post-base lever.
   (A pretrained handwriting model instead gets legibility from its pretrained weights + lots of training — see section 7.)

---

## 7. The plan — two honest paths

The smudge is mostly **"a big model trained from scratch, undertrained, with no legibility signal."**
Two ways forward:

### Path A — Keep our model, add legibility
- Add the HTR legibility check (section 6.2) to training; keep training the corrected run.
- **Time:** ~2–4 days on our GPUs to converge. **Risk:** from-scratch is hard; may cap at semi-legible.

### Path B — Fine-tune a pretrained handwriting model (recommended)
- Start from a **pretrained handwriting model** that ships weights which **already write legibly**
  (trained on English handwriting), and **fine-tune on Swedish**.
- **Time:** ~hours (~10k steps). **Risk:** low — it already knows how to write; we just teach it Swedish.

**Recommendation: Path B.** Starting from a model that already writes skips the expensive, uncertain
from-scratch phase that's causing the smudge. The cached recognizer (section 6.1) lets us measure either
path the moment it produces samples.

---

## 8. What's running right now

- **Training (`run4`):** our corrected model on **GPU 0 + 1** (~123 img/s, LR bug fixed, warm-started
  from the best earlier checkpoint `run3/step_35000`). This is the Path-A baseline.
- **App:** live at **http://localhost:7860** on **GPU 2** — loads the best model (`step_35000`) so you
  can type text and generate.
- **Recognizer:** Swedish Lion 2026 **cached** (2.0 GB) — ready for CER measurement / the legibility loss.

---

## 9. Time & hardware reference

Our data: **~299k training lines**. Throughput is fixed by the GPUs (~123 img/s on 2, ~185 on 3),
**independent of dataset size** — more data means *more diverse* epochs, not longer wall-clock.

| Scenario | Steps | 2 GPUs | 3 GPUs |
|---|---|---|---|
| Our planned run (100k steps) | 100k (~59 epochs) | ~40 h (1.7 d) | ~26 h (1.1 d) |
| From-scratch, actually converged | ~250k (~150 epochs) | **~4 days** | ~2.7 days |
| **Fine-tune a pretrained model (Path B)** | ~10k (~6 epochs) | **~4 hours** | ~2.5 h |
| One epoch over **1M** lines | — | ~2.3 h | ~1.5 h |

**Key fact:** dataset size (300k vs 1M) does **not** change time-to-converge (~days from scratch) — it
changes *quality*. The thing that changes *time* is the **starting point**: from-scratch = days;
pretrained fine-tune = hours.

---

## 10. Things we tried that turned out to be dead-ends (so we don't repeat them)

- **Tensor-Core / GPU-throughput optimization** (latent cache, profiling, bigger batch): only affects
  training *speed*, **not output quality**. Parked.
- The **LR-schedule bug** was real and is fixed — but that's training hygiene, not the quality fix.
