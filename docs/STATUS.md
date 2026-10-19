# Diffu — Where We Stand (snapshot 2026-07-13)

> **Note (2026-10-19):** a dated snapshot, kept as written. The current state and next steps are in
> [`HANDOFF.md`](HANDOFF.md). Since this snapshot: `Gabriel/diffu_sd35_128` is no longer on the Hub (its
> `step_120000` checkpoint stayed on the old box); every checkpoint now carries a `config.json` that all
> loaders read, which retires the `--line-height` footgun below; `--no-style-tokens` and `glyph_concat` were
> removed (pooled-only style and the glyph-line encoder are the only options now); the one-off experiment
> scripts were removed (git history at `c53d30e`).

## TL;DR (current)
- **Released:** `exp_sd35_128/step_120000` (128 px; glyph-line content + pooled style + SD3.5 qk-norm +
  the CFG-null fix) → [`Gabriel/diffu_sd35_128`](https://huggingface.co/Gabriel/diffu_sd35_128)
  (2026-07-05). ~2–5 % guided CER at `cfg_scale=5`; load with `diffu.pipeline.DiffuPipeline`
  (footgun: pass `--line-height 128` everywhere — the config default is now 128 but old callers pinned 64).
  Supersedes the 64 px [`Gabriel/test_sd35`](https://huggingface.co/Gabriel/test_sd35).
- **Training:** `exp_sd35_128` KILLED 2026-07-05 at step 216k — it had converged ~100k; GPUs 1+2 are
  free (GPU 0 hosts a foreign job, `rerun_plus_olof`). Next run: the conditioning fix
  (glyph conditioning must be pad-invariant — the 07-13 discovery that batch composition changes output
  quality blocks width bucketing until fixed + retrained), then Flow-OPD distillation (`RL_PLAN.md`).
- **Scale-out:** 1M pages = the `diffu-runner` Ray pipeline — generate once into a line bank
  (~3.6 lines/s/GPU), then compose pages on CPU in bank mode; page composition becomes a CPU-day.
- **Inference stack:** bf16 end-to-end + TF32 + dynamic regional compile (~0.37 s/line); quantization
  measured a LOSS on this model (`docs/PERF_AUDIT.md` §0f) — ship compiled bf16.
- **Guards:** all loaders assert the text-conditioner actually loaded (no more silent gibberish from a
  mismatched checkpoint); CLI/eval take arch flags instead of inheriting live-run config defaults.
- **Experiments (trackio):** see [`docs/EXPERIMENTS.md`](EXPERIMENTS.md). IAM adaptation 3-arm — **LoRA ≈
  full fine-tune** on flow-loss (Pearson r=0.994, ~1/100 params, a 48 MB adapter), ~9× cheaper than
  from-scratch. Downstream HTR augmentation A/B **first showed synth HURTING (9.5 %→11.9 %)** — but that was
  two stacked bugs, not a verdict: a **canvas-width bug** (`_WIDTH_BIAS` 0.55, tuned for 64px, gives ~29
  px/char at 128px → looping; `synth.py` bypassed it entirely) and an **undertrained adapter** (50 % of the
  4k-step run was LR warmup). Both fixed (`_WIDTH_BIAS`→0.45 + biased-width synth + `ink_crop`; warmup 150 +
  English gen_CER gauge → adapter renders at ~1 % guided CER). **Verdict after fixing — synth helps in BOTH
  regimes, and MORE synth is better:** low-resource (200 real, diffu's actual use case) is a decisive win —
  **25.4 %→15.5 % (−40 % rel)** at 200 real + 2000 synth. Data-rich (2000 real), once measured correctly
  (`--max-steps`-matched **5-seed** sweeps), is *monotone*: real-only **9.95 %** → 0.25:1 **9.33 %** → 2:1
  **8.45 %** (**−1.50 pt, ~15 % rel, 5/5 seeds**, t=−10). The earlier single-seed "more synth hurts (2:1→10.13 %)"
  was an **epochs-not-steps artifact** (synth-heavy arms trained 3× longer and overfit). A **fidelity probe**
  shows synth is **label-clean** (real recognizer reads it at **2.90 %**), and photometric **harmonization ≈ raw
  at every ratio** (0.25:1 *and* 2:1) — so the covariate-shift hypothesis is falsified; the synth's value is data
  diversity, not realism. Only the synth-heavy **curriculum** (pretrain-only, no mixing) backfires (forgetting
  on an already-IAM-expert base). Stage-3 **LoRA adapters (`--lora`/`--lora-scale`) now wired through
  diffu-page, diffu-runner, and diffu-studio** (studio gets a UI control; diffu-page renderers default to the
  released arch).

---

# Historical snapshot (2026-06-25)

A current-state snapshot distinct from the running history in `scratchpad/PROGRESS.md` / `scratchpad/EXPERIMENTS.md` (both frozen).
**Goal:** generate handwriting line-images that are legible, text-following, and style-faithful
(Swedish historical), for HTR data augmentation. Success metric = CER from a Swedish HTR recognizer.

## TL;DR
- **Best model we have:** `checkpoints/exp_base_weekend/step_360000/ema_model.safetensors` (the weekend
  base; what `app.py` serves). It writes legible-ish historical cursive.
- **The usable recipe (confirmed by hand):** base ckpt + **tight canvas (auto-width)** + **CFG ≈ 3–4**.
- **We cannot currently trust our gen_CER numbers** (broken eval — see §3). Fix the ruler before any rerun.
- **Next:** Phase 0 (fix eval) → Phase 1 (content-decoupled style + white-pad + loss rebalance) → ONE rerun.
- **Stack (2026-06-25):** torch 2.12.1+cu130 · diffusers 0.38 · transformers 5.12.1 · accelerate 1.14.
  Backbone is now **SD3.5 attention** (`qk_norm` on; dual-attention wired-but-off — text-blind, redundant
  with REPA, overfit risk). `flow.sample_timesteps` reuses diffusers' `compute_density_for_timestep_sampling`.
  The next from-scratch run is SD3.5 — the SD3.0 `exp_base_weekend` checkpoint won't load into this config.

## Width brittleness — RESOLVED (2026-06-25)
- **Root cause (width sweep on base EMA):** the model writes cleanly at ~20–24 px/char but OVERFLOWS
  (fills the canvas with repeated glyphs) above ~28 px/char. The app's auto-width landed at 256px =
  **32 px/char** for an 8-char word — just inside the overflow zone. *That* was "auto-width doesn't work".
- **`exp_whitepad`** (60k, `--white-pad-prob 0.4`, single variable) **did NOT fix it** — the fair
  wide-canvas A/B (both raw, cfg 0 & 3.5) shows base and white-pad overflow identically. White-pad gives
  blank-tail *examples* but the weak `fill_ratio` scalar gives no reliable "stop" signal → insufficient alone.
- **FIX (1 line):** `generate._WIDTH_BIAS → 0.55` (~15 px/char = the GT **median** density, measured
  image_w/len over 2000 real lines). A CER sweep over {30,24,18,15} px/char on varied-length lines is
  **monotonic — mean CER 0.41 / 0.18 / 0.09 / 0.037** — tighter is strictly better down to the corpus
  density, no cramming. The old default (30 px/char) was catastrophic overflow (0.41). (Absolute CER is
  `trocr-base`/Goodhart-optimistic; the *relative monotonic trend* is the solid finding.)
- **Root cause = the model has NO length control** — it fills any canvas, never learned "write N chars and
  stop." The geometry heuristic is a stopgap; the real fix is to **train `fill_ratio` as a length channel**
  (+ variable-width / white-pad), which goes into the big run. App + CLI auto-width now "just work"; no manual length needed.
- For the huge run, the deeper version (white-pad + `fill_ratio` as its own channel + padded-not-stretched
  glyph render) would make ANY canvas robust — but it's no longer urgent; the geometry fix solves the product.

## 1. Confirmed WORKING
- **Base checkpoint** `step_360000` EMA — coherent cursive; the established sweet spot (generalises better
  than the 400k+ checkpoints).
- **Classic CFG** (`cfg_scale` ≈ 3–4) — the text-following lever. Lower CER than unguided everywhere.
- **Auto-width / tight canvas + ink-crop** — sizing the canvas to the text length makes output markedly
  more legible. The model was trained on tight crops, so a too-wide canvas is out-of-distribution and it
  overflows. Always deploy with auto-width.

## 2. Confirmed NOT working / dropped
- **Auxiliary guidance term** — removed entirely (2026-06-24): broken on current checkpoints, every caller already
  passed 0. Code, config, module, tests, and UI slider deleted.
- **Discriminators** (PatchGAN placeholder → RoPE word-critic + line-critic, `exp_stage2_wordcritic`) —
  **flat**, never moved gen_CER — discriminators are not the mechanism here.
- **Recognition loss** (`exp_recog_ON` vs `OFF`, killed 2026-06-24) — NOT a clean win: gen_CER gap is
  **Goodhart-contaminated** (train recognizer == eval recognizer), recog CE stayed **flat ~0.44** (not
  falling), and held-out **val_loss climbed +59%** (real generative degradation). From-scratch recog had
  already failed (`exp_E/F`: white/speckle). Warm-start avoided that collapse but still didn't earn its keep.

## 3. Measurement is BROKEN — don't trust the numbers yet
- **n=4** fixed sample batch decides every A/B → within noise.
- **Goodhart** (the differentiable recog-loss is removed; this now applies to the GRPO reward, see
  `RL_PLAN.md`): a reward/loss recognizer must be **held out** from `htr_recognizer_eval`, or it optimises
  its own judge.
- **Three generation geometries** (train fixed-width / `eval_cer.py` len·24 / deploy natural·1.15+crop) →
  different `fill_ratio` → different ink. The famous **0.55-vs-0.16 gap is largely a width/eval artifact**,
  not a clean CFG win — but `0.16` (auto-width) is the honest *deployable* number.
- **No style-fidelity metric at all** → style is unmeasured, so unmanaged.

## 4. Structural limitations (need a rerun to fix)
- **Style/content entanglement.** Style = raw DINOv3 features of the reference → they carry the ref's
  *glyph shapes*. Tokens-on ⇒ model copies the reference's text; tokens-off (`--no-style-tokens`, current)
  ⇒ no style at all. No content-decoupled style encoder exists. *(2026-10-19: `--no-style-tokens` removed —
  pooled-only is the only dino mode; `--style-canvas` is the new in-context attempt.)*
- **Background mismatch.** `writer_id` is actually the *page id* → the style ref always shares the target's
  paper, so the model never learns to separate hand from paper; and the crop masks the background to white,
  erasing the paper the goal wants.
- **Wide-canvas overflow.** `white_pad_prob = 0.0` → the model never learned to leave a blank tail; it
  fills the canvas → overflow on wide canvases. Auto-width is the workaround.
- **16-ch Qwen VAE latent** is general-purpose, not legibility-shaped (vs DiffInk's CTC-shaped latent).

## 5. Reference teardowns (DiffInk)
- The **content conditioner + a legibility-shaped latent** are load-bearing; the **discriminator is not**.
- **DiffInk** (hard data) bakes a **CTC OCR loss into the VAE** so the latent is character-decodable — the
  precedent for fixing our content gradient at the latent level.

## 6. The plan
- **Phase 0 — fix the ruler (code only, no GPU, do first):** one shared generate-geometry (auto-width+crop)
  across train/eval/app; **N ≥ 64** held-out eval set; **hold the eval recognizer out** of any recog loss;
  micro-average + ink-trim everywhere; add a **style-fidelity metric**; then re-judge the base on the fixed
  ruler. Converts "can't trust numbers" → "now we can."
- **Phase 1 — corrected config (one conceptual change = fix style entanglement):** content-decoupled style
  encoder (writer-contrastive / content-adversarial) → style tokens back ON without copying; **REPA
  neutralised/retargeted** (it currently re-leaks content via the style DINO); **white_pad on** + fill_ratio
  promoted to its own channel; glyph-line render padded not stretched. Data left unchanged for now.
- **Phase 2 — ONE rerun**, warm-started from `step_360000`, judged on the trusted ruler.

## 7. Dead-code cleanup status
- **Done:** the auxiliary guidance term fully removed; **CANINE `TextEncoder` subsystem + `use_canine`/`text_encoder` removed**;
  `ctc_weight_max`/`content_supervision` removed; `eval_cer.py` now imports the shared `char_error_rate`.
  (`ruff` clean, imports verified.)
- **Breadcrumb stubs (kept by choice — document intended work):** `style_shots`, `length_aware_shift`,
  `timestep_dist`, `extra_chars`, `cer_filter_threshold`.
- **Keep (not dead):** discriminators, `glyph_concat`, resampler, glyph content encoder, RoPE, all scripts.
  *(2026-10-19: `glyph_concat`, the per-character content encoder, `rope=False` and the one-off experiment
  scripts were removed; in git history at `c53d30e`.)*

## 8. Forward plan (final, 2026-06-25)

**Done this session:** width fix (`_WIDTH_BIAS=0.55`, measured), dead-code removal (the auxiliary guidance term, CANINE, 5 config
fields), `ARCHITECTURE.md` + interactive `architecture.html`, and a best-practice audit vs the REPA/flow papers.

**Architecture verdict (vs HASTE + HAE):** the *core* is correct & modern (cosine REPA align, layer 8, weight
0.5, DINOv3-L, per-token, SiT velocity + logit-normal). Gaps are scheduling/inference — see `ARCHITECTURE.md` §11.

**Next, in order:**
1. **Phase 0 — fix the eval ruler** (code, no GPU): one shared auto-width geometry + N≥64 held-out set + hold
   the eval recognizer out of any loss + a style-fidelity metric → re-judge the base honestly. *Gates everything.*
2. **Big-run prep** (needs your 3 data facts): build the multilingual ingest.
3. **Pilot gate:** 0.5–1M multilingual → Swedish fine-tune; must beat ~0.16 on the fixed ruler before GPU-weeks.
4. **The big run:** ~10M multilingual pretrain → GT-Lejonet fine-tune.

**Big-run config = settled recipe + this session's learnings + the paper best-practices:**
- glyph-LINE content + classic CFG + CFG-dropout; bigger model (~dim 1536, 24–30L); from scratch; Qwen 16-ch VAE.
- **Length control baked in:** `fill_ratio` as its own conditioning channel + variable-width / white-pad training.
- **REPA fixes:** *terminate REPA early* (~30–40% of steps), add a *timestep shift* (s≈√(C·H·W/4096)), 3-layer
  SiLU projector, EMA warmup; **CFG as a low-scale interval** (`t∈[0.3,0.7]`, scale ~1.5–2), re-swept after REPA.
- Style stays pooled (`--no-style-tokens`); content-decoupled style + REPA-E are *post-pilot* bets.
  *(2026-10-19: the flag is gone; pooled is the only dino mode.)*

**Blocked on you:** the 3 data facts — languages/scripts · HTR-vs-OCR split · image+text pair format.
