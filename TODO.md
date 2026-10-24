# TODO

What is known to be wrong or unfinished. Remove an item when it is done; add the commit that did it.

## Direction (decided 2026-10-21)

- FLUX.3 Klein (with our LoRAs) is the **research reference**: what quality is reachable, and the eval target.
- **diffu is the deliverable**: our own weights, trained only on our data, using FLUX.3's architecture and
  training recipe. The FLUX Kommunity License (§1a(iii), §4b) forbids distilling from FLUX.3, training on its
  outputs, or reusing its weights or autoencoder for another image model. Allowed: diffusers'
  `Flux3Transformer2DModel` (Apache-2.0 code) instantiated from a config with random init; Apache-2.0 text
  encoders and VAEs.
- Everything is diffusers-native (model classes, Modular Diffusers blocks and pipelines) and trained with NeMo
  AutoModel. No custom model code where diffusers has the class; no backward compatibility with the old code.
- **The product (decided 2026-10-23)**: millions of different synthetic documents to train VLMs on. One sample is a
  **full page with its ALTO/PAGE**. Any template (layout) x any text x any style (a reference document's hand, ink
  and paper), each chosen independently. Document kinds: handwritten running text, **tables and forms**, **printed
  and mixed** pages. Text: real transcriptions reshuffled, **other historical Swedish text**, **generated text**.
  Generated at scale by diffu on the Ray cluster's Blackwell GPUs (FLUX.3 at 50 steps: ~3.3 min per crop on the
  GB10, far too slow for millions). Workstreams this opens:
  - [ ] Full pages: stitch a page's crops (windows overlap; shared lines must agree) or draw at 2-4 MP.
  - [ ] Tables and forms: crops and captions for table cells (the crop builder skips them today).
  - [ ] Printed and mixed pages: printed GT pages to train on, and their templates.
  - [ ] Text: sources of other historical Swedish text; an LLM text generator in period style.
  - [ ] Style that transfers: the hand from the reference (v3d, then diffu), measured by `HandFidelity`.
  - [ ] Throughput: diffu with fewer steps, batched, FP8 on Blackwell; profile on the cluster.

## Repo structure

Done 2026-10-21.

- [x] diffu-page's layout and labels kept, the rest of the old diffu-page removed
      with their tests; `layout/__init__.py` exports only public names. Checked: 331 crop plans (4,488 lines) from 60
      random GT pages are byte-identical before and after.
- [x] Removed the line-pasting era: the rest of diffu-page, `diffu-runner`, `diffu-studio`, the old model `src/diffu`
      with `tests/`, `Makefile`, `slurm/`, diffu-era `scripts/` and `docs/`. All under the tag
      `diffu-mmdit-v1`.
- [x] Root `pyproject.toml` is a virtual workspace root; diffusers pinned to `e04f0c5` (the NeMo environment's FLUX.3
      commit); ruff bans `dataclasses.dataclass` (NeMo's dataloader configs are the one exception).
- [x] `jobs/setup_nemo_env.sh` installs diffu-page, diffu-gen and diffu-train into the NeMo environment.
- [x] README says what the repo is and how it is laid out.
- [x] Grouped by pipeline stage (branch `modular-structure`): `layout/` (+ `page_xml.py`, the one PAGE/ALTO
      XML reader, shared by the layout replay and the crop builder), `crops/`, `captions/`, `modular/`,
      `generator.py`, `labels.py`, `cli/` (the `diffu-generate` / `-evaluate` / `-sheets` commands),
      `data/` (incl. `nemo.py`). `jobs/spark_guard.sh` (was `scripts/`); `.gitignore` trimmed; `.rayignore` dropped.
      Checked: the 331 crop plans byte-identical again; all `_target_`s of the three NeMo configs resolve in NeMo's
      environment. On GitHub's main since 2026-10-21; the local checkout follows after tonight's queue (its scripts
      call the old paths and `scripts/spark_guard.sh`).
- [x] Tests for plan (reading order: body block before its margin notes) and restyle (restyled caption == the crop
      captioned in the target style; the v3 edit caption takes only the paper from the reference). 50 tests.
- [x] `experiments/` deleted on main (the local checkout keeps its copy until tonight's queue, which runs it), and
      the training configs moved into diffu-train (`packages/train/configs/flux3/`).
- [x] Run `jobs/setup_nemo_env.sh` once to drop the stale page-synth / diffu-gen / old diffu installs from the NeMo
      environment and install diffu, diffu-page and diffu-train (2026-10-21 21:50, with the local checkout on main).
- [x] `docs/PIPELINE.md`: the flow from page to labelled crops, training, and the training/generation consistency table.

## Workspace (2026-10-21, branch `modular-structure`)

- [x] Members by role: diffu-page (page data, our blocks, the page and crop pipelines, the CLIs), diffu (the model),
      diffu-train (NeMo glue), diffu-runner (Ray), diffu-studio (Gradio). Pilot LoRA config pruned.
- [x] Every step is a Modular Diffusers block (`diffu_page.blocks`): `LoadLayoutStep`, `RefillStep`, `PlanCropsStep`;
      `CropPromptStep`, `ReferenceGridStep` before a model's own blocks; `LineReadbackStep`, `LabelExportStep` after.
      (`CropLoopStep`, which drew every crop of a page in one call, was removed on 24 Oct: every entry point now draws
      crop jobs one by one.) The model is chosen by its directory (`pipeline.FAMILIES`, keyed by
      the blocks its `modular_model_index.json` names), so FLUX.3 Klein and diffu run through the same blocks. The
      generator class (`CropGenerator`) and diffu-gen are gone: the CLIs, the runner and the studio call pipelines.
- [x] diffu-runner on the crop pipeline: pages planned with `PagePlanBlocks` on CPU workers, one crop per block, GPU
      actors each holding a crop pipeline; Ray 2.59.0. Fixed: a page's crops shared a block, and an actor hands a
      block on only when it has drawn all of it, so Ray Data stopped the CPU smoke run after 30 minutes without
      progress (the same would hit a ~30-crop block on the cluster).
- [x] diffu-studio as a `gr.Workflow` canvas: a plan node (`PagePlanBlocks`) feeding one draw node per drawer (a model
      and its LoRAs; drawers on one model share it and switch its LoRAs), each returning a sheet of the crops and a zip
      of their labels; app view and REST endpoint come with it. The nodes sit where the data is plain values (plans as
      JSON); embeddings and latents stay inside each crop pipeline.
- [ ] Studio: a reference-image node (v3, style by example) into the draw nodes; a glyph-map node for diffu.
- [x] diffu: FLUX.3's architecture with our own weights, laid out like a diffusers model (see "diffu" below), registered
      as a model family: the crop and page pipelines draw with a diffu repo exactly as with FLUX.3 Klein (test).
- [ ] Page assembly: crops overlap (training windows share lines) and windows that would cut a line are dropped, so
      pasting a page's crops back would draw lines twice and leave holes. Needs a tiling plan for generation and crops
      drawn consistently with their neighbours (e.g. the neighbour as a reference image); see "Full pages".

## Code audit against writing-python (2026-10-21, branch `modular-structure`)

- [x] One PAGE reader for training and generation (the replay took a line's last Word's outline, estimated baselines
      ~0.3 pitch off and ignored Transkribus reading order); 60/60 GT pages now planned exactly as the builder cut
      them. One region vocabulary (PAGE's types).
- [x] Pruned the synthetic template engine (templates, sampler, flow, edits), PageSpec, compositor-only fields,
      words / printed / JSON export, and the tests that only covered them.
- [x] Typer + Rich CLIs with exit codes; typed settings; a failed crop or page no longer ends a run.
- [x] DRY: one collection key, one latent-grid rule, slot constants from geometry; pick(…, 1) crash; images closed;
      pathlib links; typed manifest metadata; docstrings; @override on framework hooks; no private cross-imports,
      relative imports, long functions or >3 positional args left.
- [x] pyrefly checks imports again: `missing-import` was off, so an import of a module that no longer exists passed
      the type check. Only NeMo and its torchdata (NeMo's environment only) are typed as Any (red-checked with a stale
      `diffu_gen` import).
- [x] ALTO `softwareVersion` asked for the version of `page-synth`, a package that no longer exists, and always
      wrote "0".
- [x] pytest `strict = true`; pytest-cov and `[tool.coverage.run]` for per-test coverage when pruning the suite.
- [ ] Rebuild `data_flux3/layout_crops_v2` (and steer_v2, style_pairs_v3, val_eval) with the fixed archive names
      before any v2 retrain or new derived set: 521 records carry the raw folder name, and restyle now refuses them
      (`StaleCaptionError`). Rebuilt as `data_flux3/layout_crops_v2b` (2026-10-23) for v3d; v2's sets stay as they
      were trained.
- [x] The one-off job chains in `jobs/flux3/` that ran (steer, pages smoke, v3, v3b, the control probe) are deleted
      (24 Oct; git history keeps them); v3c, its guidance run and v3d wait for the v3d decision.

## Modular Diffusers (checked against huggingface.co/blog/modular-diffusers and the diffusers source, 2026-10-21)

Tested end to end on CPU with diffusers' tiny FLUX.3 test model (`packages/page/tests/test_pipeline.py`, each
test red-checked); not yet run on the GPU with FLUX.3 Klein.

- [x] Our blocks around the model's own, declared like diffusers declares FLUX.3's: `pipeline.crop_blocks` (crop
      prompt, reference grid, the model's text encoder / VAE encoder / denoise / decode, read-back, labels; workflows
      `text2image` / `image_conditioned`) and `pipeline.PagePlanBlocks` (no model). The page pipeline
      (`page_blocks`) is gone since 24 Oct: diffu-generate draws each planned crop on its own, as the runner does.
- [x] One crop pipeline serves v2 and v3: the model's auto blocks choose reference-conditioned denoising when an
      `image` is passed.
- [x] TrOCR loads through `load_components` from the read-back block's `ComponentSpec`;
      `readback.materialize_position_tables` fixes its meta table. `load_pipeline` fails fast when a component
      did not load (`load_components` only logs).
- [x] LoRAs load with the model's own LoRA loader (`lora.load_nemo_lora`, alpha/r folded into lora_B), named
      adapters, `set_adapters`. NeMo is no longer needed at inference. All 21 module kinds of the real v2 checkpoint
      take the adapter.
- [x] Training and inference see the same conditioning (checked 2026-10-21 on the real FLUX.3 text encoder, CPU): NeMo's
      FLUX.3 cache encoder and FLUX.3's text encoder block give bit-identical prompt embeddings on the same device (chat
      template, 80-token padding, position ids, layers 4…32); v2's GPU-made cache vs ours on CPU: mean per-token
      cosine 0.997, bf16 arithmetic differing by device. Latents: same VAE `mode()`, patchify, batch-norm normalization.
      RoPE ids and the reference stream: identical in NeMo's flux3 adapter and diffusers' blocks (image
      (0, h·0.5, w·0.5, 0), reference l = -1, text (0, 0, 0, i), references at timestep 0 inside the transformer).
- [x] Fixed: v3's references (10,299 of 11,152 not 16-px aligned) were squashed onto the grid by NeMo's cache encoder
      in training but would have been reflect-padded by FLUX.3's preprocessing at inference. `ReferenceGridStep` now
      resizes a reference exactly as training did, before FLUX.3's VAE encoder (test red-checked).
- [x] CFG null: NeMo's dropout zeroes the text embeddings, FLUX.3 (BFL, diffusers) uses the empty prompt's. Our LoRA
      configs set `cfg_dropout_prob: 0.0`, so v2/v3 are consistent; diffu must pick one null for training and inference.
- [ ] Publish the blocks to the Hub (`diffusers-cli custom_blocks`) if others should use them; they import diffu-page.
- [ ] Guidance: we use 4.0, BFL's own code's default (the model card says 3.5). Sampling shift: diffusers 6.93, BFL's
      code 16.0 since 2026-10-17. Compare both on the eval set.
- [x] The text-encoder layer choice is `DrawSettings.text_layers` (the model's own when unset: FLUX.3 every 4th of
      Qwen3-VL's; diffu 4, 12, 20, 28 of Qwen3.5's, from its repo).

## diffu: FLUX.3 architecture, our weights

Research 2026-10-21 (reports: FLUX.3 recipe, NeMo feasibility; both read-only).

- BFL published no technical report for FLUX.3 / Klein: the architecture is fully in the code, most of the training
  recipe is not public. What is: "FLUX 3 builds on Self-Flow" (bfl.ai/blog/flux-3; arXiv 2603.06507): two noise
  levels per sample, an EMA teacher (0.9999) on the cleaner input, a cosine feature loss (student layer 0.3·depth to
  teacher layer 0.7·depth, weight 0.8); it reports clearly better text rendering than plain flow matching. Candidate
  for diffu (needs an EMA teacher: a NeMo recipe subclass).
- Architecture details that matter: the 5 early blocks are per mode (image, reference, text each attend only within
  their mode), then 28 shared joint blocks (62% of the weights); one modulation per mode per stage; no biases; no
  pooled text vector; CFG null = the empty caption. Captions: Klein trained on its upsampler's output (200–350-word
  English prose, quoted text verbatim, JSON layout boxes [y0, x0, y1, x1] on 0–1000).
- Timestep shift: FLUX.2-latent experiments found logit-normal with shift 4.63 best for training at 256², scaled by
  √(token ratio) for other resolutions; NeMo has only a constant `flow_shift` (no resolution-dependent shift).
- The autoencoder: FLUX.3's `ae.safetensors` is byte-identical (sha256) to the autoencoder BFL also publishes under
  Apache-2.0 (32 channels, semantically regularized like REPA-E). Legal question: if that license holds, diffu can
  use FLUX.3's own autoencoder instead of Qwen-Image's 16-channel VAE.
- NeMo trains diffu without a NeMo patch: the flux3 adapter and `Flux3Transformer2DModel` take channel count and
  text width from the config. Use an init directory (modular index + our initialized weights) with `mode: finetune`
  and no `peft:` block, so every rank loads identical weights (pretrain mode inits per rank and HSDP never resyncs).
  Our caching code writes NeMo's cache layout (Qwen-Image VAE: `vae.encode(x.unsqueeze(2))`, per-channel
  mean/std, patchify to 64 channels; Qwen3.5-4B: with 8 stacked layers NeMo's rule picks exactly its 8
  full-attention layers). No EMA in the diffusion recipe; validation is held-out loss only; consolidated export with
  `diffusers_compatible: true` should load with `from_pretrained` (verify in the smoke run).
- Biggest risk: the text cache. v2's FLUX.3 cache is 524 GB for 11,329 crops (layout prompts ~1,150 tokens). For diffu
  at 8 / 4 / 2 / 1 stacked layers: ~47 / 24 / 12 / 6 MB per crop, 4.7 / 2.4 / 1.2 / 0.6 TB per 100k crops (Leonardo:
  1 TB `$WORK`, 20 TB `$SCRATCH` with a 40-day purge). Options: ≤ 2 layers, shorter prompts, or encoding text during
  training (~+45% compute, 9 GB per GPU).
- Measured 2026-10-21 (Qwen3.5-4B tokenizer + chat template, all 11,329 v2 captions): median 1,077 tokens (p95 1,483,
  max 3,810); the prose paragraph is only ~162 of them, the JSON layout rows (each repeating the style words around
  the line's text) the rest. A compact prompt for diffu (paragraph + `[id, box, text]` per line) is median 554 tokens:
  at 4 stacked layers 12.4 MB per crop, 1.2 TB per 100k crops; at 2 layers 0.6 TB. The line texts are most of what
  remains; rendering them as glyphs into a reference image (FLUX.3's reference stream) would leave only the style
  paragraph as text and spell exactly by construction.
- Order: measure Qwen3.5 token counts and latent statistics on ~1k crops → init-directory script → caching script →
  YAML (finetune mode, flux3 adapter, logit-normal shift ≈ 3, fp32 weights / bf16 compute, `dp_replicate_size: 8`,
  our interleaved dataloader, consolidated diffusers export) → GB10 smoke run → diffu inference blocks → EMA /
  Self-Flow recipe subclass → Leonardo.

- [x] **Model** (the root package, `src/diffu`, laid out like a diffusers model: `configs.py`, `init.py`, `latents.py`,
      `modular_pipelines/diffu/` with encoders, decoders, `DiffuAutoBlocks`, `DiffuModularPipeline`):
      `Flux3Transformer2DModel` from our presets with random adaLN-Zero init; FLUX.3's denoise blocks reused unchanged.
      diffu-S 618.0M (hidden 1536, 2 per-mode + 12 joint: FLUX.3's per-mode share of depth), diffu-M 1636.7M (2048,
      4 + 16); Klein is 5.54B. `diffu-init` writes a model repo that `from_pretrained` and NeMo's finetune mode load;
      `diffu-export` makes a run's last checkpoint a model repo, its scheduler at the run's `flow_matching.flow_shift`.
      Start with S.
- [x] **Text encoder**: Qwen3.5-4B (Apache-2.0), chat template with thinking off, 80-token windows, hidden layers 4,
      12, 20 and 28 stacked (10,240 wide: the early orthographic, middle compositional and late layers; never 32, the
      post-norm state). Its BPE hides 3 of 4 å/ä/ö inside merged subwords and no layer spells an unseen subword
      linearly, so exact spelling may need the glyph map (below).
  - [ ] Standardize each stacked layer per channel with statistics from ~1k prompts (a GPU pass), stored in the repo
        and applied by both the text encoder step and the cache writer (layer RMS grows 16× from layer 4 to 32).
- [x] **VAE**: Qwen-Image's 16-channel VAE (Apache-2.0) through our encode / decode steps (`diffu.latents`: its
      per-channel mean/std, 2×2 patchify in FLUX.3's channel order, 64 channels). Reconstruction gate on 23 of the 24
      held-out crops (CPU, 2026-10-21): read-back CER 0.107 real vs 0.106 reconstructed (gate |Δ| ≤ 0.01; per crop
      −0.014 to +0.011).
  - [ ] Measure the latent mean/std on our pages; if they differ from Qwen-Image's config, write them into the repo's
        `vae/config.json` (`diffu.latents` reads only that).
- [ ] **References**: FLUX.3's reference stream replaces the old pooled DINO style vector (which gave generic style).
      A glyph map (each line's text rendered into its box, along the PAGE baseline) at l = −1 spells by construction
      and fixes line length; a style reference (another region of the same volume) at l = −2. Both already pass
      through `image=[glyph_map, style_ref]`, but FLUX.3's preprocessing center-crops each of several references to
      the 16-px grid: render the glyph map at exactly the target's grid size. Renderer: Unifont covers all 158
      characters of our transcriptions.
- [x] **Training glue**: `diffu_train.cache` writes NeMo's text-to-image cache with diffu's own encoder steps
      (`prompt_embeds`, `encode_latents`, FLUX.3's image processor). A test pins every cached crop to exactly what
      diffu's pipeline makes at generation. `diffu_train.cache_diffu` is its command (NeMo's bucket ids);
      `packages/train/configs/diffu/diffu_s_smoke.yaml` trains every weight in NeMo's finetune mode from
      `diffu-init`'s repo (float32 weights, bfloat16 compute, logit-normal with shift 6 at 1 MP, no loss weighting);
      `jobs/diffu/run_smoke.sh` chains init, cache (2,000 v2 crops, ~46 GB), a 5-step run, 2,000 steps x 8, and
      the held-out eval of the export as a diffu repo.
- [ ] **E1 smoke run**, running since 16:03 24 Oct (`jobs/diffu/run_smoke.sh`): the cache of 2,000 crops took 40
      minutes (diffu's text encoder in transformers' slow fallback, see Inference speed), the 5-step run passed, 2,000
      steps x 8 crops take about 3.5 hours (1.3 crops a second). Pass: the loss falls, the export loads as a diffu
      repo, read-back CER on the held-out crops goes down from the random init's. Its eval samples at the 6.0 it
      trains at: the running job's step 5 copies the init repo's index, which got a scheduler at 6.0 by hand on
      24 Oct (to drop once the eval has run); later runs export with `diffu-export`.
- [ ] **Training, after the smoke run**: the empty-prompt null swapped in by our dataset (p 0.1; NeMo's own dropout
      zeroes the embeddings; `diffu_train.dropout.CaptionDropout` does it for FLUX.3's image-edit items since
      2026-10-23: reuse it for diffu's text-to-image items, with the empty prompt cached by diffu's own text encoder),
      zero latents for a dropped reference, then the EMA / Self-Flow recipe subclass
      (per-token timesteps through three forward pre-hooks; bit-identical to the stock forward with uniform
      timesteps).
- [ ] **Staged experiments** (design study 2026-10-21): E1 GB10 smoke + overfit (~3–4 h) → E2 conditioning ablation
      at 512² (text only / glyph map + style prompt / both) → E3 Self-Flow vs plain flow matching → E4 style reference
      (swap-test copy rate ≤ 5 %) → E5 main run (~12 h on 32 A100s for diffu-S). Each judged on held-out read-back
      CER (≥ 200 crops, bootstrap CIs) against the real crops and the FLUX.3 LoRA on the same crops.
- [ ] **Sampling shift**: diffu samples at the shift it trained at (`diffu-export` writes it into the repo's
      `scheduler/scheduler_config.json`); sweep 3–7 by editing that config, it costs nothing.
- [ ] **Leonardo**: compute nodes have no internet; pre-fetch Qwen3.5-4B and the Qwen-Image VAE at the pinned
      revisions (a local repo copy with the files loads from the copy).
- [ ] **Data**: GT-Lejonet, 103,285 pages (~3M lines). Never FLUX.3 outputs.
  - [ ] Crops from every page, not 40 per collection: ~6 crops per page (v2: 1,866 pages -> 11,329 crops), so
        ~600k crops. `build_crops` reads PAGE only: wire the ALTO reader (`layout/xml_replay.py`) into it for the
        100k+ ALTO pages, each with its scan image.
  - [ ] Augmentation that stays true to the scan, 2-4 variants per crop (~1-2M): shifted windows, line pitch
        56-72 px, other bucket shapes, greyscale copies. No flips or rotations; refilled text is not a training
        target (no real image).
  - [ ] Rebuild with the fixed archive names first (the 521 stale captions, see Code audit).
- [ ] **E5 on Leonardo**: no job or config yet: an sbatch script (cineca-hpc skill), the full-data config, the
      offline model prefetch below.
- [ ] **Compute**: estimated ~8×10¹⁹ FLOPs for diffu-S (10M samples at 256², 5M at 512², 2M at 1024²; GB10
      measured at ~25 TFLOPS effective on the v2 LoRA). GB10 alone ~40 days: too slow, use it for smoke runs and
      short ablations. Leonardo, 8 nodes (32 A100s): roughly 6–30 h. First a short GB10 run at 256² to see loss
      and read-back CER fall, then book compute.
- [ ] **Eval**: the same held-out crops and metrics as the FLUX.3 LoRAs (per-line CER, layout, style score), so
      diffu and the reference are compared head to head.

## Labels (ALTO / PAGE are only output formats; the label data is what must be right)

- [x] Caption archive names: two GT collection folders end in U+F028 (a trailing space as macOS SMB encodes it), so
      `collection_title` fell back to the raw folder name: 521 of v2's 11,329 training captions (RA arjeplogs
      tingslag, RA carl fredrik pahlmans resejournaler) name "RA … 20260220\uf028". Fixed on `modular-structure`
      (+ test). v3's captions name no archive (unaffected); a v2 retrain would learn the right names.

- [x] Reading order across columns: labels follow the source blocks in page order, top to bottom in each
      (`CropPlan.reading_order`).
- [x] Line polygons: checked against the GT, not a defect (GT bands are 1.6x the line pitch; our slots 1.0x;
      slot IoU 0.72 vs traced ink 0.37). To match the GT convention, widen the slot.
- [ ] Narrow columns get no crops (under the 640 px minimum when a margin note narrows a column). Decide: crop
      narrow columns side by side, or leave such pages out.

## Templates from scans (YOLO26)

Any scanned page as a template, not only pages with PAGE/ALTO ground truth.

- [ ] `DetectLayoutStep` in diffu-page, an auto block beside `LoadLayoutStep` chosen by its input (a page image instead
      of a PAGE/ALTO file). It runs Riksarkivet/yolo26s-handwritten-regions-2 to find text regions, then
      `yolo26s-handwritten-lines_within_regions-2` to find the lines within each region: HTRflow's two stages. It
      writes the same `Layout`, with line boxes from the masks and baselines estimated from each mask's lower edge, so
      refill, crop planning, the captions' layout control and the labels run unchanged.
- [ ] Region roles. These models have one class each (`textregion`, `textline`), so every region would be running
      text. Map a model's classes to our roles (marginalia, heading, page-number, ...) through a table, so a
      multi-class layout model drops in; until one exists, infer margin notes from geometry (narrow regions beside the
      main column).
- [ ] Check it on GT pages: detected lines against their PAGE lines (line recall, box IoU, baseline offset).
- [ ] The weights are CC-BY-4.0, but `ultralytics`, which runs them, is AGPL-3.0. Internal use is fine (HTRflow
      already depends on it); serving the studio publicly with it needs AGPL compliance or an Ultralytics licence.
- [ ] Studio: an image input next to the template file.

## FLUX.3 reference runs

- [x] v2 eval, steering test (style by name), first generated crops (`runs/flux3_pages/smoke`) and the v3 decision:
      done 22 Oct (below).
- [ ] Full fine-tune of FLUX.3 Klein instead of a LoRA: possible (NeMo's finetune mode), not on the GB10
      (~100 GB of weights, gradients and optimizer state): Leonardo, 4x A100 with FSDP, and ~100k+ crops to be worth
      it. Last resort after v3/v3b and a bigger LoRA (rank 64-128, longer); never shippable (license), so compute
      is better spent on diffu.
- [x] LR schedule: in v2 and v3 the LoRA's B weights stop growing after ~1000 of 1600 steps (|B| 21.4 -> 21.8):
      the cosine decay to min_lr 1e-5 ends learning early. v3c (overrides) and v3d (its config) train at a constant
      LR after warmup, the diffusers Klein trainer's default; judge by val_loss.
- [x] val_loss: v3b, v3c and v3d score NeMo's validation loss (64 val-split pairs, fixed noise, every 100 steps, in
      trackio); v3d's config holds it, and its smoke run scores it once.
- [ ] Job scripts: the 5-step smoke runs need `--lr_scheduler.lr_warmup_steps 1` and an empty checkpoint dir (both
      fixed 2026-10-21, `069bd47`, `4a8d3b4`); keep any new job script on that pattern.
- [x] v3b (its job chain is in git history): v3's pairs, half the captions also naming the archive and look. The
      words win: named in words, the template's own paper overrides the reference's (by-example paper dE 8.3 to the
      reference vs 8.9 to the other references, against v3's 5.0 vs 10.5).
- [ ] Hand fidelity (`diffu_page.captions.hand`, in every `diffu-evaluate` results.json since 2026-10-23;
      `diffu_page.cli.hands` rescored v3 and v3b): v3's held-out drawings follow their reference's paper (CIE76 dE
      6.2 vs 10.8 to the other references; the real crop is 3.6 from it) and partly its slant (9 deg vs 12; real 6),
      not its ink (30.2 vs 29.8; real 6.7): they draw near-black, bolder strokes whatever the reference's brown ink
      (checked by eye, `runs/flux3_style_v3/eval_val_step_1599/022_*`). Watch to_reference against
      to_other_references in v3c's and v3d's results.
- [x] Queue since 11:56 23 Oct, done: the control probe ran; v3d stopped while caching (15:54 23 Oct) for the
      per-hand LoRA pilot, the user's call, and is not trained (whether to, now that four pages per writer suffice,
      is the user's call). The plan as it stood (`runs/queue_v3d_first.log`): the control probe (done; its job chain
      is in git history: base FLUX.3 draws v3c's held-out crops with plain, described and contradicting captions: do
      described words move the hand at all?), then v3d (`jobs/flux3/run_v3d.sh`: described
      captions from the rebuilt crops, caption dropout 0.1, constant LR, 3,000 steps; results ~16:00 24 Oct), diffu
      E1, v3c from zero (the control for v3d: same-hand references without the description; trackio
      `flux3_v3c_lora_rerun`), and v3c at guidance 2.5 and 3.5 (`run_guidance_v3c.sh`). v3c's first run was stopped
      at step 280 of 3,000 to give v3d the GPU (val_loss 0.6333 at step 99, 0.6273 at 199, as v3b at those steps).
- [x] Control probe, first two parts (base FLUX.3, v3c's 21 held-out crops, same seeds): captions that describe
      the reference's measured look carry its colours with no training (carry-over paper colour 0.36 -> 0.86, ink
      darkness 0.02 -> 0.50, ink hue 0.01 -> 0.70; a real crop of the same hand 0.93, 0.98, 0.90; CER 0.036 ->
      0.041), not the slant (0.15 -> 0.18; real 0.75) or the letterforms. The ink stays flat and opaque (contrast
      to the paper 149 -> 146 grey levels, variation inside strokes 9.5 -> 9.2; real crops 103 and 17): the
      "pasted on" look is a separate problem. Third part, captions describing a different reference than the
      image: the drawings follow the words (carry-over from the words' look: paper colour 0.83, ink darkness 0.77,
      ink hue 0.97) and not the image (-0.73, -0.29, -0.86); slant follows neither (0.02, 0.06); CER 0.043. The
      caption decides the colours; the reference image barely counts. A described caption is a reliable colour
      control, and a wrong one overrides the image.
- [x] Realism test (runs/realism_test, scripts in the session scratchpad; measures.json, figures on the status
      page): one new document (Bergskollegium E3_5 1691-92 template 00128, text of 00339 + 00200, the 1715
      Överståthållarämbetet letter book as the example), two windows, same seeds, base FLUX.3. A, our caption: ink
      stands out 158 grey levels (paper minus ink; the example 102, real crops 103), varies 8.5 inside the strokes
      (18.9; 17.2), ink colour dE 38 to the example, CER 0.016. B, FLUX.3's prompt upsampler (pup, i2i_bbox, boxes
      locked): reads the example as the page to edit (described its date, signature, page number and the
      operator's glove, "keep identical pixel for pixel"); the drawings copied the glove and, in the tall window,
      "1715", "708" and ghost lines of the example's writing (CER 0.30); it changed 21 of 42 line texts (restored);
      1.6-4.4 min a window. Out. C, a second FLUX.3 edit pass over A ("keep every word and line, change only how
      the page looks", the example's measured ink/paper hex): ink 86 / 16.5 / dE 7, CER 0.025, but "1000:de" became
      "1000cde", the paper more stained than the example's (dE 8 -> 11); a second draw, ~2 min a window. PiD (pinned
      from-clean path of the former diffu_page.upscale, decoder and Gemma weights downloaded 23 Oct, load checked:
      456 tensors, none missing): alone it sharpens and leaves the ink flat (160); after C it restores the
      sharpness the pass blurs (17.8); 11-26 s a window at 4x. None of them changes the hand: the same round hand,
      slant 36-44 deg, in every variant. Next: the second pass and PiD as page-pipeline blocks (edit caption
      written from the measured look, no upsampler), checked on more windows and with fewer steps.
- [x] Second-pass control probe (runs/realism_test, measures.json; same windows and seeds as A; means of 2 windows:
      CER / ink stands out / ink varies / ink dE to the example; the example 102 / 18.9 / 0): A 0.016/158/8.5/38;
      C0, redrawn with no look words, 0.026/153/10.2/39 (redrawing alone does nothing: the words do it); C1, our look
      words directly, 0.020/135/12.3/29; C2, C1 at 28 steps, 0.021/133/12.1/27 (28 steps as good as 50); C1q, BFL's
      Klein edit shape with every line quoted, 0.025/120/18.7/21 (the most natural in-stroke variation); C, the
      prompt upsampler's wording of the same instruction (plain i2i), 0.025/86/16.5/7: by far the strongest. It states
      the edit as an instruction ("Render the writing as faded brown iron-gall ink in the exact color #81674E...
      Replace the plain background with an aged cream sheet in #FAECD6..."), ours described the target. Next: C's
      wording as a template (no 9B model), at 28 steps, over the per-hand drawings (H's ink 122 / 14.4 is better than
      A's, but its ink and paper are the volume's, dE 25 and 17 to this example).
- [x] Per-hand LoRA pilot (the user's call, 23 Oct 15:50, after "it's not following the reference image"): FLUX.3
      Klein Base text to image, trained only on the 1715 Överståthållarämbetet letter book's own pages (160 windows of
      36 pages; v2's recipe, 700 steps of 2, constant LR; session scratchpad hand/). It learns the hand. On 8 held-out
      windows of 4 other pages (same captions and seeds; median distance to the real hand, lower is closer):
      untrained FLUX.3 CER 0.015, slant gap 11.0 deg, pen width 0.25, letter size 0.26, spacing 0.24, ink dE 16.1,
      paper dE 5.0; step 350: 0.056, 2.0, 0.16, 0.51, 0.12, 24.3, 6.2; step 700: 0.061, 2.5, 0.15, 0.33, 0.12, 22.5, 6.1.
      All 16 held-out windows at step 700: CER 0.073 against 0.036 for the real windows. By eye: the clerk's chancery
      cursive, flourished month headings, raised abbreviations, names in a rounder Latin hand as on the real pages;
      the realism test's new Bergskollegium document drawn in the hand (runs/realism_test/H, CER 0.085 and 0.040).
      Costs: legibility (harder cursive and some drawing slips: lines to tag unclear or drop by read-back CER) and
      darker ink than the real (the second pass's job). The hand is in by step 350 (~50 min of training).
- [x] Pahlman's travel journals 1825-26, the same recipe (runs/flux3_hand_pahlman; 292 windows of 37 pages, 18 of
      3 pages held out; 8 scored): untrained FLUX.3 CER 0.019, slant gap 14.0 deg, pen width 0.29, letter size 0.26,
      spacing 0.51, ink dE 16.0, paper dE 5.1; step 700: 0.049, 4.0, 0.12, 0.27, 0.26, 24.3, 2.1. Slant closer in 8 of
      8 windows. By eye Pahlman's thin leaning hand with his dashes at sentence ends. It also writes text no label
      asks for: the caption's date range "1825–1826" as a running header at the top right (2 of 8 windows), where the
      journal's real pages carry page numbers the ground truth does not transcribe (see Data checks).
- [x] Per-hand LoRAs on two microfilms, the same recipe (runs/flux3_hand_{kopparberg,landskap}; 211 windows of 37
      pages and 208 of 36; 8 held-out windows each; median distance to the real hand, untrained -> LoRA): Bergskollegium
      Stora Kopparbergs bergslag 1743: slant 4.0 -> 3.5, pen width 0.92 -> 0.15, spacing 0.94 -> 0.38, letter size
      0.40 -> 0.77 (worse), CER 0.013 -> 0.037 (real 0.053); Landskapshandlingar 16th c.: slant 24.5 -> 9.0, pen width
      1.32 -> 0.28, spacing 1.58 -> 0.27, letter size 0.50 -> 0.57, CER 0.117 -> 0.165 (real 0.375). By eye the
      clerks' hands (long s, gothic letters, dash rules, the film's edges); the films' stains and fading not yet. No
      unlabelled writing (hand/unlabelled.py: outside ink is film edges and bindings, as on the real windows). Four of
      four volumes hold: one LoRA per volume (230 volumes) is the way to real hands. Letter size worse on 3 of 4.
- [x] The chain on the per-hand drawing (runs/realism_test H -> HC, the second pass in C's wording at 28 steps, 67-73 s
      a window -> HC_pid; means of 2 windows; CER / ink stands out / ink varies / ink dE / paper dE to the example):
      H 0.062/122/14.4/25/17, HC 0.076/112/16.7/23/16, HC_pid 0.084/116/18.1/27/18 (real crops 103/17.2, the example
      102/18.9). The second pass adds the ink's unevenness, not this page's colour: H's ink and paper are the
      volume's average. For a volume's own hand the second pass is optional; it earns its time where the ink must
      match one page, or on untrained drawings.
- [x] Reference guidance probe (runs/refguide_probe; untrained FLUX.3, the described captions and seeds of the control
      runs/probe_described/described, its first 11 crops; scratchpad refguide/compare_refguide.py). Carry-over from
      the example (paper yellowness / ink lightness / ink yellowness / slant) at reference scale 1, 2, 3.5:
      0.84/0.41/0.66/0.32, 0.93/0.53/0.72/0.38, 1.00/0.63/0.74/0.47; median distance to the example's hand (slant,
      pen width, letter size, spacing, ink dE, paper dE): 5/0.30/0.69/0.21/12.4/3.3, 6/0.30/0.56/0.18/12.6/2.1,
      4/0.18/0.55/0.15/11.1/1.65; CER 0.027, 0.031, 0.025. Peak 53 GB, ~3.5 min a crop (three passes). By eye: at
      3.5 the example's mottled paper, grain and heavier strokes arrive, the letterforms stay FLUX.3's neat cursive.
      It carries the page around the letters, not the hand: keep it for paper and ink (maybe instead of the second
      pass), not as a way to copy a writer. 11 crops: carry-over steps of ~0.05 are near noise; the trend is monotonic.
- [x] FLUX.3 Klein's own reference patterns (runs/ref_tests; the reference guidance probe's 11 crops, captions'
      layouts and seeds; scratchpad reftests/). (a) Canvas edit, the example page at its full size told to replace its
      writing with our lines in exactly the same hand: 10 of 11 rewritten on the example's own paper, stains, binding
      and glove, in FLUX.3's own neat hand; 1 of 11 (a 16th-c. microfilm) returned unchanged; the lines land outside
      our boxes, so read-back and box measures do not apply. (b) The prompt upsampler (plain i2i) writing the caption's
      opening: asked for a new page it invents text and headings (Latin, German, Danish), which get drawn (CER 0.123);
      cut to its style sentences (reftests/trim_pup.py): carry-over paper/ink/slant 0.75/0.32/0.29 against our
      described caption's 0.84/0.41/0.32, paper dE 6.0 against 3.3 (cream under grey films). (c) Examples at 1 MP:
      0.86/0.59/0.38, by eye the same drawings. None carries the letterforms; without training one image carries the
      page (paper, ink, grain; reference guidance 3.5 most), not the writer. Side idea: a blank page from
      GT-RA_tomma_sidor as the canvas for realism.
- [x] Caption probe (runs/caption_probe; untrained FLUX.3, the described control's 21 crops and seeds): (scribe)
      "another page of the same volume, by the same scribe: copy the handwriting exactly"; (ink) the ink described
      physically (two hexes, hairlines, downstrokes, soaked into the fibres); (script) "Swedish Kurrent written with a
      quill: looped ascenders, the long s, e like a narrow n". Carry-over paper/ink/slant: described 0.86/0.50/0.18,
      scribe 0.85/0.51/0.22, ink 0.77/0.43/0.16, script 0.84/0.45/0.17; by eye the same letters in all four. No wording
      moves the letterforms: v3d keeps the described captions; the hand must come from training.
- [x] Writer identification (scratchpad writer/: build_lines.py, train_id.py; data_flux3/writer_id/lines.npz). A small
      CNN learns 37 collections from 300 real lines each (layout_crops_v2b train; lines normalised to handwriting only,
      64 px high), then names held-out real lines and the per-hand evals' lines (same text and boxes). Top-1, grey /
      black-and-white strokes only: held-out real 0.77 / 0.59; 1715 real 0.87 / 0.85, LoRA 0.83 / 0.83, untrained 0.00
      / 0.02; Pahlman 0.97 / 0.95, 0.97 / 0.91, 0.00 / 0.00; Kopparberg 0.74 / 0.83, 0.94 / 0.96, 0.00 / 0.00;
      Landskapshandlingar 0.31 / 0.21, 0.76 / 0.49, 0.00 / 0.00. The LoRAs' lines are named as their collection as
      often as the real ones (more on the films: a cleaner, more typical hand than the faded originals); untrained
      FLUX.3's never (they go to Göteborgs poliskammare, the neatest 19th-c. hand). Machine-level writer identity
      holds; a human examiner's blind test is still to do (the LoRA lines are bolder and cleaner than the real).
- [x] Table probe (runs/table_probe, data_flux3/table_probe; scratchpad tables/stage_tables.py): 6 windows of real
      ruled registers (one TableRegion, <= 30 cell lines), untrained FLUX.3 three ways with one seed: column rules +
      top/bottom from the cells' borders, a full grid, no rules. All three draw the table (aligned columns, numbers
      right, letter headings centred; CER 0.01-0.09, the Sollentuna ledger 0.30-0.52); asked for, the rules and the
      grid are drawn where the boxes put them; unasked, FLUX.3 adds a column rule itself but never a row grid.
- [x] Tables, print and ruled lines in generation (branch tables): the loader reads a line's PAGE @production
      (printed/typewritten -> printed; unknown values fail), table cells in both forms (Transkribus TableCell
      @row/@col, PAGE 2019 TextRegion + TableCellRole; a TableCell without row/col fails) and SeparatorRegions as
      ruled lines; the planner keeps a table as one region (windows of whole rows, pitch from row to row) and clips
      the ruled lines into each window; captions describe printed lines ("Printed text reading ... in a serif
      typeface in black printer's ink"), cell lines ("... in a table cell") and a rule_k row per ruled line, plus a
      table and a rule sentence; PAGE labels write a TableRegion of nested cell TextRegions with TableCellRole,
      SeparatorRegions and @production, validated against PRImA's 2019 schema (now vendored; the 2013 one is gone);
      ALTO writes cells as TextBlocks and rules as GraphicalElements. A page without tables, print or rules plans
      byte-identically (300 random ground-truth pages compared: 299 identical, the 300th a table page that main
      planned as nothing). Red-checked tests: the loader (3), the planner (3), the labels (schema, round trip, ALTO).
- [ ] Tables in training data: build_crops still leaves table cells out. Needs a ruled-line detector on the scans
      (the ground truth has no SeparatorRegions; cells are not rules), a filter for the 135 of 249 tables whose cells
      overlap (Sollentuna's "tables" are paragraphs in one cell), then table crops for the per-hand LoRAs.
- [ ] Typeface for printed lines: PAGE TextStyle@fontFamily (Fraktur vs antiqua) into the printed row's desc.
- [x] Form probe (24 Oct; runs/form_probe, scratchpad forms/): 5 PAGE templates of our own that exist nowhere in the
      ground truth (a letter on printed letterhead, a moving certificate, a list form, a parish register, a ledger),
      every line printed or handwritten, every ruled line given; untrained FLUX.3 and Pahlman's hand LoRA, one seed.
      Both draw every form: print in the typeface asked for (Fraktur, antiqua), handwriting in the fields and cells,
      the rules and grids where the boxes put them, line CER mostly 0. Pahlman's LoRA, trained on handwriting only,
      keeps the print printed and moves only the hand (to its 1820s hand) and the paper (darker, a scan edge, faint
      show-through); untrained FLUX.3 writes modern copperplate on clean paper. The LoRA adds faint ruling under the
      list form's rows that was not asked for.
- [ ] Read-back of printed lines: the handwriting TrOCR reads the correctly drawn letterhead "KONGL. MAJ:TS OCH
      RIKETS KAMMARRÄTT" at CER 0.71-0.80, and one-letter cells ("a", "b", "i") at 1.0, so a printed line or a single
      character would be tagged unclear though drawn right. Read printed lines with a printed-text OCR (or don't tag
      them from this read-back), and skip the unclear tag for lines too short to read.
- [ ] Forms and mixed print/hand still need ground truth to train and evaluate on (none in ours).
- [x] Damage test (24 Oct; runs/flux3_hand_landskap_dmg, scratchpad damage/compare_damage.py, damage_sheet.jpg): the
      Landskapshandlingar LoRA retrained on captions that state each window's measured damage (same recipe, 700
      steps), on the 8 held-out windows. Medians, real / plain-caption LoRA / damage-caption LoRA / untrained FLUX.3 on
      the damage captions: uneven light 0.170 / 0.103 / 0.137 / 0.061, stains 0.123 / 0.011 / 0.074 / 0.017, film
      edges 0.210 / 0.075 / 0.180 / 0.066, ink fading 0.091 / 0.060 / 0.040 / 0.286. The words carry the damage only
      through training, and only part of it: by eye the drawings stay clean and crisp beside the film's heavy water
      stains and blurred ink, and the ink gets darker, not fainter. The hand holds (slant gap 9.0 -> 7.0, spacing 0.27
      -> 0.19; letter size 0.57 -> 0.72). Damage as a caption knob works but is weak; the second pass or a
      damage-specific LoRA may do more.
- [x] Per-hand LoRA from few pages (24 Oct): the 1715 recipe on the windows of 4 and of 10 of its 36 training
      pages (data_flux3/hand_overstat_p4, _p10; 23 and 44 windows), 350 steps of 2, on the same 8 held-out windows
      as the 36-page LoRA. Four pages carry the hand: by eye the three LoRAs draw one writer, and the writer
      identifier (retrained with these lines) names the 4-page LoRA's lines as the 1715 writer 82% of the time, the
      10-page 81%, the 36-page 81% (83% at 700 steps), the real held-out lines 87%, untrained FLUX.3 0%. Medians of
      the hand distance (slant, stroke, letters, density): 4 pages 1.5/0.16/0.82/0.15, 36 pages 2.0/0.16/0.51/0.12,
      untrained 11.0/0.25/0.26/0.24; letter size is the one measure that more pages improve. One drawing of the
      10-page LoRA wrote the archive's name from the caption's prose ("records of Överståthållarämbetet") as a
      heading: a caption word outside the layout rows can be drawn.
- [ ] The tokenizer warns that FLUX.3 Klein's Qwen3-VL tokenizer has "an incorrect regex pattern" (transformers;
      `fix_mistral_regex=True`). Training and generation load it alike, so they agree; check whether BFL's pipeline
      sets the flag before changing it.
- [x] Read the profile (`runs/flux3_pages/smoke/profile/trace.json`, 2.8 GB: two 1472x640 crops, 50 steps, FLUX.3
      Klein Base **with the v2 LoRA unfused** (the deleted pages smoke chain passed `--lora v2=...`), torch 2.10 from
      the NeMo env, which warns that the GB10's capability 12.1 is outside its build; pytorch-skills:profile-model as a
      streaming pass, session scratchpad profile/analyze_trace.py, 32 s). GPU-bound: the GPU is busy 99.5% of the run.
      Denoising is 97.5% of the kernel time (2.66 s a step with guidance's two passes), text encoding 0.4%, the VAE
      decode 0.5%, TrOCR's read-back 1.6%. Matmuls 41% (cutlass_80 sm80 kernels, one GEMM 23%; nvjet_sm121 10%),
      elementwise 38%, copies 9%, flash attention 8% (SDPA flash for 7,624 calls, efficient for 146). Corrected by the
      docs/speed audit: ~92k of the 141k aten::mm calls are the unfused LoRA's thin GEMMs (cutlass_80_wmma 7.8 s +
      the second cutlass_80_tensorop row 6.7 s), plus PEFT's scale-and-add over every adapted output: the LoRA is an
      estimated 15-20% of GPU time; base GEMMs ~33%, other elementwise ~25%, cat/copies ~10%. Re-baseline on the repo's
      torch 2.14.1 before optimizing (see Inference speed).
- [ ] Full pages: generate at 2–4 MP, or stitch windows that share a reference.
- [ ] Style score: a classifier on real crops that says whether a crop matches the requested archive or reference.

## Audits of 23 Oct 2026 (four parallel read-only audits: writing-python/testing-python, the diffusers skills, BFL's FLUX skills, the official docs and inference speed)

Fixed (branch `labels-blocks`, each with a red-checked test): labels group lines by role and template block (a crop's
two columns stay two regions); ReferenceGuidance extends diffusers' CFG, cannot leak its hook after a raised pass,
refuses a reference-free crop; the HTR read-back pinned to revision aa79fcb; DrawSettings' guidance doc.

Captions (BFL; each needs a control probe on base FLUX.3, same crops and seeds, before it changes training data):
- [ ] The `paper_1` row takes the paper `from` ref_image_0 over the whole canvas: in FLUX.3's annotation format a
      full-canvas source box means the reference *is* the scene (the upsampler read it so). Try a new-content paper row
      and "on the same paper <paper_1> as <ref_image_0>" in the paragraph; rows without "the reference's".
- [ ] Ink by colour only (one hex from the darker half, implying contrast 120 vs 103 real): describe it physically
      (a dark and a pale hex, hairline upstrokes, heavier downstrokes, ink soaked into the fibres) in the first pass
      and the training captions; if it reaches the second pass's numbers, the second pass can go.
- [ ] "cursive" is FLUX.3's word for modern handwriting: name the script and the pen (e.g. Swedish Kurrent, a
      broad-edged quill, looped ascenders, long s), probe against the described caption.
- [ ] Lines cut by a crop's top edge are captioned as whole lines with their full text (2,438 of 143,941 rows, 21% of
      crops): count a line whole only if its x-height band is inside, else a fragment row.
- [ ] Colour names and hexes come from different measurements (6% of v3d prompts disagree); the microfilm look is
      named as a colour mode, not a capture; fragment rows use a negation; the reference is shrunk to 0.5 MP (FLUX.3
      takes 1 MP).
- [ ] Second-pass prompt: BFL's own Klein edit shape (scope first, every line quoted; C1q in tonight's probe), only
      artifacts the example shows, no stacked hedges.

Code (writing-python, diffusers):
- [ ] Captions are made by rewriting generated prose (restyle `_replace_once`, `StaleCaptionError`): render reference
      captions directly from their parts.
- [x] One row model for the eval-dir rows (EvalCrop, 24 Oct); `by_example_rows` leaves a stale caption out (24 Oct);
      the unclosed file, the dead constant, the Sampler overrides, the duplicated `load_rgb`/`jpeg` helpers and
      `sys.maxsize` are gone; exact-value edit-caption tests, the subset twin and pytest-timeout done (24 Oct).
- [ ] Still open: one Style (PageStyle and restyle.Style, with the runner's test); the studio graph built from dicts;
      the runner's `int(str(...))` round-trip (with the Ray work). (24 Oct: `plan_page`'s unused `settings` is gone,
      the optional arguments of `crop_hands`, `plan_page` and CaptionDropout are keyword-only, and `pick`'s preference
      for a crop with a margin note has its test.)
- [x] diffu sampled at FLUX.3's shift 6.93 but trained with flow_shift 6.0: `diffu-export` (24 Oct) writes a trained
      repo's scheduler at its checkpoint's `flow_matching.flow_shift`.
- [x] Guider: at `stop` < 1 the guider kept the previous crop's last step, so the next crop's text encoder step (which
      asks the guider at its current step) skipped the negative caption and the crop crashed: `draw_crop` sets the
      guider to step 0 first (24 Oct, with `--guidance-stop`). The NeMo LoRA loader refuses rsLoRA and per-module
      ranks or alphas. (CropLoopStep's copy of ModularPipeline.__call__'s state seeding went on 24 Oct.)
- [x] diffu's index names its Hub components (text encoder, tokenizer) with an empty subfolder: harmless (checked
      24 Oct). diffusers takes a component from a local repo only when its files are in that subfolder, and a diffu
      repo's root holds only the index; transformers reads subfolder "" as the Hub repo's root.

Inference speed (ranked; re-profile on torch 2.14.1 first, one change at a time, gated on CER and the hand metrics):
- [ ] 1. Fuse the LoRA (`fuse_lora` + `unload_lora_weights` after `set_adapters`; `unfuse_lora()` restores the base for
      a per-hand swap, or hotswap instead of fusing): est. 15-20% on the GB10 (the docs give no figure); prerequisite
      for compile, FP8 and QKV fusion. Built 24 Oct (`--fuse-loras`); measured tonight.
- [ ] 2. Built 24 Oct (`--compile-blocks`, dynamic shapes: on CPU a static compile recompiled for every new crop size,
      5-7 s a crop, the dynamic one 2.9 s after the first); measured tonight. Regional compile
      (`transformer.compile_repeated_blocks(fullgraph=True, dynamic=True)`; `dynamic` because
      captions pad in 80-token steps and the guidance passes differ in length; our guider's hook stays outside the
      compiled blocks): the docs say full compile's speedup with 8-10x shorter compile time; their benchmarks show
      1.26x on FLUX.1-dev's transformer (not the ~1.5x noted before). Check recompiles with
      `torch._dynamo.config.patch(error_on_recompile=True)`.
- [ ] 3. FP8 (torchao after fusing; FP8 dynamic + compile measured 1.96x on FLUX.1-dev, L20), with an INT8 A/B on
      sm_120. BFL's fp8r and nvfp4 Klein transformers load through `BFLQuantizationConfig` (runs without a LoRA; LoRA on
      quantized weights is not documented: fuse, then quantize, one hand per actor). NVFP4 needs FlashInfer, nvcc and
      CUDA >= 12.9 on sm_120 (sm_121 not stated).
- [ ] Sweeps: CFG `stop` 0.6-0.8 (`--guidance-stop`, 24 Oct; with three passes, 0.7 cuts 150 transformer passes to
      120); steps 50/40/30 x shift 6.93/16 x guidance 3.5/4.0 (we use 4.0 with 6.93, which matches neither diffusers'
      3.5/6.93 nor BFL's code 4.0/16); cuDNN attention (`set_attention_backend("_native_cudnn")`; attention is 8% of
      GPU time on the GB10).
- [ ] FLUX.3 Klein distilled (16 steps, no CFG, ~6x) for FLUX.3 research runs; base-trained LoRAs on it are unverified,
      and it has no guider, so no reference guidance.
- [ ] Exact savings from the diffusers docs (read 24 Oct; 0.42.0.dev0 has every API named): split the crop pipeline into
      stages with `pipe(state=...)`: the empty negative prompt encoded once per actor, each reference encoded once and
      passed as `image_cond_latents` (which skips the VAE encode), the text encoder and the read-back as stages of their
      own (frees ~8 GB on the drawing actor); `transformer.fuse_qkv_projections()` after the LoRA is fused (1.025x;
      it copies base weights, so an unfused q/k/v LoRA would silently drop); TF32 matmuls; `HF_ENABLE_PARALLEL_LOADING`
      and a compile cache for actor start-up.
- [ ] From reading FLUX.3's transformer (not in the docs, to verify): the text and reference blocks before the joint
      blocks always see timestep 0, so they compute the same thing every step; caching them would save ~5% exactly.
- [ ] Batching crops in one forward needs pre-encoded `prompt_embeds` and crops of one image size, reference size and
      80-token caption window (no attention mask, shared position ids); the empty-prompt passes batch across any crops
      of one size. `num_images_per_prompt` batches variants of one plan exactly. The GB10 is already 99.5% busy at batch
      1 (23 Oct: matmul 41%, elementwise 37%, copy 10%, attention 8%).
- Not for us: diffusers' caching methods (they change outputs; FLUX.3 has no cache support in 0.42), offloading,
  context/tensor parallelism, the other quantizers. Hub kernels: a `*_hub` attention backend from a publisher the Hub
  marks trusted loads even with DIFFUSERS_TRUST_REMOTE_KERNELS off (attention_dispatch.py): never select one, and keep
  `kernels` uninstalled.

Data checks (23 Oct evening, the user's question: any leakage or anything weird?):
- [x] Per-hand LoRAs: training and held-out pages disjoint (1715: 36 vs 4 pages; Pahlman: 37 vs 3); held-out lines
      also in training: 3 of 224 set phrases (signature, "Öfwerståthålls wägnar") and 0 of 209. The realism example
      page (1244_cac (29)) is held out. Tonight's reference tests run untrained FLUX.3, and no probe reference shares
      a line with its target. diffu E1 trains on real crops only (layout_crops_v2/train; its 24 held-out pages share
      no page and no line > 25 chars with training). v2b: train/val/val_eval page-disjoint; 20 of 6,120 val lines
      repeat in training, all set phrases ("Wy Gustaf Adolph. Giöre witterligitt").
- [x] v3d's pairs: 85 of 10,766 targets had a facing-page reference holding some of their own lines (consecutive
      scans overlap: Strödda kamerala A0073405_00018/_00019 show the same physical lines; the rest account formulas).
      Fixed in `pick_same_hand` (3d4b0c6, red-checked).
- [ ] Rebuild data_flux3/style_pairs_v3d with the fix before v3d trains (85 references change).
- [ ] Held-out volumes for v3d: the split is by page, so 23 of the 24 val_eval volumes (50 of 52 val) are also in
      training. Fine for new text in a known hand; it cannot show a trained model copying an unseen writer from one
      image. Hold out whole volumes (different eras and scans) for v3d's eval.
- [x] The read-back model (Riksarkivet/trocr-base-handwritten-hist-swe-2) was trained on lines of most of our
      collections (Pahlman, Bergskollegium, Bergmästaren i Nora, Finnish court records, Svea and Göta hovrätt,
      Sollentuna, Krigshovrätten, Trolldomskommissionen, the Stockholm city archives). Tested (scratchpad
      leak/htr_memory.py, CPU bf16): the middle 40-70% of 114 real lines painted out in the paper's colour; it wrote
      back 0 of 124 painted-out words (>= 5 letters), on collections it trained on (Pahlman, Bergskollegium) and not
      (Väse, Arjeplog); the reads shrink to 0.71-0.73 of their length. It reads the pixels, so the slip filter holds.
      Its CER is lower on its own collections: mean 0.012 / 0.041 against 0.136 / 0.189 (its card's held-out 0.025 /
      0.048), so CER compares only within a collection.
- [ ] Writing no label asks for (scratchpad hand/unlabelled.py: ink outside every box the caption names, grown by
      half a line): Pahlman's LoRA writes the caption's "1825–1826" as a running header in 2 of 8 windows (marks in 4
      of 8; untrained FLUX.3 none; real windows have untranscribed page numbers there, which is where it learned the
      place). The 1715 LoRA's outside ink is the clerk's flourishes, as on the real pages. Options: keep digits of the
      date out of the caption (the era in words), caption untranscribed marks as their own rows, and check every drawn
      window for outside ink (inpaint or drop) before its labels are written.

## Inference at scale

- [x] Ray Data entrypoint (diffu-runner): layout / refill / plan as a CPU `flat_map` (`PagePlanBlocks`), one crop
      per block, the crop pipeline as a stateful GPU actor (`map_batches`, concurrency = actors), TrOCR read-back
      co-located, Parquet sink.
- [ ] After profiling on the cluster: text encoding as its own stage (FLUX.3's text encoder takes one prompt per
      call), a Lance sink if Parquet's per-file writes get in the way.
- [ ] Runs on our Ray cluster's Blackwell GPUs (96 GB each, sm_120). Re-profile there before optimizing; measure
      FP8 / NVFP4, fewer steps, a compiled transformer, several actors per card.

The order of the speed work (24 Oct): measure, then restructure for Ray Data, then speed up one change at a time, all
on the Spark; the cluster only once the tests are done. Every change is gated on the hand measures and writer-ID of the
same held-out windows (CER as information only).
- [x] 1. Baseline profile (24 Oct; runs/profile_baseline, scratchpad profile/): six held-out 1715 windows with their
      LoRA (FLUX.3 Klein Base, bf16, the LoRA unfused, 50 steps, guidance 4, torch 2.14.1 on the GB10). Median 142 s a
      1024x1024 crop: denoise 139 s (98%; 100 transformer passes at ~1.4 s), read-back 1.6 s, decode 0.8 s, text
      encoder 0.35 s, the rest ~0; peak 25.3 GB. The trace of two crops (6.1M events, streamed with
      profile/stream_trace.py, never loaded whole) has the GPU 99.6% busy inside denoise: GPU-bound, no host gaps.
      Kernel time: GEMM 43.5%, elementwise 35.8% (~350 us each: bandwidth-bound at the GB10's ~273 GB/s), cat/copy
      9.0% (rotary's neg/cat/stack, fp32 casts), attention 8.6% (flash, 3 ms a call), norms 3.0%. 92,400 of the
      141,547 aten::mm calls are the unfused rank-32 LoRA (231 adapted linears a pass); with each one's scale and add
      over the full output, about 15% of every pass is the LoRA. The order holds: fuse the LoRA, then a regional
      compile (it fuses the elementwise chains, the GB10's other half), then FP8 for the GEMMs.
- [ ] 2. From that breakdown, the Ray Data split below: which steps get their own stage and resources, and the batch
      size per bucket; built on the Spark with a local Ray and measured with Ray Data's own stats before and after.
- [ ] 3. Memory: a snapshot of one crop (pytorch-skills' analyze-memory-snapshot), to size the actors per 96 GB card
      (FLUX.3 actors, read-back actors).
- [ ] 4. Then the diffusers speedups, one at a time in the ranked order of the 23 Oct audit (fuse the LoRA, regional
      compile, attention backend, FP8, guidance on part of the steps, fewer steps), plus diffusers' caching of
      transformer blocks between steps; graph breaks found with pytorch-skills' debug-graph-breaks. Built 24 Oct:
      `--fuse-loras` and `--compile-blocks` (FLUX.3's two repeated block types compile with fullgraph on CPU, no
      graph break; the tiny model draws within 2/255 of eager). Queued after diffu E1 (scratchpad
      profile/run_fuse_ab.sh): the profile fused and fused + compiled, and the 8 held-out windows evaluated as an
      adapter, fused and fused + compiled, against the 142 s baseline and its hand measures.
- [ ] NeMo AutoModel moves the transformer to bf16 with `.to()`, so FLUX.3's `norm_q`/`norm_k` (diffusers'
      `_keep_in_fp32_modules`) train in bf16 in every LoRA and diffu run, while load_pipeline loads them in float32
      (the log warns: "should be kept in float32"). Measure whether it matters: one LoRA drawn with those norms in
      bf16 and in float32 on the same windows; if it does, load them in float32 for training (NeMo's pipeline
      config) or match inference to training.
- [ ] diffu's Qwen3.5 text encoder runs its linear-attention layers in transformers' reference PyTorch code
      (`causal-conv1d` and `flash-linear-attention` are not installed in either environment): caching 2,000 crops
      took about 40 minutes. Installing them into NeMo's environment needs the user's approval.

Where our code does the compute engine's job, or blocks Ray Data from optimizing (review of 24 Oct; the boundary is
right in diffu_runner: Ray does fan-out, actors, retries and the sink; these are the exceptions). On the Spark with a
local Ray, profiled before and after; no cluster runs until the tests are done:
- [ ] A crop is drawn one per call (`draw_crop`, `batch_size=1`): give the compute layer a batch of same-size crops
      (one bucket, one forward) and let Ray Data pick the batch size; group crops by bucket upstream.
- [ ] One actor does everything in turn (reference image read, text encoding, denoising, VAE decode, HTR read-back,
      label XML, JPEG): split into stages with their own resources (CPU read and encode, GPU denoise, a small GPU
      read-back pool, CPU labels and sink) so Ray Data pipelines them. Modular Diffusers blocks already split there.
- [ ] The driver reads every job into a list (`read_pages` -> `from_items`) and names and seeds crops by the page's
      position in the run file (two run files reuse the same seeds): read jobs with `ray.data.read_json`, and give
      every page a content-based id (template, text and style hashed) that names and seeds its crops.
- [ ] Our own parallelism: `style_pairs` runs a `ProcessPoolExecutor`: build the pairs as a Ray Data map over pages.
- [x] `page_blocks` looped over a page's crops inside one pipeline call: removed (24 Oct, the user's call); generate
      draws crop jobs one by one, so one failed crop costs no other.
- [ ] `DrawActor` catches every exception as a failed crop, CUDA faults included: let CUDA errors (OOM, illegal
      access) raise so Ray restarts the actor; keep per-crop rows for content errors only.

Lessons from NVIDIA NeMo Curator (github.com/NVIDIA-NeMo/Curator, Apache-2.0, read 24 Oct: stages/base.py,
resources.py, tasks/tasks.py, pipeline/pipeline.py, backends/base.py, backends/ray_data/{adapter,executor}.py and its
AGENTS.md tuning guide, failed_task_markers.py, utils/resumability_actor.py, utils/atomic_io.py, the CLIP embedding
and image writer stages). Its core idea is ours to copy: the compute code is a stage that declares what it needs, and
an executor (Ray Data, a Ray actor pool, Xenna, a Slurm array) decides how it runs:
- [ ] A small stage contract for our generation steps: `setup()` per worker (load the model once), `setup_on_node()`
      (fetch weights once per node: Leonardo's offline nodes), `process_batch(list[In]) -> list[Out]`, declared
      `resources` (CPUs, GPUs or GPU memory in GB, turned into a GPU fraction: the 96 GB cards hold one FLUX.3 actor
      at ~50 GB but many small read-back actors) and `batch_size`; one generic Ray Data adapter maps a stateful stage
      to an `ActorPoolStrategy` and a stateless one to tasks, and repartitions after fan-out stages (Curator's rule:
      one Task per block, so the Task's size sets the parallelism). Pydantic models and a Protocol, not Curator's ABC
      and metaclass registry (writing-python).
- [ ] Deterministic task ids owned by the framework: a child's id is its parent's plus its own segment, and a source
      task's segment is content-based (Curator hashes a file group's sorted paths) so adding or removing pages does
      not shift the others. Our crop names and noise seeds come from these ids (with the content-based page id above).
- [ ] Failures as sentinels: a stage returns a failed marker instead of raising, the adapter strips markers before the
      next stage and records the attempt in a failure manifest; filtered items are a None marker. Our DrawnRow error
      column does part of this but carries failures into the sink.
- [ ] Resumability for million-page runs: per-source completion counters (Curator: one LMDB file per writer, the union
      read at start, so a rerun skips every source any earlier writer finished); for us at least: deterministic output
      names per page plus a completed-pages manifest, so a restarted run skips finished pages instead of appending.
- [ ] Atomic writes (temp file, fsync, rename) for every manifest and dataset file, and sinks that name their files
      from the task id (a rerun overwrites the same file, never adds a second): fixes the audit's half-written builds
      and the runner's Parquet appended into an earlier run.
- [ ] Composite stages: a user-facing "draw pages" stage that decomposes into plan (CPU fan-out), draw (GPU), labels
      and write (CPU) at planning time, with per-sub-stage overrides (Curator's `with_()`).
- [ ] Declared inputs and outputs per stage, checked when the pipeline is built, not mid-run.
- [ ] Tuning by diagnosis, not by knob: Curator's Ray Data guide classifies starvation, actor saturation, resource
      admission and downstream backpressure from Ray Data's scheduler events before changing a worker count; one
      lever per run, compared on the same input.
- [ ] Later, a spike: our draw stage as a Curator ProcessingStage on its RayDataExecutor, on the Spark, to judge
      using Curator itself (its executors, resumability, failure manifests and diagnostics for free) against its
      weight (loguru, lmdb, dataclass stages, Ray >= 2.57 coupling). Its image stages (CLIP embeddings, aesthetic and
      NSFW filters, dedup) could also curate our real crops and filter generated windows.

## Design and quality audit of 24 Oct (nine parallel audits against writing-python; 86 findings verified, 67 worth fixing)

Done on 24 Oct, each with a test that was red with its fix removed:
- [x] 1. Labels put each line on its true baseline (edge lines were up to 0.21 pitch off); a plan's reading order is
      checked. Old records still get 0.7 of the slot: delete that branch when the datasets are rebuilt.
- [x] 3. The diffu cache stops when an encoder fails to load and never marks an empty cache finished.
- [x] 4. ty beside pyrefly; both check torch, numpy and PIL; the hook runs both.
- [x] 5. Every run input checked before the expensive work (LoRA dirs and scales, the model index, guidance without a
      reference, empty plans, a used --out, drawers, FLUX3_MODEL with no default path and no Hub fallback).
- [x] 6. The dataset builders never leave a half-written dataset (build_crops, style_pairs).
- [x] 7. Outputs record what produced them (evaluate and generate summaries, a page id, style_pairs' build.json).
- [x] 8. Unmeasurable hand features are None, never black ink or a crash.
- [x] 13. diffu-generate draws crop by crop; page_blocks, CropLoopStep and LoadReferenceStep are gone.
- [x] Smaller: the studio refuses page and drawer names that are paths; machine JSON printed on one line; the HTR
      model in Settings; isinstance for TrOCR's tables; escaped SFM characters; the Python floor explained.

Still to do, in this order (the refactors after the fixes, the user's call):
- [x] 2. (24 Oct) ALTO carries what PAGE carries: a `Tags` element (an OtherTag for unclear, a StructureTag per region kind) and
      TAGREFS on lines and blocks; one test over both exporters; validate ALTO against the 4.4 XSD; check PAGE's
      unclear tag against a real Transkribus export (it writes `unclear {offset:..; length:..;}`).
- [x] 15. (24 Oct) The interleaved sampler resumed from `state_dict` yields exactly the rest of an epoch;
      pytest-timeout and pytest-xdist; the suite pruned with per-test coverage (testing-python's recipe): of 136 tests
      none runs no production line outside the runner, 64 lie within another's lines, 4 asserted nothing a kept test
      does not and are gone (union of executed lines 1,893 before and after; each red-checked on its kept test).
- [ ] The runner's draw test calls `run()` and reads the Parquet and run_meta.json back; its RunConfig `[no-drawer]`
      case runs no production line (Pydantic's min_length refuses it): both wait for the Ray work.
- [ ] The suite's time is two tests (the studio's draw test 6.5 min, draw_pages 2.8 min, of ~12 in one process): both
      draw whole planned pages (bucket-sized crops) and read every line back with the real TrOCR on CPU. Profile one
      (cProfile or py-spy) before choosing a fix: smaller test pages, or a tiny read-back model for the tests.
- [x] 11. (24 Oct) One eval-run module: EvalCrop the one row model (reference and reference_lines together), read_jsonl and
      write_jsonl, generated_path, the hand-fidelity statistics out of the CLI module.
- [x] 12. Draw options and shared constants declared once (24 Oct): Annotated option aliases with defaults from one
      `DrawSettings()`, max_line_cer in DrawSettings (`--max-line-cer`), `--reference-guidance` for generate,
      LabelFormat a StrEnum, MAX_SEQUENCE_LENGTH and COMPONENT_DTYPES from diffu, one JPEG quality and helper. Not in
      the runner (no Ray work yet): diffu-run keeps its own option copies and `_jpeg` until then.
- [x] 9. (24 Oct) One reader: build_crops cuts training crops from `regions_of(load_xml_layout(page))` at the scan's
      size (parse_page and the builder-versus-planner test are gone); `regions_of` lives beside Region in crops; an
      element without an id is keyed by its place, so it stays a region of its own (ALTO TextBlocks, PAGE regions,
      tables, cells); an ALTO page in other units than pixels is refused; a page that cannot be read names its file.
      Golden diff over the 1,952 trained pages: generation byte-identical; training identical in every crop's rect,
      size, caption, layout rows and line records, the lines now carrying their block.
- [x] 10. (24 Oct) One caption grammar: `phrases_for` in captions.prompt spells every style phrase (`StylePhrases`)
      and both caption_for and restyle use it. That fixed a bug: restyle's own copy knew neither a printed page's
      opening ("a page printed and filled in by hand") nor printed and ruled rows, so a form or table drawn from a
      reference (v3) failed as a "stale" caption; those rows name no style and are now left as written. Only a real
      grammar mismatch becomes StaleCaptionError, and stored layout rows are compared as models, not re-serialized.
      Golden diff over the 11,827 trained crops: the caption, restyled, edit, named-edit and described-edit captions
      byte-identical. Left: PageStyle and restyle.Style as one model (the runner's test imports PageStyle, so with
      the Ray work).
- [x] 14, first two parts (24 Oct): a line's kind is one `LineKind` StrEnum (running text, a table cell's line or a
      role; the stored LineRecord keeps its role as written and reads its kind through one property), and one
      `TextLine` (text, page box, baseline, kind, block, production, cell) runs from the reader through the crops:
      LineBox, the crop cutter's Line and their rectangle-polygon round trip are gone. Golden diff byte-identical on
      both paths, the boxes now the outlines' exact extremes.
- [x] 14, the rest (24 Oct): `LayoutBox` (0-1000, y first, a 4-tuple) for every box captions, records, plans and
      eval sets store, so a box of the wrong length fails when it is read, not mid-draw; `Paper` and `Ink` StrEnums
      for the scan look (every stored value was a band name: 6 papers, 6 inks), so `--paper` and `--ink` offer the
      names the LoRAs were trained on; `band_name` generic. Golden diff byte-identical on both paths. The golden diff
      (scratchpad golden/golden.py) re-cuts the 1,952 trained pages on the training and generation paths and writes
      every crop's caption in each form; the two paths differ on 35 pages only by the page size (stale PAGE sizes;
      Byggningskollegium's TIFFs), and no trained page has a SeparatorRegion or a TableRegion.
- [x] Smaller (24 Oct): one `crop_record` for build_crops and the tests; refill takes the text and starts a paragraph
      after a blank line; probe_set refuses references whose words never differ; diffu-train and diffu-studio declare
      their direct imports; shared crop and record fixtures in conftest; exact paragraphs in the edit-caption tests;
      stale docstrings and references to missing files.
- [ ] Still open: diffu-runner's direct imports (numpy, pillow) wait for the Ray work; generation checks
      `--reference-caption` against the LoRA's build.json once training copies build.json into its checkpoints.

## Quality control of generated windows (plan of 24 Oct: embeddings, a tabular model, a VLM judge)

The plan: embed every generated window (TIPSv2 or Nemotron 3.5 Embed), reduce the embedding, give it with our own
measures to NVIDIA's KumoTabular as a QC model, and let Nemotron 3.5 Super VL judge as a VLM. Production QC runs as
Ray Data stages on the Blackwell cluster (96 GB cards); the Spark only tests each piece, one GPU job at a time through
jobs/spark_guard.sh. Every source below was read in full on 24 Oct. Findings that shape the plan:
- KumoTabular (github.com/NVIDIA/structured-data-models; code Apache-2.0, weights `nvidia/Kumo-Tabular` v1.0.1
  OpenMDW-1.1, 28M-215M, a pip package, `.pt` loaded `weights_only=True`, no remote code; CUDA or CPU) learns in
  context: labelled rows in, predictions for new rows in one forward pass, classification (10 classes a pass) or
  regression (999 quantiles). Numerical features only, so an embedding is N numerical columns. The default recipe
  keeps at most 500 columns and does not reduce them (`sp.PCA` exists, opt-in, and replaces the whole numerical
  block). It was pretrained on tables of up to 100 columns and contexts of 1k-60k rows, and degrades when query rows
  come from another distribution than the context. Its context needs examples of every class.
- KumoAnomaly, sdm's label-free model (NV-Tesseract-AD, arXiv 2610.12754; weights `nvidia/Kumo-Anomaly` Apache-2.0,
  32 MB, `.pth` loaded `weights_only=True`, no remote code), scores every row zero-shot: after ImputeMean, PCA to 40
  columns and MinMaxScale it hides alternating blocks of 25 rows in windows of 100 and reconstructs them by
  conditional diffusion from the visible blocks, with a time embedding per row; a row's score is its mean absolute
  reconstruction error, and two thresholds (SCS, MACS) label anomalies. It is built for ordered rows: its card also
  evaluates it on SECOM, wafers in production order with 1,558 process features. For us that order is the production
  stream: generated windows in the order a run draws them.
- TIPSv2 (Google DeepMind, CVPR 2026, arXiv 2604.12012, Apache-2.0; B/L/SO400m/g with 768-1536-d, two global tokens
  and 32x32 patch features at 448 px) is native in Transformers since 5.14 (we have 5.20.1), though its Hub repos
  still ship custom code: load it natively with remote code off and check the outputs. Nemotron-3.5-Embed-1B/8B
  (OpenMDW-1.1, native `Mistral3Model`, no custom code; 2048-/4096-d, truncatable to the first 1024 or 512 dims and
  renormalised) embeds page images for retrieval. Neither was trained or evaluated on handwriting: TIPSv2 on natural
  images, Nemotron Embed on printed documents (its image retrieval not evaluated in Swedish). Avoid the colembed and
  omni-embed models (non-commercial) and llama-nemotron-embed-vl (custom code).
- Nemotron 3.5 Super VL 120B-A12B NVFP4 (OpenMDW-1.1, which leaves outputs unrestricted; 75 GB of weights; listed
  for one DGX Spark or one 96 GB RTX PRO 6000; vLLM, SGLang, TRT-LLM): one tile per image up to about 3.4 MP,
  reasoning off/medium/on, OCRBench v2 EN 68.3, no handwriting numbers, card languages en/es/fr/de/it/ja (Swedish
  only in its pretraining web data). **Every serving recipe needs trust_remote_code** (custom NemotronH Omni modules),
  which our rule keeps off: the user decides whether to allow it once we have read the code and pinned a revision.
  The smaller VL models (Nano-Omni-30B-A3B NVFP4 21 GB, Nano-12B-v2-VL, Bolt-74B-A9B) need it too. On the Spark a
  third party measured about 100 GiB resident, ~19 tok/s (38 with the LiLiCorr drafter), an 11-minute start: a test
  takes the whole box; in production it is one replica per 96 GB card.
- vision-intern (github.com/merveenoyan/vision-intern; README says MIT but the repo has no licence file: ideas only):
  the judge sees a numbered overlay of the boxes, never coordinates, and answers JSON `{id, verdict, score, reason}`;
  an LLM writes one positive definition per class, approved once by a human; judges from different model families
  vote (`min_agree`, mean score) beside deterministic geometry checks; it calls its own scores uncalibrated and
  scores the result against human labels. Its flaw to avoid: a dead or unparsable judge silently votes "incorrect".
- The retriever blog (huggingface.co/blog/nvidia/nemotron-retriever-world-class-agentic-retrieval, 21 Oct): agentic
  retrieval over Nemotron 3.5 Embed, Rerank (custom code) and Parse (disclaims handwriting). For us the embedder is a
  tool for nearest-real-window search (reference picking, near-copies of training windows) more than for QC.

- [ ] Alignment of the training crops (found 24 Oct): a crop cut from the wrong image has its line boxes on blank
      paper. Byggningskollegium's JPEG/TIFF pairs did that to 15 trained pages (119 crops of layout_crops_v2/v2b, some
      with every box on blank paper); build_crops now cuts the image the PAGE file names. An ink-inside-the-boxes
      measure (scratchpad golden/alignment.py) finds no other: its lowest crops are faint brown hands, aligned. To
      check a whole set, read its real crops' lines back with the HTR model (a misaligned crop reads as nothing): CER
      as a QC signal on our own ground truth, not as a judge of generated windows.
- [ ] Probe first, on the Spark: embed the real held-out windows and the LoRA and untrained FLUX.3 drawings of the four
      per-hand volumes with TIPSv2-L/SO400m and Nemotron-3.5-Embed-1B (windows tiled or padded to the 448 px square,
      not squashed). Per embedding: does kNN retrieve the right writer, and does a linear probe separate real from
      drawn, as well as our writer-ID CNN does? An embedding that does not carry the hand cannot judge it.
- [ ] A label-free realism baseline: a window's kNN distance to the real windows of its hand and layout type.
- [ ] KumoAnomaly on the stream: per run and generator version, the windows in drawing order, their embedding
      components and our measures as columns. Inject known failures into a stream of good windows (blank and black
      drawings, windows of the wrong LoRA, untrained FLUX.3 windows, a caption batch gone wrong) and check that it
      flags them and nothing else. In production it runs on each run's stream and stops or reroutes a run that drifts.
      Also try it as a per-window realism score: blocks of real and drawn windows alternating, so each drawn block is
      reconstructed from real ones; adopted only if it beats the kNN baseline.
- [ ] KumoTabular QC: rows are windows; at most ~100 numerical columns (the chosen embedding's PCA, fit on the
      context's real windows, beside our hand measures, writer-ID probabilities, rule geometry checks, CER as
      information only); target real vs drawn (a classifier two-sample test whose P(real) is the realism score) and,
      once we have them, judged or archivist verdicts. One context per generator version and hand. It is adopted only
      if it beats the kNN baseline on held-out labels.
- [ ] The VLM judge, once remote code is decided: numbered PAGE line, cell and rule overlays; pairwise questions
      (which of two windows is the real one; is the drawn window in the reference's hand); verdicts as JSON with
      reasons; a failed call is no verdict, never a vote. It labels the context set and audits samples (a few
      thousand windows a day per replica), not every window. Our deterministic checks stay the gates.
- [ ] A calibration set: the archivist blind test's windows, labelled once, to calibrate the judge and KumoTabular.
- [ ] Production: QC as Ray Data stages after drawing: embedding actors (fractional GPUs), KumoTabular fit once per
      context and `predict` in `map_batches`, judge replicas through vLLM on whole 96 GB cards for the sampled windows.

Templates outside our ground truth: DocLayout-YOLO's Mesh-candidate BestFit (github.com/opendatalab/DocLayout-YOLO,
`mesh-candidate_bestfit/`, read in full; arXiv 2410.12628; the code is AGPL-3.0, so reimplement from the paper, never
copy). It builds pages as 2-D bin packing: from a pool of category-wise element crops it samples 500 candidates
(99 % large), places one at random, then repeatedly takes the grid of every placed element's edges, keeps the empty
rectangles and puts the candidate that fills one best at its centre; it stops when nothing fits or after five small
elements, shrinks each element to 80-95 % for white space, renders with PyMuPDF at random page sizes and writes YOLO
labels. Pretraining on its 300k synthetic pages (DocSynth-300K) improved layout detection.
- [ ] BestFit as a template generator: region boxes (sizes from our ground truth's regions, tables and forms) packed
      into new pages, each region refilled with lines, ruled lines and table cells, then planned into windows as any
      template. Windows then show new contexts (a note beside a table, a heading over a form).
- [ ] A cheap baseline beside it: real ground-truth regions pasted into BestFit layouts with their PAGE lines (real
      hands, exact labels, no drawing), with the seams checked, to compare against drawn windows for VLM training.
