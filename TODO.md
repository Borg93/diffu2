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

## Repo structure

Target (work happens on the `next` worktree, `~/Desktop/repos/diffu2-next`; main is frozen until tonight's
queue, v2 → steering → generator smoke → v3, has finished, ~09:00 2026-10-22, because the queue imports
page-synth through editable installs in the NeMo venv):

```
pyproject.toml        virtual uv workspace root: members, dev tools, ruff / pytest config
packages/
  page-synth/         pages: layout (from diffu-page), labels, plan, captions, Modular blocks and pipelines,
                      generator, evaluation, training-data builders, NeMo dataloader configs
  diffu/              the model (planned, see "diffu"): configs for Flux3Transformer2DModel, VAE blocks
configs/flux3/        NeMo configs (FLUX.3 reference LoRAs);  configs/diffu/ later
jobs/flux3/           GPU job scripts;  jobs/diffu/ later
scripts/spark_guard.sh
docs/                 README, PIPELINE.md (the page pipeline), DIFFU.md (the model plan)
```

- [ ] Move `diffu_page.layout` and `diffu_page.labels` into page-synth (`page_synth.layout`, `page_synth.labels`)
      with their tests; drop the private re-exports in `layout/__init__.py`.
- [ ] Remove the line-pasting era: the rest of diffu-page (compose, render, augment, upscale, htr, paper, profiles,
      validate, overlay, dataset, generation, cli), `packages/diffu-runner` (its Ray Data pattern is rebuilt in
      page-synth), `packages/diffu-studio` (the app: not wanted), the old diffu model `src/diffu` with its
      `tests/`, `Makefile`, `slurm/`, diffu-era `scripts/` and `docs/`. Tag `diffu-mmdit-v1` first so the old
      model stays reachable.
- [ ] Root `pyproject.toml` becomes a virtual workspace root. diffusers pinned to the git commit that has Flux3
      (`e04f0c5`, the NeMo venv's), so the repo env and the NeMo env run the same blocks.
- [ ] Record the NeMo environment: page-synth (and later diffu) installed editable into
      `~/Desktop/repos/Automodel/.venv-managed` (`uv pip install --no-deps -e`); a `jobs/setup_nemo_env.sh`
      recreates it.
- [ ] Delete `experiments/` once the running v2 job's evals have used it (~16:45 2026-10-21).
- [ ] Tests for page-synth: plan, restyle, caption identity, pipeline wiring (CPU, no weights).
- [ ] Docs: README says what the repo is (page pipeline + diffu model); `docs/PIPELINE.md` from the pipeline page.
- [ ] Merge `next` into main after the queue: fast-forward, push, delete the branch and the worktree; rerun the
      queue's commands from the new paths once to check them.

## Modular Diffusers (checked against huggingface.co/blog/modular-diffusers and the diffusers source, 2026-10-21)

- [ ] Text encoder as its own pipeline: pop `text_encoder` from the Flux3 blocks, encode prompts in batches,
      feed `prompt_embeds` to the denoise pipeline. For Ray Data a separate stage that streams ahead of the GPU
      denoiser; for training the same embeddings NeMo caches.
- [ ] One `ComponentsManager` shared by the text-encoder, text2image (v2) and image_conditioned (v3) pipelines:
      no double loads; offloading matters on the inference cluster's discrete-VRAM cards.
- [ ] TrOCR through `load_components` from the read-back block's `ComponentSpec` (then fix its meta position
      table), not the side loader `load_trocr`.
- [ ] Self-contained blocks in page-synth (no imports of local scripts), publishable with
      `save_pretrained(push_to_hub=True)` / `diffusers-cli custom_blocks`.
- [ ] The base model is a component, not a constant: FLUX.3 Klein for the reference, diffu for ours, swapped with
      `update_components`; the same blocks drive both.
- Done right already: `get_workflow` + `sub_blocks.pop/insert`, standard block shape, automatic data flow (only
  `crop_plan` is a pipeline input), `load_components(dtype={...})`, `update_components`.

## diffu: FLUX.3 architecture, our weights (plan only, nothing implemented)

- [ ] **Model**: `Flux3Transformer2DModel.from_config(...)`, random init. Sizes measured on the meta device:
      diffu-S 0.66B (hidden 1536, 3 double + 10 single blocks), diffu-M 1.66B (2048, 4 + 16); Klein is 5.54B
      (3072, 5 + 28). Start with S.
- [ ] **Text encoder**: Qwen3.5-4B (Apache-2.0, Feb 2026, hidden 2560 like FLUX.3's Qwen3-VL-4B, so FLUX.3's 8
      stacked layers = 20,480 carry over); Qwen3.5-2B as the cheaper alternative. Open: which layers to stack
      (Qwen3.5 mixes linear- and full-attention layers). Exact spelling (å ä ö, historical forms) through a BPE
      encoder is a risk; fallback: rendered glyphs as a reference image (old diffu's glyph strip, in 2D).
- [ ] **VAE**: Qwen-Image's 16-channel VAE (Apache-2.0; passed our å ä ö reconstruction gate). Needs our own
      encode / decode blocks: the Flux3 VAE blocks assume FLUX's 32-channel autoencoder and its batch-norm stats.
      `in_channels` = 16 × 2 × 2 = 64.
- [ ] **References**: up to N reference images in the sequence with the 4-axis RoPE (FLUX.3's in-context
      conditioning) replace the old pooled DINO style vector, which gave generic style.
- [ ] **Training**: NeMo AutoModel's flux3 recipe in pretrain mode. Check it accepts a random-init transformer
      and a different VAE / text encoder; otherwise add a diffu processor. Progressive resolution 256² → 512² →
      1024², aspect buckets, our captions. Recipe details (timestep shift, sampling, schedule, data mix) from
      BFL's FLUX.3 technical report: find and read it.
- [ ] **Data**: GT-Lejonet, 103,285 pages (~3M lines). Never FLUX.3 outputs.
- [ ] **Compute**: estimated ~8×10¹⁹ FLOPs for diffu-S (10M samples at 256², 5M at 512², 2M at 1024²; GB10
      measured at ~25 TFLOPS effective on the v2 LoRA). GB10 alone ~40 days: too slow, use it for smoke runs and
      short ablations. Leonardo, 8 nodes (32 A100s): roughly 6–30 h. First a short GB10 run at 256² to see loss
      and read-back CER fall, then book compute.
- [ ] **Eval**: the same held-out crops and metrics as the FLUX.3 LoRAs (per-line CER, layout, style score), so
      diffu and the reference are compared head to head.

## Labels (ALTO / PAGE are only output formats; the label data is what must be right)

- [x] Reading order across columns: labels follow the source blocks in page order, top to bottom in each
      (`CropPlan.reading_order`).
- [x] Line polygons: checked against the GT, not a defect (GT bands are 1.6x the line pitch; our slots 1.0x;
      slot IoU 0.72 vs traced ink 0.37). To match the GT convention, widen the slot.
- [ ] Narrow columns get no crops (590 px < the 640 px minimum when a margin note narrows a multi_column page).
      Decide: crop narrow columns side by side, or keep such templates out.

## Layout (from diffu-page): verify before relying on it

- [ ] Tune the template sampler to the GT before using synthetic templates (2026-10-21, 300 vs 300 pages:
      running-text lines per page 30 real vs 9 sampled; line pitch 0.029 vs 0.090 of page height; line width 0.82
      vs 1.00 of the block). Until tuned: real templates only.
- [ ] The `landskap` template gives no crops: the windowing rejects its short label lines.

## FLUX.3 reference runs

- [ ] Tonight's results: v2 eval, steering test (style by name), first generated crops (`runs/flux3_pages/smoke`).
- [ ] Decide on v3 (style by example) from the steering results; its results land tomorrow morning.
- [ ] Read tonight's profile (`runs/flux3_pages/smoke/profile/trace.json`; pytorch-skills:profile-model): SDPA
      backend on sm_121, GPU idle vs busy, top kernels, text-encoder vs denoise vs VAE vs TrOCR share.
- [ ] Full pages: generate at 2–4 MP, or stitch windows that share a reference.
- [ ] Style score: a classifier on real crops that says whether a crop matches the requested archive or reference.

## Inference at scale

- [ ] Ray Data entrypoint in page-synth: layout / refill / plan as CPU map steps, text encoding as its own
      stage, the generator as a stateful GPU actor (`map_batches`, concurrency = GPUs), TrOCR read-back
      co-located, Lance sink.
- [ ] Runs on our Ray cluster's Blackwell GPUs (96 GB each, sm_120). Re-profile there before optimizing; measure
      FP8 / NVFP4, fewer steps, a compiled transformer, several actors per card.
