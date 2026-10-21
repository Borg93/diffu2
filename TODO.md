# TODO

What is known to be wrong or unfinished. Remove an item when it is done; add the commit that did it.

## Repo structure (do after the v3 run finishes; queued jobs run today's script paths)

- [ ] Decide what the repo is now: the FLUX page-synthesis pipeline as the core, diffu as the line-level model inside it, or diffu moved aside.
- [ ] Move the generator out of `experiments/flux3_klein/` into a package (`packages/page-synth/`): caption grammar, crop planner (`page_plan.py`), Modular Diffusers blocks (`page_blocks.py`), generator CLI (`generate_pages.py`), with tests. It depends on `diffu-page`, never the reverse.
- [ ] One import direction: experiments import packages only; no script imports another script (`page_blocks` → `eval_layout`, `page_plan` → `build_layout_crops` today).
- [ ] Move the PAGE XML parser (`diffu.data.parse`) into `diffu-page`, so the FLUX side doesn't import the old diffu model package.
- [ ] Record the FLUX environment: `diffu` and `diffu-page` were hand-installed into `~/Desktop/repos/Automodel/.venv-managed` with `uv pip install --no-deps -e`. Nothing recreates that today.
- [ ] Split `experiments/flux3_klein/` into data / train / eval; keep only run scripts and configs at the top.
- [ ] `scripts/`: about 25 diffu-era one-offs next to `spark_guard.sh`, which every GPU job needs. Give the guard a clear home, and keep or delete the rest.
- [ ] `packages/diffu-runner`, `packages/diffu-studio`: diffu-era, unused by the FLUX side. Keep with a note, or remove.
- [ ] Docs: `docs/` describes diffu only. Add `docs/FLUX.md` (the pipeline page, https://claude.ai/artifact/X47DU54cesTiH3qw854U7g) and say in the README which part is which.
- [ ] Re-run the generator test from the new package and compare with the run before the move.

## Labels (ALTO / PAGE are only output formats; the label data is what must be right)

- [x] Reading order across columns: labels now follow the source blocks in page order, top to bottom in each (`CropPlan.reading_order`); the prompt keeps the training numbering. Checked on a refilled Bergskollegium spread.
- [x] Line polygons: checked against the GT, not a defect. GT line polygons are line bands, median 1.6x the line pitch tall and 0.95 pitch above the baseline (80 random GT pages); our slots are 1.0x and 0.7. A traced ink outline matched the GT polygons worse than the slot (median IoU 0.37 vs 0.72, 1,933 lines). If labels should match the GT convention exactly, widen the slot, don't tighten it.
- [ ] Narrow columns get no crops: a margin note narrows a multi_column page's columns to 590 px at training scale, under the builder's 640 px minimum, and narrow regions only join a wide one (the v2 training rule). Decide: crop narrow columns side by side, or keep such templates out.
- Checked 2026-10-21: `test_labels.py` passes with `xmlschema` installed (22 tests); our ALTO is valid against the official ALTO v4 schema but can't carry line roles; our PAGE keeps roles on a round trip. PAGE schema checked only against 2013 (diffu-page's own test); the official repo is now at 2024.

## diffu-page: verify before relying on it

- [ ] Tune the sampler to the GT before using synthetic templates. Measured 2026-10-21 (300 GT pages vs 300 sampled): running-text lines per page median 30 (real) vs 9 (sampler); line pitch 0.029 vs 0.090 of page height; line width 0.82 of the text block vs exactly 1.00 (every synthetic line fills its column). 156 of 300 sampled pages have no running text by design (tables, ledgers, registers, landskap, blank). Margins weren't comparable (single-leaf replay re-pads at a fixed 3%). Until tuned: real templates only.
- [ ] The `landskap` template gives no crops: the windowing rejects its short label lines.
- [ ] `layout/__init__.py` re-exports many private names (old flat-module surface).

## FLUX pipeline

- [ ] Look at tonight's results: v2 eval, steering test (style by name), first generated crops (`runs/flux3_pages/smoke`).
- [ ] Decide on v3 (style by example) from the steering results; its results land tomorrow morning.
- [ ] Full pages: generate at 2–4 MP, or stitch windows that share a v3 reference.
- [ ] Speed: about 2 minutes per crop at 50 steps with CFG. Measure fewer steps and a compiled transformer before using them.
- [ ] Read tonight's profile (`runs/flux3_pages/smoke/profile/trace.json`, Perfetto; pytorch-skills:profile-model): which SDPA backend runs on sm_121 (flash/efficient/cudnn/math by kernel name), GPU idle vs busy, top kernels, text-encoder vs denoise vs VAE vs TrOCR share.
- [ ] Ray Data at scale, the diffu-runner pattern: template/refill/plan as CPU map steps, `CropGenerator` as a stateful GPU actor (`map_batches`, concurrency = GPUs), TrOCR read-back co-located, Lance sink. Decide from the profile whether to batch same-bucket crops per forward pass.
- [ ] Style score: a classifier on real crops to measure whether a generated crop matches the requested archive or reference.
- [ ] Caption grammar is shared by training and generation but lives in a data-building script (`build_layout_crops.py`).

## Modular Diffusers conventions (checked against huggingface.co/blog/modular-diffusers, 2026-10-21)

- [ ] Text encoder as its own pipeline: pop `text_encoder`, encode prompts in batches, feed `prompt_embeds` to the denoise pipeline (the blog's pattern). For Ray Data a separate stage that streams ahead of the GPU denoiser. Size the win from tonight's profile.
- [ ] One `ComponentsManager` shared by the v2 text2image, v3 image_conditioned and text-encoder pipelines (no double loads).
- [ ] Load TrOCR through `load_components` from the block's ComponentSpec, then fix its meta position table; today it bypasses the spec (`load_trocr`).
- [ ] Self-contained blocks (no imports of local scripts) in the page-synth package, so they can be published with `save_pretrained(push_to_hub=True)` from the custom-block template.
- Done right already: `get_workflow` + `sub_blocks.pop/insert`, standard block shape, automatic data flow (only `crop_plan` is a pipeline input), `load_components(dtype={...})` (`dtype=` is this diffusers version's name; the blog's `torch_dtype` is older), `update_components`.

## diffu (line model)

- [ ] Old-recipe baseline to step 12k, to tell whether the refactor helped; then resume the canvas run from step 10k.
- [ ] Default recognizer `Riksarkivet/trocr-large-handwritten-hist-swe-3-char` is 404 on the Hub.
