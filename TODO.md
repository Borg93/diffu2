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

- [ ] Compare template layouts (line pitch, lines per page, margins, kind mix) with the real GT pages before using synthetic templates for anything that matters. Until then: real templates only.
- [ ] The `landskap` template gives no crops: the windowing rejects its short label lines.
- [ ] `layout/__init__.py` re-exports many private names (old flat-module surface).

## FLUX pipeline

- [ ] Look at tonight's results: v2 eval, steering test (style by name), first generated crops (`runs/flux3_pages/smoke`).
- [ ] Decide on v3 (style by example) from the steering results; its results land tomorrow morning.
- [ ] Full pages: generate at 2–4 MP, or stitch windows that share a v3 reference.
- [ ] Speed: about 2 minutes per crop at 50 steps with CFG. Measure fewer steps and a compiled transformer before using them.
- [ ] Style score: a classifier on real crops to measure whether a generated crop matches the requested archive or reference.
- [ ] Caption grammar is shared by training and generation but lives in a data-building script (`build_layout_crops.py`).

## diffu (line model)

- [ ] Old-recipe baseline to step 12k, to tell whether the refactor helped; then resume the canvas run from step 10k.
- [ ] Default recognizer `Riksarkivet/trocr-large-handwritten-hist-swe-3-char` is 404 on the Hub.
