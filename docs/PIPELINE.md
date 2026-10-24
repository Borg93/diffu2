# The page pipeline

How a synthetic page is made, how the models behind it are trained, and what keeps training and generation
consistent. The module layout is in the [README](../README.md); open work is in [TODO.md](../TODO.md).

## From a page to labelled crops

Every step is a Modular Diffusers block; `diffu_page.pipeline` assembles our blocks around a model's own.

```
PAGE/ALTO page ──► LoadLayout ──► Refill ──► PlanCrops ──► per crop: CropPrompt ──► model's blocks ──► LineReadback ──► LabelExport
 (any real page)   (lines,        (any       (training-   (caption,    (text encoder,     (HTR reads       (PAGE/ALTO,
                   regions,       text)      scale        boxes,       VAE encoder,       every line)      unclear lines
                   blocks)                   windows)     size)        denoise, decode)                    tagged)
```

1. **Layout** (`LoadLayoutStep`, `diffu_page.layout`): a real PAGE or ALTO page read as its lines
   (`load_xml_layout`): each line's text, outline, baseline, region role and source block, regions in Transkribus
   reading order. The training-crop builder reads PAGE with the same code (`layout/page_xml.py`), so a page is planned
   for generation exactly where its training crops were cut. PAGE and ALTO are only formats here: the layout is what
   the rest uses.
2. **Refill** (`RefillStep`, `layout.refill`): the running-text lines get new text at the page's own writing density;
   margin notes, headings and page numbers keep theirs.
3. **Crops** (`PlanCropsStep`, `diffu_page.crops`, `diffu_page.plan`): the page is cut into the windows the models
   were trained on: whole lines, about 1 megapixel at a 64 px line pitch, NeMo bucket sizes, margin notes joined to
   the column beside them. Each window is a `CropPlan`: its size, its caption and its lines with boxes (0–1000, y
   first) and reading order. Windows overlap by design (training windows share lines), so a page's crops are not a
   tiling of the page.
4. **Caption** (`diffu_page.captions`): FLUX.3's layout prompt. A paragraph names the archive, years, scan, paper and
   ink and cites every line; a JSON table gives each line's box and text. Training data and generation build it
   with the same code. A page drawn from a reference crop (`plan.Reference`, style by example) is captioned as the
   reference pairs were (`restyle.reference_caption`): the style's words give way to `<ref_image_0>`, plain (v3, v3c)
   or with the reference's hand, ink and paper described as measured (`captions.hand`: slant, pen width, letter
   size, spacing, colours as hex; v3d). The form must be the one the LoRA in use was trained on.
5. **Drawing** (`diffu_page.pipeline`): a model directory names its blocks in `modular_model_index.json`, and
   `FAMILIES` maps those to the model's blocks and pipeline class (FLUX.3 Klein: `Flux3AutoBlocks` without its prompt
   upsampler, which would rewrite the quoted lines; diffu: `DiffuAutoBlocks`). Around them:
   - `crop_blocks`: `CropPromptStep` (plan → prompt, boxes, size), `ReferenceGridStep` (a reference image on the
     training grid), the model's text encoder, VAE encoder (only with a reference image), denoise and decode, then
     `LineReadbackStep` (TrOCR reads every line box) and `LabelExportStep` (ALTO or PAGE labels; a line that reads back
     badly is tagged `unclear`). The model's auto blocks choose plain or reference-conditioned denoising from the
     inputs, so v2 (style by name) and v3 (style by example) share it.
   - `PagePlanBlocks`: the plan steps alone, no model. `jobs.crop_jobs` turns a page's plans into crop jobs, crop k
     named `<page>_<k>` and drawn from noise seed `seed + k`, each with the page's reference image if it has one (so a
     reference only ever reaches the crops beside the captions planned for it).
   LoRAs load through the model's own LoRA loader as named adapters (`diffu_page.lora`).

Who runs what: every entry point plans pages with `PagePlanBlocks` and draws each crop on its own with the crop
pipeline (`pipeline.draw_crop`). `diffu-generate` draws a run's crop jobs one after another; diffu-runner plans pages
on CPU workers and draws one crop per block on GPU actors; diffu-studio shows the same two seams as `gr.Workflow`
nodes, a plan node (`PagePlanBlocks`) feeding a draw node per model; `diffu-evaluate` draws stored plans. All of them
name and seed crops alike, so a page comes out the same from any of them.

## Training (NeMo AutoModel)

- `diffu_page.data.build_crops` cuts the GT pages into crops with captions; `steer_set`, `style_pairs` and
  `pick_eval` build the steering, v3 and evaluation sets (`style_pairs --caption described` for v3d's captions;
  `probe_set` the control probe that checks whether such words steer the base model at all).
- NeMo caches latents and prompt embeddings once (its FLUX.3 processor; `diffu_train.cache_pairs` for v3's
  pairs, with the empty prompt beside them), then trains a LoRA with the configs in `packages/train/configs/flux3/`
  and our interleaved bucket dataloader (`diffu_train.nemo`), which can train a share of the pairs on the empty prompt
  (`caption_dropout`, `diffu_train.dropout`) and scores `val_loss` on held-out pairs every 100 steps.
- `diffu-evaluate` scores every drawn crop for CER and for how closely its hand follows its real crop's and its
  reference's (`HandFidelity`: each drawing against its own reference and against the other references, which is
  chance); `diffu_page.cli.hands` scores a finished run's saved drawings the same way, without the GPU.
- Every training config logs to the local trackio project `diffu` (`~/.cache/huggingface/trackio`, no Space); the
  job scripts name the runs (`flux3_v3b_lora`, `diffu_s_train`, ...). Read them with the trackio CLI:
  `trackio list runs --project diffu --json`, `trackio get metric --project diffu --run <run> --metric loss --json`,
  `trackio show --project diffu`. trackio is installed in NeMo's environment (2026-10-22); logs of runs before that
  are replayed into the project from their NeMo log.
- Job scripts in `jobs/flux3/` run each GPU step through `jobs/spark_guard.sh`: one GPU job at a time on the DGX
  Spark, whose 128 GB of memory is shared by CPU and GPU.

## What keeps training and generation consistent

Checked 2026-10-21 (reference rows 2026-10-23); each would silently cost quality if it drifted.

| What | Training (NeMo) | Generation (Modular Diffusers) |
| --- | --- | --- |
| Caption | `diffu_page.captions` | the same code, from the `CropPlan`; from a reference, the same `reference_caption` call as its training pair (`test_plan`) |
| Prompt embeddings | NeMo's FLUX.3 encoder | FLUX.3's text encoder block: bit-identical on the same device |
| Latents | VAE `mode()`, 2×2 patchify, batch-norm stats | the same in FLUX.3's blocks |
| Line geometry | `page_xml.read_page_line` (builder) | the same reader (replay): 60/60 GT pages planned as cut |
| Reference image | shrunk to 0.5 MP on the 16 px grid (`crops.grid.as_reference`) | `ReferenceGridStep`, the same function: a no-op on such references, a full crop shrunk to the trained size (it was kept at 1 MP before 2026-10-23) |
| Reference look (described captions) | measured at the reference size (`captions.hand.measure_reference`) | the same function |
| Position ids | image (0, h·0.5, w·0.5, 0), reference l = −1, text (0, 0, 0, i) | the same |
| No-prompt input | v3d: a tenth of the pairs on the empty prompt as NeMo encodes a caption (`diffu_train.dropout`), bit-identical to diffusers' unconditional input; earlier runs none (`cfg_dropout_prob: 0`; NeMo's own would use zeros) | the empty prompt |
