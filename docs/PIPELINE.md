# The page pipeline

How a synthetic page is made, how the models behind it are trained, and what keeps training and generation
consistent. The module layout is in the [README](../README.md); open work is in [TODO.md](../TODO.md).

## From a page to labelled crops

```
PAGE/ALTO page ──► layout ──► refill ──► crops ──► caption ──► text pipeline ──► crop pipeline ──► image + labels
 (any real page)   (lines,     (any       (training-  (prompt:    (prompt            (draw, read back,
                   regions,    text)      scale       style +     embeddings)        PAGE/ALTO labels)
                   blocks)                windows)    line boxes)
```

1. **Layout** (`diffu_page.layout`): a real PAGE or ALTO page read as its lines (`load_xml_layout`): each line's text,
   outline, baseline, region role and source block, regions in Transkribus reading order. The training-crop builder
   reads PAGE with the same code (`layout/page_xml.py`), so a page is planned for generation exactly where its
   training crops were cut. PAGE and ALTO are only formats here: the layout is what the rest uses.
2. **Refill** (`layout.refill`): the running-text lines get new text at the page's own writing density; margin
   notes, headings and page numbers keep theirs.
3. **Crops** (`diffu_page.crops`): the page is cut into the windows the LoRAs were trained on: whole lines, about
   1 megapixel at a 64 px line pitch, NeMo bucket sizes, margin notes joined to the column beside them. Each window
   is a `CropPlan`: its size, its caption and its lines with boxes (0–1000, y first) and reading order.
4. **Caption** (`diffu_page.captions`): FLUX.3's layout prompt. A paragraph names the archive, years, scan, paper and
   ink and cites every line; a JSON table gives each line's box and text. Training data and generation build it
   with the same code.
5. **Generation** (`diffu_gen.generator.CropGenerator`, built on `diffu_gen.modular`): two Modular Diffusers
   pipelines over one `ComponentsManager`:
   - `CropTextBlocks`: the crop prompt, then FLUX.3's text encoder (Qwen3-VL). Plan in, prompt embeddings out.
   - `CropBlocks`: the crop prompt, `ReferenceGridStep`, FLUX.3's reference VAE encoder (only with a reference
     image), denoise and decode, then `LineReadbackStep` (TrOCR reads every line box) and `LabelExportStep` (ALTO or
     PAGE labels; a line that reads back badly is tagged `unclear`). FLUX.3's auto blocks choose plain or
     reference-conditioned denoising from the inputs, so v2 (style by name) and v3 (style by example) share it.
   LoRAs load through diffusers' FLUX.3 LoRA loader as named adapters.

## Training (NeMo AutoModel)

- `diffu_page.data.build_crops` cuts the GT pages into crops with captions; `steer_set`, `style_pairs` and
  `pick_eval` build the steering, v3 and evaluation sets.
- NeMo caches latents and prompt embeddings once (its FLUX.3 processor; `diffu_train.cache_pairs` for v3's
  pairs), then trains a LoRA with the configs in `configs/flux3/` and our interleaved bucket dataloader
  (`diffu_train.nemo`).
- Job scripts in `jobs/flux3/` run each GPU step through `jobs/spark_guard.sh`: one GPU job at a time on the DGX
  Spark, whose 128 GB of memory is shared by CPU and GPU.

## What keeps training and generation consistent

Checked 2026-10-21; each would silently cost quality if it drifted.

| What | Training (NeMo) | Generation (Modular Diffusers) |
| --- | --- | --- |
| Caption | `diffu_page.captions` | the same code, from the `CropPlan` |
| Prompt embeddings | NeMo's FLUX.3 encoder | `CropTextBlocks`: bit-identical on the same device |
| Latents | VAE `mode()`, 2×2 patchify, batch-norm stats | the same in FLUX.3's blocks |
| Line geometry | `page_xml.read_page_line` (builder) | the same reader (replay): 60/60 GT pages planned as cut |
| Reference image | written on the 16 px grid (`crops.grid`) | `ReferenceGridStep` (a no-op on such references) |
| Position ids | image (0, h·0.5, w·0.5, 0), reference l = −1, text (0, 0, 0, i) | the same |
| No-prompt input | none: `cfg_dropout_prob: 0` (NeMo would use zeros) | the empty prompt |
