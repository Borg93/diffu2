# The page pipeline

How a synthetic page is made, how the models behind it are trained, and what keeps training and generation
consistent. The module layout is in the [README](../README.md); open work is in [TODO.md](../TODO.md).

## From a page to labelled crops

Every step is a Modular Diffusers block; `diffu_page.pipeline` assembles our blocks around a model's own.

```
PAGE/ALTO page ──► LoadLayout ──► Refill ──► PlanCrops ──► per crop: CropPrompt ──► model's blocks ──► judge ──► repair ──► labels
 (any real page)   (lines,        (any       (training-   (caption,    (text encoder,     (reads      (rejected    (PAGE/ALTO,
                   regions,       text)      scale        boxes,       VAE encoder,       every       lines drawn  unclear lines
                   blocks)                   windows,     size)        denoise, decode)   line)       again)       tagged)
                                             or whole)
```

1. **Layout** (`LoadLayoutStep`, `diffu_page.layout`): a real PAGE or ALTO page read as its lines
   (`load_xml_layout`): each line's text, outline, baseline, region role and source block, regions in Transkribus
   reading order; an element without an id is keyed by its place, so it stays a region of its own. The training-crop
   builder cuts its crops from this same layout (`crops.regions_of`, at the scan's size), so a page is planned for
   generation exactly where its training crops were cut. PAGE and ALTO are only formats here: the layout is what the
   rest uses. Given a side (`PageJob.side`), it reads that page of a two-page opening as a page of its own
   (`layout.scan_pages`: the scan split at its gutter, the page's lines and ruled lines in its own coordinates), so
   its crops and labels are a single page's; diffu-generate gives each page of an opening its own job
   (`jobs.single_pages`).
2. **Refill** (`RefillStep`, `layout.refill`): the running-text lines get new text at the page's own writing density;
   a blank line in the text ends a paragraph and the next starts on a fresh line; margin notes, headings and page
   numbers keep theirs.
3. **Crops** (`PlanCropsStep`, `diffu_page.crops`, `diffu_page.plan`): the page is cut into the windows the models
   were trained on: whole lines, about 1 megapixel at a 64 px line pitch, NeMo bucket sizes, margin notes joined to
   the column beside them. Each window is a `CropPlan`: its size, its caption and its lines with boxes (0–1000, y
   first) and reading order. Windows overlap by design (training windows share lines), so a page's crops are not a
   tiling of the page.
4. **Caption** (`diffu_page.captions`): FLUX.3's layout prompt. A paragraph names the archive, years, scan, paper and
   ink and cites every line; a JSON table gives each line's box and text. Training data and generation build it
   with the same code, and the phrases that carry the style are spelled in one place (`captions.prompt.phrases_for`),
   which restyling rewrites; a printed line's row and a ruled line's name no style and are left as written. A page drawn from a reference crop (`plan.Reference`, style by example) is captioned as the
   reference pairs were (`restyle.reference_caption`): the style's words give way to `<ref_image_0>`, plain (v3, v3c)
   or with the reference's hand, ink and paper described as measured (`captions.hand`: slant, pen width, letter
   size, spacing, colours as hex; v3d). The form must be the one the LoRA in use was trained on.
5. **Drawing** (`diffu_page.pipeline`): a model directory names its blocks in `modular_model_index.json`, and
   `FAMILIES` maps those to the model's blocks and pipeline class (FLUX.3 Klein: `Flux3AutoBlocks` without its prompt
   upsampler, which would rewrite the quoted lines; diffu: `DiffuAutoBlocks`; MOSS-Image: `MossAutoBlocks`, a
   `diffu-moss-init` directory), the guidance scale it draws at unless `--guidance` names one (4; MOSS's 3) and
   whether LoRAs load into it at draw time (not into MOSS's: a MOSS LoRA is merged into the DiT folder its directory
   names). Around them:
   - `crop_blocks`: `CropPromptStep` (plan → prompt, boxes, size), `GlyphMapStep` (a glyph model's glyph map of the
     plan), `ReferenceGridStep` (a reference image on the training grid), the model's text encoder, VAE encoders (only
     with a reference image or a glyph map), denoise and decode. The model's auto blocks choose plain or
     reference-conditioned denoising from the inputs, so v2 (style by name) and v3 (style by example) share it. Both
     denoise branches carry a line repair (`blocks.repair`), idle without a mask; `crop_blocks` puts it into a copy of
     the model's denoise blocks, because diffusers shares block instances across every pipeline in a process.
   - diffu's denoise step is FLUX.3's with a glyph model's glyph map in both branches
     (`modular_pipelines/diffu/denoise.py`, in its own copy): a glyph model (`diffu-init --glyph`, `glyph_channels` in
     its model index) takes the map's latents beside its own on channels, each image token with its own cell of the
     map, in every guidance pass, and steps its latents with the first channels of the transformer's prediction (FLUX.3's
     transformer predicts as many channels as it takes). Every other model draws through it as through FLUX.3's own;
     `GlyphMapStep` draws the map only for a glyph model.
   - MOSS-Image's blocks (`diffu_moss`, our implementation around its released weights): `MossLayoutPromptStep`
     compiles the plan's caption and lines (`CropPromptStep`'s `prompt` and `page_lines`) into MOSS's Layout prompt,
     as the MOSS prompt-enhancer skill compiles a draft: the caption's prose as the summary, a text item per line with
     its exact text and its box in pixels from the canvas centre (`yxyx`), the paper, fragments and ruled lines as
     objects. The prompt is encoded by the frozen Qwen3.5-9B text encoder (its last hidden states over the chat
     template, thinking enabled), refused past 4,096 tokens. The denoise step is laid out as FLUX.3's (`prepare_latents`,
     `set_timesteps`, a loop of `denoiser` and `after_denoiser`, the latents as tokens in FLUX.3's order), so the line
     repair runs in it unchanged; the noise is drawn in fp32, the sampling grid is the shifted steps moved to the
     nearest level of the 1,000-step training grid (the scheduler's shift, 5, is the sampling shift, and the grid is
     stepped as given), and guidance combines in fp32 with the unconditional pass skipping block 22 (its model index's
     `uncond_skip_layers`). The FLUX.2 VAE decodes (`diffu_moss.latents`, which also encodes a drawn crop for a
     repair). It takes no reference image. On a tiny random model it draws bit for bit what the scratchpad's verified
     MOSS code (`moss_draw`) draws, and that code redrew the official demo's published images at 20.9 and 26.2 dB.
   - The judge (`diffu_page.judge`: a `Judge`, by default `TrOCRJudge`) reads every line box of the drawn crop, and the
     labels (`blocks.ground_truth.crop_labels`, ALTO or PAGE) tag a line it rejects `unclear`: a read-back CER above
     `max_line_cer`, or any number read otherwise. Drawing, judging and labelling are separate stages
     (`pipeline.draw_crop` runs the three in one process).
   - `pipeline.repair_lines` draws a crop's rejected lines again in place: the crop is drawn from new noise while
     after every step the latents outside the lines' boxes are set back to the drawn crop's own at that noise level
     (`(1 - sigma) * drawn + sigma * noise`), so the lines are redrawn in the context of the page and every other
     pixel stays where it was. A redrawn line replaces the old one only if the judge passes it; the labels then
     follow the judge's verdicts on the result.
   - `PagePlanBlocks`: the plan steps alone, no model. `jobs.crop_jobs` turns a page's plans into crop jobs, crop k
     named `<page>_<k>` and drawn from noise seed `seed + k`, each with the page's reference image if it has one (so a
     reference only ever reaches the crops beside the captions planned for it).
   LoRAs load through the model's own LoRA loader as named adapters (`diffu_page.lora`); with
   `DrawSettings.fuse_loras` (`--fuse-loras`) they are fused into the transformer's weights at load, which saves the
   adapters' matmuls every step but fixes the LoRAs for the run; `DrawSettings.compile_blocks` (`--compile-blocks`)
   then compiles the transformer's repeated blocks (diffusers' regional compilation), refused around unfused LoRAs.

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
- diffu trains from `diffu-init`'s repo in NeMo's finetune mode on caches its own encoders make
  (`diffu_train.cache_diffu`; `cache_glyphs` with each crop's glyph map), with the configs in
  `packages/train/configs/diffu/`, through `torchrun -m diffu_train.finetune`: NeMo's diffusion recipe, which trains a
  glyph model with `diffu_train.nemo.GlyphConcatAdapter` (each pair's glyph map beside its noisy latents on channels,
  the loss on the latents' channels) in place of NeMo's FLUX.3 adapter.
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
| Glyph map (a glyph model) | `glyphs.glyph_map` of the crop's lines at its size, encoded by `diffu_train.cache.encode_image` | `GlyphMapStep`, the same function on the plan; diffu's glyph encoder encodes it as the cache does (`test_cache`); each token's cell beside its latents, as `GlyphConcatAdapter` puts it |
| Position ids | image (0, h·0.5, w·0.5, 0), reference l = −1, text (0, 0, 0, i) | the same |
| No-prompt input | v3d: a tenth of the pairs on the empty prompt as NeMo encodes a caption (`diffu_train.dropout`), bit-identical to diffusers' unconditional input; earlier runs none (`cfg_dropout_prob: 0`; NeMo's own would use zeros) | the empty prompt |

MOSS-Image trains with KohakuMossTrainer (its own environment, `~/Desktop/repos/kohakumoss`), on manifests whose
captions the scratchpad's `moss_prompt.layout_prompt` compiled: `diffu_moss.layout_prompt` compiles a crop plan's
caption to the same string (`packages/page/tests/test_moss.py`). The trainer's text caches are written with
transformers' forward, one prompt a call (`forward: hf`), as `MossTextEncoderStep` encodes; its latents (the VAE's
mode, 2x2, batch-norm statistics) matched ours at cosine 0.99998 (27 Oct).
