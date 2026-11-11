# The page pipeline

How a synthetic page is made: a real page's layout, new text and a style in; drawn crops or whole pages with their
labels out. The repo map is in the [README](../README.md); how the models are trained: [MOSS.md](MOSS.md) and
[FLUX3.md](FLUX3.md); open work in [TODO.md](../TODO.md).

## From a page to labelled crops

Every step is a Modular Diffusers block; `diffu_page.draw.pipeline` assembles our blocks around a model's own.

```
PAGE/ALTO page ──► LoadLayout ──► MarkFromScan ──► OpeningPage ──► Refill ──► PlanCrops ──► per crop: CropPrompt ──► model's blocks ──► judge ──► repair ──► labels
 (any real page)   (lines,        (if asked: print  (a side of     (any       (training-   (caption,    (text encoder,     (reads      (rejected    (PAGE/ALTO,
                   regions,       and rules read     an opening,   text)      scale        boxes,       VAE encoder,       every       lines drawn  unclear lines
                   blocks)        off the scan)      if asked)                windows,     size)        denoise, decode)   line)       again)       tagged)
                                                                              or whole)
```

1. **Layout** (`LoadLayoutStep`, `diffu_page.page.layout`): a real PAGE or ALTO page read as its lines
   (`load_xml_layout`): each line's text, outline, baseline, region role and source block, regions in Transkribus
   reading order; an element without an id is keyed by its place, so it stays a region of its own. The training-crop
   builder cuts its crops from this same layout (`crops.regions_of`, at the scan's size), so a page is planned for
   generation exactly where its training crops were cut. PAGE and ALTO are only formats here: the layout is what the
   rest uses. A line's PAGE `@production` (printed or handwritten) is kept when the file states it and is None when
   it does not (drawn as handwriting); `SeparatorRegion`s are the page's ruled lines. Most GT PAGE files state
   neither, so with a print/hand classifier (`diffu-generate --mark-from-scan`) `MarkFromScanStep` marks the template
   from the scan beside it first (`diffu_page.page.marking.marked`): each line it says nothing of is read as printed or
   handwritten by a small CNN (`marking.print_hand`, 64 px high line images reduced to their ink, read window by
   window; less than 90 % sure stays handwritten), lines it states keep theirs, and when it has no ruled lines they are
   found on the scan (`marking.rules`, OpenCV: long thin straight runs of ink, told apart from handwriting, page edges,
   an opening's fold and the scan's border). The whole scan is marked, then `OpeningPageStep`, given a side
   (`PageJob.side`), reads that page of a two-page opening as a page of its own (`layout.scan_pages`: the scan split at
   its gutter, the page's lines and ruled lines in its own coordinates), so its crops and labels are a single page's;
   diffu-generate gives each page of an opening its own job (`jobs.single_pages`). A scan wider than tall whose lines
   leave no gutter (`layout.is_unsplit_spread`: on the GT 1,176 of 30,983 scans, about 880 openings whose pages' lines
   overlap side to side, the rest mostly single wide sheets, which the lines cannot tell apart) is refused here, with a
   side or without, and a run records it as failed: a spread is never drawn, and the page-pair builder skips it too.
   The classifier is trained on CPU (`python -m diffu_page.page.marking.print_hand_set`, then
   `python -m diffu_page.page.marking.train`): printed lines from the OCR ground truth, handwritten ones from the HTR
   ground truth, validated on held-out collections.
2. **Refill** (`RefillStep`, `layout.refill`): the running-text lines get new text at the page's own writing density;
   a blank line in the text ends a paragraph and the next starts on a fresh line; margin notes, headings and page
   numbers keep theirs. On a page whose running text is handwritten, a form's printed lines keep theirs too, and the
   density is the handwriting's; a page whose running text is all printed, a printed book's, has its print rewritten.
   Lines with no text to measure (a layout given as boxes) take `CHARS_PER_LINE_ASPECT` (3.7) characters per unit of
   their width over their height. A job without text skips this step: every line keeps its own. A job may give its
   layout inline (`PageJob.layout`) instead of a template: it is one page, neither marked from a scan nor split.
3. **Crops** (`PlanCropsStep`, `diffu_page.page.crops`, `diffu_page.draw.plan`): the page is cut into the windows the
   models were trained on (whole lines, about 1 megapixel at a 64 px line pitch, NeMo bucket sizes, margin notes joined
   to the column beside them), or planned whole as one crop at a pixel budget (`plan_whole_page`, `--whole-page`). Each
   is a `CropPlan`: its size, its caption and its lines with boxes (0–1000, y first) and reading order. Windows
   overlap by design (training windows share lines), so a page's windows are not a tiling of the page.
4. **Caption** (`diffu_page.captions`): FLUX.3's layout prompt. A paragraph names the archive, years, scan, paper and
   ink and cites every line; a JSON table gives each line's box and text. Training data and generation build it with
   the same code, and the phrases that carry the style are spelled in one place (`captions.prompt.phrases_for`); a
   printed line's row and a ruled line's name no style and are left as written. A page drawn from a reference crop
   (`plan.Reference`, style by example) is captioned as the model's training pairs were (`restyle.reference_caption`):
   the style's words give way to `<ref_image_0>`, with the reference's hand, ink and paper described as measured
   (`described`: `captions.hand`, slant, pen width, letter size, spacing, colours as hex), or in one of MOSS-Image's
   prompt forms (`plain`, `named`, `hand`, `sample`: [MOSS.md](MOSS.md)). The form must be the one the model was
   trained on (`--reference-caption`). MOSS-Image compiles the caption into its own Layout prompt.
5. **Drawing** (`diffu_page.draw.pipeline`): a model directory names its blocks in `modular_model_index.json`, and
   `FAMILIES` maps those to the model's blocks and pipeline class (FLUX.3 Klein: `Flux3AutoBlocks` without its prompt
   upsampler, which would rewrite the quoted lines; MOSS-Image: `MossAutoBlocks`, a `diffu-moss-init` directory) and
   the guidance scale it draws at unless `--guidance` names one (FLUX.3's 4, MOSS's 3). Around them:
   - `crop_blocks`: `CropPromptStep` (plan → prompt, boxes, size), `ReferenceGridStep` (a reference image on the
     training grid), the model's text encoder, VAE encoders (only with a reference image), denoise and decode. The
     model's auto blocks choose plain or reference-conditioned denoising from the inputs. Both denoise branches carry
     a line repair (`blocks.repair`), idle without a mask; `crop_blocks` puts it into a copy of the model's denoise
     blocks, because diffusers shares block instances across every pipeline in a process.
   - `diffu_page.draw.guidance.ReferenceGuidance`: diffusers' classifier-free guidance with a third pass without the
     reference, so guidance can push toward the reference as well as the caption (`--reference-guidance`).
   - The judge (`diffu_page.quality.judge`: a `Judge`, by default `TrOCRJudge`) reads every line box of the drawn crop,
     and the labels (`blocks.ground_truth.crop_labels`, written as ALTO or PAGE) tag a line it rejects `unclear`: a
     read-back CER above `max_line_cer`, or any number read otherwise. Drawing, judging and labelling are separate
     stages (`pipeline.draw_crop` runs the three in one process).
   - `pipeline.repair_lines` draws a crop's rejected lines again in place: the crop is drawn from new noise while
     after every step the latents outside the lines' boxes are set back to the drawn crop's own at that noise level
     (`(1 - sigma) * drawn + sigma * noise`), so the lines are redrawn in the context of the page and every other
     pixel stays where it was. A redrawn line replaces the old one only if the judge passes it; the labels then
     follow the judge's verdicts on the result.
   - `PagePlanBlocks`: the plan steps alone, no model. `jobs.crop_jobs` turns a page's plans into crop jobs, crop k
     named `<page>_<k>` and drawn from noise seed `seed + k`, each with the page's reference image if it has one.
   LoRAs load as named adapters, at `--lora-scale`: FLUX.3's (NeMo checkpoints) through the model's own LoRA loader
   (`diffu_page.draw.lora`), MOSS-Image's (PEFT adapters of its DiT, `diffu-moss-convert-lora`'s: [MOSS.md](MOSS.md),
   "Hand LoRAs") onto the DiT (`diffu_moss.adapters`); with `DrawSettings.fuse_loras` (`--fuse-loras`) they are fused
   into the transformer's weights at load, and `DrawSettings.compile_blocks` (`--compile-blocks`) then compiles the
   transformer's repeated blocks (diffusers' regional compilation), refused around unfused LoRAs, as MOSS-Image's MXFP8
   and AOTInductor packages are.

Who runs what: every entry point plans pages with `PagePlanBlocks` and draws each crop on its own with the crop
pipeline (`pipeline.draw_crop`). `diffu-generate` draws a run's crop jobs one after another; diffu-runner plans pages
on CPU workers and draws one crop per block on GPU actors ([RAY.md](RAY.md)); diffu-studio shows the same two seams as
`gr.Workflow` nodes, a plan node feeding a draw node per MOSS-Image model; `diffu-evaluate` draws stored plans. All of
them name and seed crops alike, so a page comes out the same from any of them.

## Running it: diffu-generate

```bash
jobs/spark_guard.sh --need-gb 50 --name pages -- uv run diffu-generate --model runs/moss/eight_full_r6b_step_000750 \
    --templates page_a.xml --templates page_b.xml --text-from other_page.xml \
    --title "the records of Överståthållarämbetet" --paper yellowed --ink brown --whole-page --page-megapixels 2 \
    --out runs/pages
```

- `--text-from` takes a PAGE/ALTO page (its running text) or a plain text file, written as it stands (a blank line
  ends a paragraph).
- `--whole-page` draws each template whole (`--page-megapixels`) instead of as training-scale windows. A template that
  is a two-page opening, as most GT scans are, is drawn as its single pages (`summary.json` lists them; a job line can
  ask for one, `"side": "left"` or `"right"`); a spread its lines cannot split is refused and recorded as failed.
- `--mark-from-scan` marks each template from the scan beside it before planning (`--print-hand-model`, by default
  `runs/print_hand/print_hand.pt`); `summary.json` names the classifier.
- `--jobs run.jsonl` gives every page its own template or inline layout, text (or none, to keep the lines' own),
  style, seed, reference and size: one `diffu_page.draw.jobs.PageJob` per line, the run file diffu-runner reads
  (fields: [INFERENCE.md section 5](INFERENCE.md#5-writing-a-run-file)):

  ```json
  {"template": "page_a.xml", "text": "Anno 1720 den 5 maj ...", "style": {"provenance": {"title": "the court records of Jämtlands domsaga", "years": "1720"}, "look": {"greyscale": false, "paper": "cream", "ink": "dark brown"}}, "page_pixels": 2000000, "id": "jamt_1720_p1"}
  ```

- `--styles styles.json --style NAME` (or a job line's `"style": NAME`) draws in a named preset
  (`diffu_page.draw.styles.StylePreset`: the archive and years, the look, a LoRA and a reference, paths relative to the
  file); a run with several presets draws one pipeline per set of LoRAs in turn (`summary.json`: `groups`). The repo's
  `styles.json` holds FLUX.3's hands; a preset of MOSS-Image's names a hand's adapter folder as its `lora`.
- `--lora hand=<folder>` draws every page whose style is spelled out with a LoRA: a NeMo checkpoint's `model/` dir for
  FLUX.3, a hand's adapter for MOSS-Image ([MOSS.md](MOSS.md), "Hand LoRAs").
- `--repair N` draws the lines the read-back rejects again in place, up to N times, each pass from its own noise and
  kept only where it reads back; `summary.json` gives each crop's lines repaired.
- `--reference <crop>` draws every page from a sample (a model trained with references), captioned in
  `--reference-caption`'s form.

## Labels for page readers

`--reader-labels` also writes each crop's labels in the formats page readers train on, beside its image and from the
same regions as its XML (`diffu_page.page.labels`): `<name>.md`, the page as Markdown (its regions in reading order, a
blank line between two, each line of the page on a line of its own, a heading after `## `, a table as HTML);
`<name>.grounded.md`, DeepSeek-OCR 2's grounded form of it (each region under
`<|ref|>text<|/ref|><|det|>[[x0, y0, x1, y1]]<|/det|>`, on 0-999 of the page); and `<name>.elements.json`, the
regions as PaddleOCR-VL's layout elements (each one's PP-DocLayoutV3 category, box in pixels, recognition prompt and
the text its recogniser reads there, a table as OTSL, and its lines with how each was made and whether it read back
badly). `diffu-reader-set` turns such a run into a fine-tuning set, JSON lines in the reader's own conversation format,
image paths absolute (train readers on `--whole-page` runs: a window's labels leave out the lines its edges cut):

```bash
uv run diffu-reader-set --run runs/pages --model deepseek-ocr-2 --out data/readers/deepseek_ocr_2.jsonl
uv run diffu-reader-set --run runs/pages --model deepseek-ocr-2 --task markdown --out data/readers/free_ocr.jsonl
uv run diffu-reader-set --run runs/pages --model paddleocr-vl --out data/readers/paddleocr_vl.jsonl
uv run diffu-reader-set --run runs/pages --model paddleocr-vl --task spotting --out data/readers/spotting.jsonl
```

Pages are sent to the readers whole, so each is taught whole pages. DeepSeek-OCR 2 gets a sample per page, asked with
its document prompt and answered with the grounded Markdown (`--task markdown`: its prompt for a page without layouts,
Free OCR, and the Markdown). PaddleOCR-VL gets a sample per page asked `OCR:` and answered with the same Markdown
(`--task page`, the default), or asked `Spotting:` and answered with each line's text and the corners of its box on
0-1000 (`--task spotting`); `--task elements` gives its recogniser a sample per element instead, the element cut from
its page into `<out stem>_crops/` and answered with what the recogniser reads there.

`diffu-reader-set --xml` reads real pages too, and a drawn run's exports: PAGE or ALTO files with their scans, a folder
of them (the GT's root, a `diffu-run-export` folder) or a file listing them, one a line, read from its folder
(`diffu_page.data.reader_pages`). Each scan is read as its layout and split into its pages, an opening's two each cut
from it into `<out stem>_pages/`; a region's PAGE type is its element's kind, an untyped region running text; a scan
both a PAGE and an ALTO file describe is read from its PAGE file (ALTO, as read, has neither types nor tables). Left
out, each with why in `<out stem>.skipped.jsonl`: a scan wider than tall that its lines cannot split, one without a
transcribed line, one that is not the size its file says, a file without its scan or that cannot be read, and the GT's
second export of a collection (`GT-RA_landskapshandlingar_20260227`, the scans of `..._20260220` again). More
readers, as ms-swift fine-tunes them (`messages` and `images`):

- `--model unlimited-ocr-2` (ms-swift's model type `unlimited-ocr`, template `unlimited_ocr`, which shows each
  `<image>` as `<PAGE><image>`): `--task parse` (the default) a sample per page, asked `<image>Multi page parsing.` and
  answered with `<PAGE>` and a line per region in reading order, `<BLOCK>{label}[x0, y0, x1, y1]<FORK>{its lines}`, the
  box on 0-1000 of the page and the label `text` (running text), `title` (a heading), `aside_text` (a margin note),
  `page_number`, `header`, `footer` (a footer, catchword or signature mark) or `table` (its lines an HTML table);
  `--task opening` a sample per scan, an opening's two pages as two images (`<image><image>Multi page parsing.`),
  answered with a `<PAGE>` each.
- `--model minicpm-ocr` (model type and template `minicpm_ocr`): `--task elements` (the default) a sample per region,
  its crop asked `<image>Text Recognition:` and answered with its lines, a table's asked `<image>Table Recognition:`
  and answered in OTSL between `<otsl>` and `</otsl>`; `--task lines` a sample per text line of every region (a table
  cell's too), its box cut into `<out stem>_lines/<page id>_<k>_<j>.jpg` (region `k`, line `j`), asked
  `<image>Text Recognition:` and answered with the line's text, as OpenBMB fine-tunes it on handwritten lines (a line
  with no text, a drawn line tagged `unclear` and a box narrower or lower than 8 px are left out and recorded); `--task page` a
  sample per page asked `<image>Text Recognition:` and answered with its plain text, a line of the page a line, a
  blank line between two regions. `--max-crop-pixels N` scales a region's or line's crop down to at most N pixels
  (200,704 = 448 x 448: a MiniCPM-OCR line in one view of 256 visual tokens).
- `--model lightonocr-3` (LightOnOCR-3-4B; ms-swift's model type and template `qwen3_5`): a sample per page, the page
  fitted to a longest edge of 2,048 px as LightOn's client fits it, cut into `<out stem>_pages/`. `--task grounding`
  (the default) asked `<image>grounding` and answered with a block per region in reading order, `![label](x0,y0,x1,y1)`
  (0-1000 of the page) and on the next line the region as Markdown, a blank line between two; the label `text`,
  `title` (a heading, after `## `), `aside_text`, `page_number`, `header`, `footer` (a footer, catchword or signature
  mark) or `table` (an HTML table). `--task page` shows the image alone (`<image>`, its transcription mode), answered
  with the Markdown.

Beside each set `<out stem>.meta.jsonl` gives a line per sample: its id (a page's is the MOSS page-pair sets', the
digest of its PAGE file's path below the folder given and its side, `<digest>_left`), `source` (`real`, or
`synthetic` for a page diffu_page drew), its collection (`GT-...`), volume, scan, side of an opening and split; a
crop's sample also its region's `kind` and place (`element`), a line's its `production` and place in its region
(`line`). A set is one variant of a reader's data: `--collection` and `--exclude-collection` choose collections,
`--max-per-collection N` caps a collection's training samples, never its test ones (`--seed` shuffles the scans
first, so the cap takes from all its volumes), and `<out stem>.options.json` records every option and what the set came to. The readers are
trained from these sets by `htr-train` ([HTR_TRAIN.md](HTR_TRAIN.md)). To keep the evaluation honest:

```bash
uv run diffu-reader-set --xml ~/Desktop/data/GT-Lejonet_HTR_202602 --model unlimited-ocr-2 \
    --split-from-gt-test-lists --exclude data/readers/generator_pages.txt --out data/readers/unlimited_ocr_2.jsonl
uv run diffu-reader-set --xml out/run1/files --model minicpm-ocr --task page --out data/readers/minicpm_drawn.jsonl
```

`--split-from-gt-test-lists` sends each scan a test list of its volume names (the GT's `page/test_*`, nested, so their
union is `test_0100`; the second export's lists count for the first's scans, the same scans) to
`<out stem>.test.jsonl` and `<out stem>.test.meta.jsonl`, never to `--out`. `--exclude` leaves out the pages a file
lists, one a line (`#` a comment): a page by its page-pair id (`<digest>_left`; the digest alone names a scan's every
page) or by the path of its PAGE file or scan, whole or its last parts (`<collection>/<volume>/<volume>/page/x.xml`, as
the training crops record a page): the pages a generator was trained on.

## In Python

A page planned into crop jobs on CPU, then each crop drawn by the crop pipeline, as diffu-generate and diffu-runner do
it:

```python
from pathlib import Path

from diffu_page.captions.look import ScanLook
from diffu_page.captions.prompt import PageStyle
from diffu_page.captions.provenance import Provenance
from diffu_page.draw.jobs import PageJob, crop_jobs
from diffu_page.draw.pipeline import DrawSettings, crop_blocks, draw_crop, load_pipeline

style = PageStyle(
    provenance=Provenance(title="the court records of Jämtlands domsaga", years="1690–1700"),
    look=ScanLook(greyscale=False, paper="cream", ink="dark brown"),
)
settings = DrawSettings(steps=50)
pipe = load_pipeline(model_dir, crop_blocks, loras={}, settings=settings)  # a MOSS-Image or FLUX.3 Klein directory
page = PageJob(template=Path("page.xml"), text=text, style=style, seed=0)  # a blank line in text ends a paragraph
crops = [
    draw_crop(pipe, job.plan, name=job.name, seed=job.seed, reference=job.reference_image(), settings=settings)
    for job in crop_jobs(page, 0)
]
# each crop: .image, .labels (the label data; diffu_page.page.labels.page_elements(crop.labels) for the page readers'),
# .labels_xml (ALTO), .line_reads, .line_cers, .unclear_lines
```
