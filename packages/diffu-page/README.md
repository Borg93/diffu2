# diffu-page

Full-page document synthesis from generated handwriting lines — a uv **workspace member**, not part
of the main training package. It composites per-line output (Diffu, Eruku, or a placeholder) into
full document **pages** with pixel-perfect labels, for training **segmentation / end-to-end** HTR
models on Swedish historical documents.

It is *docTR's page-assembly approach + a real handwriting renderer + your real ALTO layouts* — docTR
renders fonts (printed text) and cannot do handwriting; this does.

## Install & run

```bash
uv sync                                   # installs the workspace (diffu-page + deps)
uv run pytest packages/diffu-page/tests   # 179 tests, all green

# audit the augmentation vs the GT-Lejonet corpus (CPU, no model): PASS/FAIL per condition + a
# contact sheet at audit_out/contact_sheet.png to LOOK at (see AUGMENTATION_SPEC.md, C1..C15):
uv run python scripts/audit_augmentation.py

# placeholder ink (no model/GPU) — proves layout + compositor:
uv run diffu-page --template multi_column --n 3

# real handwriting (needs a GPU):
uv run diffu-page --renderer eruku --style a_line_crop.png --template single_column
uv run diffu-page --renderer diffu --ckpt run/final/model.safetensors --style a_line_crop.png
```

Label formats (`json` default; add `page` / `alto`) live in `GenConfig.formats` — set them via
`--config run.yaml` (`formats: [alto, json]`); flags override the YAML (CLI > YAML > default).

The **full `PageSpec`** — every knob (margin_keys, drop_cap, catchword, section_headings, table columns…) —
is drivable from the CLI via `--spec spec.json` (one object, or an array cycled across pages), so an agent or
user can author any layout without code:

```bash
echo '{"template":"landskap","line_height":40,"margin_keys":["N:o 1"],"drop_cap":true}' > spec.json
uv run diffu-page --spec spec.json --n 4
```

Library API (equivalent, in Python):

```python
from diffu_page import PageSpec, build_layout, compose_page, PlaceholderRenderer, PageLabels, export

spec = PageSpec(template="single_column", header_lines=["Ahr 1744"], header_scale=1.6,
                marginalia=["No 478", "Resolution"], marg_scale=0.95, page_number="61.", recto=True)
renderer = PlaceholderRenderer()                       # or DiffuRenderer(ckpt, style) / ErukuRenderer(style)
layout = build_layout(spec, paragraphs, seed=0, measure=renderer.measure_width)
#   or:  layout = load_xml_layout("your_real.alto.xml")     # replay a real layout
page, lines = compose_page(layout, renderer, paper_dir="assets/paper", seed=0)
labels = PageLabels(width=spec.width, height=spec.height, image="page.png", lines=lines)
open("page.alto.xml", "w").write(export(labels, "alto"))    # or "page" / "json"
```

## Pipeline

`layout` (geometry) → `render` (one line → ink raster) → `compose` (place + blend + degrade) →
`labels` (export). The `LineRenderer` Protocol is the seam — any line generator drops in.

## Control surface (`PageSpec`)

- **Layout source**: a template (`build_layout`) *or* a real ALTO/PAGE file (`load_xml_layout` — carries
  region semantics via `TextRegion@type`/`TableRegion` and turns `SeparatorRegion`s into furniture rules,
  single-leaf split of a spread via `prefer=`) *or* the corpus-proportion sampler
  (`sampler.sample_page_spec(rng)` → `(kind, PageSpec)` drawn in the measured GT-Lejonet type mix; reweight
  via `LayoutMix`) so a whole run is corpus-faithful without a hand-set plan.
- **Templates** (11): prose `single_column · multi_column · columns · key_value · letter`; non-prose
  `semi_table` (TYPED ruled grid: proportioned columns · row-number/name/place/date/text/right-aligned-money
  cells · `Summa` totals) · `register` (leader-dot: section headers + `name ─ value` + dashed leader) ·
  `landskap` (boskapslängd / landskapshandling — a sparse 16-17th c. tax/cattle roll: DISPLAY section
  headers over indented `due ···· figure` leader rows + `Summa`, wide right margin) · `ledger` (wide
  description col + narrow right-aligned number cols + `Summa` subtotals) · `blank_leaf` (near-empty
  substrate) · `spread` (two-leaf, split to one leaf for single-page training).
- **Text furniture** (labelled GT, from the GT-Lejonet corpus study): `catchword` (custos at the
  bottom-outer corner → PAGE `catch-word`); `underline_prob` (name/date underline → PAGE `underlined`);
  `section_headings` (centered rubric that reads in flow) + `enum_prob`/`enum_marker` (`§`/`N:o` article
  enumerators); `margin_keys` (a keyed marginal note per paragraph — case №, `Resol.`, a date — seated in
  the gutter beside its paragraph and read INTERLEAVED with the body, unlike `marginalia` which reads last);
  `drop_cap` (enlarge the opening line's first letter into a versal/`initial` box, the rest shifting right).
  Display section headers come from the `landskap` template. Manuscript edits: `insertion_prob`
  (interlinear `add`) · `deletion_prob` (struck `del`).
- **Size invariants** (enforced by pydantic): `header_scale ≥ 1.0`, `marg_scale ≤ 1.0`; body uniform per paragraph.
- **Relative geometry**: `margin`/`gutter`/`marg_gutter` are ratios of page size; `line_gap`/`para_gap`
  are fractions of `line_height`. **Alignment**: left/right/center. **Wrapping**: pass `measure=` for
  renderer-accurate wrapping. **Reading order**: page-number → header → body (with rubrics + keyed marginal
  notes interleaved in flow) → tables/registers row-major (cell-by-cell) → unkeyed marginalia last.
- **One writer per page**: the renderer holds one style reference, used for every line.

## Scan realism (`AugmentConfig` + scan profiles)

`compose_page(..., aug=, paper_tone=, trace=)` ages the seated page into a real-looking archive scan.
Four named `ScanProfile`s in `profiles.py` (`bitonal · warm_sepia · faded_gray · yellowed`) bundle a
`PaperTone` (paper hue/luminance) + a tuned `AugmentConfig`. Photometric + physical marks (all label-safe
except the geometric warps, which carry the polygons): foxing/wash/bleed-through, binding gutter shadow +
spine warp, scanner bed + edge fray + clasps, gloved-operator fingers, worm/punch holes, curled corners,
fold creases, archival stamps, dust, microfilm; scribal `paraph` flourishes + keyed `margin_mark`s. An
optional `trace` dict records which stages fired (used by the audit).

## Output

Pluggable exporters in `EXPORTERS` (dict-dispatch, none privileged): `json`, `page` (PRImA PAGE-XML),
`alto` (ALTO v4, Riksarkivet/htrflow shape with word `String`s). Add a format = one function + one
dict entry.

## Validation

- `validate_layout(layout)` — pre-render box guards (off-page / overlap).
- `validate_table(layout, lines)` — **structural** correctness of a composed table/register: cell-in-column
  (no ink crossing a rule), no cell overlap, header separation, row-major reading order. The audit rejects a
  structurally-broken table on this (not just `rules > 0`).
- `validate_labels(labels)` — **label** correctness of any composed page (C14): polygon ≥3 pts + in-bounds,
  baseline on-line + left→right, word centroids on-line, word transcriptions reconstruct the line, reading
  order a contiguous permutation. Warp-robust (half-line-height tolerances) so it holds on the aged page.
- `validate_xml_roundtrip(labels)` — the PAGE / ALTO / JSON exports are well-formed and preserve the line count.
- `overlay.draw_label_overlay(page, lines)` — draw polygons/baselines/word-boxes/reading-order onto a page
  to verify the image and the GT labels together; the audit renders it for a prose page + a table page.
- `validate.py` recognition metrics (IoU / CER / match → `MatchResult`) — model-agnostic; they await a
  detector integration (`Riksarkivet/yolov9-regions-1` + `-lines-within-regions-1` +
  `trocr-base-handwritten-hist-swe-3` — `htr.py` runs that stack end-to-end via HTRFlow today).

## Status

**Done & verified** (placeholder ink, `AUGMENTATION_SPEC.md` C1..C15, audit 15/15 PASS): 11 templates +
page elements + text furniture; faithful ALTO/PAGE replay (region semantics via `TextRegion@type`/`TableRegion`
+ `SeparatorRegion` rules) with single-leaf split (rules re-origined) + height normalization; 4 scan
profiles + the physical/scan repertoire; row-major reading order + `validate_table`; corpus-proportion
layout sampler (`sampler.py`, C13); label/XML correctness (`validate_labels` + `validate_xml_roundtrip`, C14)
with GT overlays in the contact sheet; 3 label formats + word boxes. ruff + ty clean (bar pre-existing
cv2/numpy-stub noise) + 179 pytest green; follows the `writing-python` skill (Pydantic, typed, no mutable
global state). Compositor ≈1.3 s/page (CPU). Real-ink round-trip verified too: composed pages read at
2–5% CER through the real Riksarkivet HTRFlow stack (`htr.py`) — ≈ real-document territory.

### TODO

`drop_cap` renders the enlarged initial with the placeholder ink; a true swash/display TTF would only
make it fancier. `validate.py`'s recognition metrics (IoU / CER / match) still await a detector
integration for automated page-level scoring (see Validation).

### Caveats

- **One writer per page** is by construction; writers vary *across* pages (the diffu-runner page
  pipeline draws a fresh document + hand per page job).
- **Eruku is English/modern** (IAM/CVL/RIMES) — a *pipeline test*, not domain-correct data; å ä ö may
  render poorly. The Swedish-historical model (Diffu) is the real renderer.
- **OpenCV**: default `opencv-python-headless>=4.13` (cv5 has no usable wheel — verified). For real
  cv5 speedups, compile via `scripts/build_opencv5.sh`. Code uses the cv2 API identical on 4.x/5.x.
- **`ty`**: clean except numpy/cv2 stub strictness (not bugs).
