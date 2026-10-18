# diffu-page augmentation spec

## Goal

`diffu_page` generates synthetic handwritten documents whose **layout** and
**scan-appearance** match the real GT-Lejonet corpus (`/mnt/work/GT-Lejonet_HTR_202602`, 59 volumes,
~31k scans, 60,892 transcribed PAGE-XML), so the output is faithful HTR training augmentation.

Single leaves by default, **two-page spreads at the sampled fraction**. Real scans are mostly two-page
book spreads; the corpus-proportion sampler emits the prose-like kinds as an open-book `spread` at
`LayoutMix.spread_frac` (default 0.35) and single portrait leaves otherwise. The tabular kinds
(register / ledger / semi_table / landskap) stay single-leaf until the `spread` template can lay out a
ruled grid per leaf. XML replay still reduces a spread XML to one leaf via `to_single_leaf`.

Corpus facts the conditions are calibrated to (measured, `scripts/audit_augmentation.py --corpus-stats`):
real pages carry a **median ~55 lines** (range 1–215); paper tone/contrast span **bitonal-white →
warm-sepia → faded-gray → yellowed**; recurring physical features are gloved-operator fingers at the
edges, a curved+shadowed binding gutter on one edge, water stains / mottling / holes / fold creases /
curled corners, and a bleed-through that ranges from invisible to a whole facing page.

## Conditions (verified by `scripts/audit_augmentation.py` + `tests/test_augmentation_conditions.py`)

Each condition is either a hard **invariant** (a unit test asserts it) or a batch **distribution** target
(the audit measures it over a batch and the contact sheet is eyeballed). `[inv]` = test, `[dist]` = audit.

- **C1 density** `[dist]` — batch median lines/page ∈ [40, 70]; the batch spans sparse (≤15) to dense (≥100).
- **C2 scan-profile variety** `[dist]` — the batch draws ≥4 named profiles (`bitonal`, `warm_sepia`,
  `faded_gray`, `yellowed`); per-page mean-luminance spread (stdev) ≥ 22 and a wide contrast spread.
- **C3 horizontal fill** `[inv]` — a full body line fills ≥ 0.90 of its slot; a short/ragged line stays
  ragged (< 0.90 allowed); no line is stretched beyond `spec.max_stretch` × its natural ink width.
- **C4 in-bounds** `[inv]` — every seated line's ink lies inside the page (0 pixels outside), across the batch.
- **C5 label validity** `[inv]` — every `LineLabel.polygon` has ≥3 points, all within `[0,W]×[0,H]`;
  reproducible: same `(spec, seed)` → identical page + labels.
- **C6 gloved fingers** `[dist]` — a labelled gloved-fingertip occluder appears at a page edge on a
  realistic fraction (≈15–40%) of pages, never covering more than a small text area.
- **C7 single-page inner gutter** `[dist]` — a spine-side curvature + shadow on ONE edge appears on a
  fraction of pages (a single page cut from a bound spread).
- **C8 damage variety** `[dist]` — across the batch, each of {water-stain/wash, hole, fold-crease,
  curled-corner} appears ≥ once, at realistic per-page rates.
- **C9 letter/sparse layout** `[dist]` — a fraction (≈15–25%) of pages use the letter format: text in the
  top region, a blank lower area, a signature line near the bottom.
- **C10 bleed variance** `[dist]` — per-page verso bleed strength spans subtle → strong across the batch.
- **C11 leaf/spread geometry** `[inv]` — geometry matches the page kind: a single leaf is one
  portrait-ish page (no spread geometry); a `spread` page is landscape-ish (an open-book opening of two
  leaves around a central gutter). Single leaves by default, spreads at the sampled `spread_frac`.
- **C12 tabular/register variety** `[dist]` — the batch includes the corpus's non-prose layouts: a TYPED
  ruled grid (`semi_table`: proportioned columns with typed cells — row number / name / place / date / text
  / right-aligned money — + a `Summa` totals row), a leader-dot register (`register`: section headers +
  `name — value` entries with a
  leader rule), a numeric ledger (`ledger`: a wide description column + narrow right-aligned number
  columns + `Summa` subtotal rows), and a `landskap` boskapslängd/landskapshandling (sparse DISPLAY-headed
  tax/cattle roll: `due ···· figure` leader rows grouped under enlarged section headers + `Summa`). Each
  must carry furniture rules and keep every label in bounds.
  Real ledger/register geometry can also be replayed one-leaf via `load_xml_layout` (single-page split).
- **C13 corpus-proportion sampling** `[dist]` — `diffu_page.sampler.sample_page_spec` draws page kinds in
  the measured GT-Lejonet type mix (`_CORPUS_WEIGHTS`: prose-dominant, sparse letter/blank tail, structured
  register/ledger/grid minority). Over a batch the empirical per-kind fraction matches its target within a
  tolerance (audit: `max|emp−target| ≤ 0.04` over 4000 draws), so a generation run is corpus-faithful by
  default rather than hand-planned. Every drawn kind must build a valid, in-bounds single-page layout.
- **C14 label/XML correctness** `[inv]` — every composed page's ground truth is not just *present* but
  *correct*: `validate_labels` checks each line's polygon has ≥3 vertices; polygon/baseline/word points are
  in-bounds; the baseline midpoint sits on its line and runs left→right; each word's centroid falls on its
  line; word transcriptions reconstruct the line text; and reading order is a contiguous `0..n-1`
  permutation. `validate_xml_roundtrip` confirms the PAGE / ALTO / JSON exports are well-formed and preserve
  the line count. The audit runs both over every batch + table page (0 problems required) and draws the GT
  onto two pages (`overlay:prose`, `overlay:semi_table`) so the labels can be eyeballed against the ink.
  (Tolerances are half-line-height, so the invariants hold *after* the scan warp — a pass means the labels
  are right, not that warping was disabled.)
- **C15 two-page spreads** `[inv]` — spreads drawn from the sampler (`spread_frac=1` over the
  spread-eligible prose-like kinds) are structurally correct: a landscape canvas (~2 leaves + gutter),
  text laid out in two leaves around a text-free CENTRAL gutter band (no box intrudes; one leaf may be
  blank, as on real openings), `validate_layout` clean, and every composed label point in bounds.

## Verification loop (how "done" is proven)

1. `uv run pytest packages/diffu-page/tests/ -q` → 0 failures (no existing test weakened/skipped). [C3,C4,C5,C11]
2. `uv run python scripts/audit_augmentation.py` → prints a PASS/FAIL line per condition + 0 in-bounds
   violations over ≥12 pages, and writes a contact sheet. [C1,C2,C6–C10 + C4 batch]
3. A human/agent **Reads the contact sheet** and reports ≥3 concrete visual observations mapped to
   conditions, confirming no single failure mode dominates.

Status is tracked by the audit report itself — a FAIL line is the punch-list for the next iteration.
