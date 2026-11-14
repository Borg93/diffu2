# Curating the pools of old-labelled scans: what we know

The ledger of the curation work: what was measured, what was seen, what works, what does not, and what is open. Add to
it the day a thing is learnt. The code is `packages/curate` (`diffu-curate`); the reader runs are in
[HTR_RUNS.md](HTR_RUNS.md).

## The goal

Two pools of archive scans were read in 2024 by the old pipeline (htrflow 0.1.3: YOLO regions and lines, TrOCR), their
labels kept as ALTO files. We want a pretraining set for the whole-page readers out of them, then fine-tune on
GT-Lejonet and score on the same 200 test pages as every reader so far.

| Pool | Labels | Scans | Volumes | Scans |
|---|---|---|---|---|
| `svea` | `/mnt/work/haas/deliveries_old/svea-all` | `/mnt/scratch/haas/input/svea` | 2,019 | 1,251,856 |
| `andraarkiv` | `/mnt/work/haas/output/andraarkiv` | `/mnt/work/haas/input/andraarkiv` | 4,360 | 1,902,401 |

Svea is one archive (Svea hovrätt); andra arkiv spans many series. No volume is in both.

## Rules

- **Single pages only.** Every training image is one page: an opening is cut at its gutter, a scan that cannot be cut
  is left out.
- **Sample and verify.** Every table, filter and fix has a random sample looked at before it is reported, 20 pages or
  more a group, each flag and tail its own sample. Every surprise below came from looking, not from a summary number.
- **Checks score, selection drops.** A check adds a column; a training set is selected from the columns afterwards.
- **The GPUs' queue comes first.** Side work runs on the CPU while a run or a queue holds GPUs 1 and 2
  (`train_pages.sh` waits until they hold no process at all, and a side job slowed training by half).

## The tables (`runs/htr/curate/`)

| Table | A row | Made by |
|---|---|---|
| `scans.lance` | a scan of a pool: what its ALTO file alone says (3,154,257 rows) | `diffu-curate features --pool P --table …` |
| `pages.lance` | a single page: pointers to its scan and its rectangle, lines, what makes its cut doubtful (5,415,081 rows: andra arkiv 3,294,208, svea 2,065,920, gt 54,953) | `diffu-curate pages --pool P\|gt --table …` |
| `calibration/`, `calibration_andraarkiv/` | `scans.lance`, `pages.lance`, `lines.lance`: old label against truth | `diffu-curate calibration --pool P --out …` |

`pages.lance` holds every id of the existing GT sets (53,436 of 53,436, no duplicate id) and all 200 test-read pages,
marked test.

## What the old labels are worth (measured)

| | Svea | Andra arkiv |
|---|---|---|
| Calibration scans (old label + truth) | 259 | 3,392 |
| CER over matched lines | 7.4 % | 4.8 % |
| Lines exactly right | 40 % | 45 % |
| True lines missed | 2.7 % | 1.4 % |
| Old outline area on true outlines | 98.5 % | 98.3 % |

- As a page reader, scored as our test reads are: **10.7 %** on 207 test pages. On 11 pages all three read: old labels
  11.2 %, Nemotron 7.6 %, LightOnOCR-3 3.4 %. Our readers already read better than the labels.
- The calibration scans are easier than the pools (median page confidence 0.996 against 0.9905), so a typical pool page
  is nearer 6-8 % on its matched lines. In Svea, 848 GT scans hold a 2024 Transkribus transcription in the pool, not
  the old label.
- About 30 % of the differences from truth are convention: case 10 %, punctuation and spacing 20 %; ß for ss, y for ij,
  ¬ for a hyphen. The misreadings are mostly a/e, ä/a, m/n. A single wrong digit ("14" for "11") is common and
  invisible to every check.
- Reading order is right on most scans (median 98 % of lines in the truth's relative order); the worst cases are the
  two pages of an opening swapped, which cutting removes.

## What looking at pages showed

- **"No lines" is mostly not blank.** Of 10 such scans, 5 full ledger pages the segmenter missed, 2 "EMPTY PAGES"
  sheets, 1 microfilm target, 2 blank. 85,272 scans. Never use them as blank-page examples.
- **Writing the labels skip.** About 6 % of normally labelled pages (5+ lines) have under 70 % of their writing under
  a polygon; e.g. `40006282_00841`, confidence 0.991, its whole right page unlabelled.
- **Rotated writing gets fluent invented labels** ("Johansson och dhe säkerhet till den 12" on a vertical address);
  sideways ledgers are left unlabelled.
- **High confidence is not error-free.** At 0.9967: "sedeslösheten" labelled "sedrelåthetene".
- **Repeated lines are two things.** Short ones are table content ("Dito", "24."); a long line said twice is one line
  segmented twice.
- **Not everything is handwriting.** Typed SVAR title cards, placeholder sheets, some printed pages.
- **The pools look unlike the test set.** Greyscale microfilm: Svea 69 %, andra arkiv 89 %, GT test pages 26 %.

## Cutting to single pages

`diffu_page`'s cut is read from the labelled lines alone, and fails in three known ways (rows of `pages.lance`):

| Flag | Pages | What is wrong | Seen in a sample |
|---|---|---|---|
| `alone` | 621,386 | the facing page has no line: the cut is the middle of the band without text, beyond the fold | 5 of 8 carry a strip of the facing page |
| `spread` | 154,874 scans | wider than tall and no band free of text: thrown away | 11 of 12 are ordinary well-filled openings (~310,000 good pages) |
| whole, nearly square | 8,281 | a gutter is only looked for in a scan wider than tall | 11 of 12 are two pages in one image |
| whole, no lines | 85,272 | nothing to cut by | see above |

Pages with both sides labelled (4.27 million): 22 of 24 clean in a sample. The GT sets have the same flaws at a small
scale (about 30 whole openings, strips on some one-side pages).

## What works

| Thing | Evidence | Use |
|---|---|---|
| A linear "writing here" layer on frozen DINOv3 ViT-L or TIPSv2-L patch features | IoU 0.93 with true outlines on 60 held-out pages (240 to train); the two backbones' maps agree at 0.94; 4 of 4 flagged pages real | the check for unlabelled writing: first rank |
| Agreement of the two backbones' maps | disagree (IoU < 0.6) on 5 % of pages, the faint and damaged ones | sends a page to be looked at |
| TIPSv2's PCA of patch features | best single component correlates 0.84 with true outlines (DINOv3 0.77); lines and ledger rules visible | the viewer |
| INSID3's position removal for DINOv3 (project features off the subspace a blank image's features span, 500 directions) | DINOv3's PCA becomes as readable as TIPSv2's | the viewer only: scores unchanged (0.865 / 0.864 / 0.859) |
| A gutter layer on TIPSv2 features pooled a column | 99.5-99.7 % of held-out openings cut inside the gap between their text blocks; 8-10 % false alarms on single sheets | the reference; 14 s a scan on CPU, too dear for 870,000 pages |
| The old pipeline's page confidence | ranks pages by text error in every collection (rank correlation -0.60 to -0.76) | one feature among several; never a filter alone (uncalibrated; correlation 0.21 with writing covered) |
| Text length against line width | flags 1 % of lines; 34 % of those badly wrong against 12 % | a weak, free feature |
| DINOv3 ViT-L page embeddings (class token + mean patch token, page fitted to 448 px), 0.4 s a page on CPU | by one example: the 11 nearest to a typed SVAR card are all typed cards; the nearest to an "EMPTY PAGES" sheet are that, "BLANK PAGES / BLANKA SIDOR", "SLUT / END" and "BÖRJAN / BEGIN" sheets; identical "END" sheets pair at cosine 1.000. In 30 k-means clusters of 3,007 pages: one is the ruled registers with no labelled line (156 pages, 99 % Svea), one the folded letter covers with a small address, six are 90-100 % no-line pages | dropping what is not handwriting by example, near-duplicates, and naming the kinds of page for balancing |
| A 0.19 M-parameter CNN on a 256 x 256 greyscale page thumbnail saying which way up the writing is, trained on 12,400 of our pages turned on purpose (square input, so the page's shape gives nothing away) | 99.94 % on 1,600 held-out pages; reads the writing, not the scan: the sideways ledger of `A0068129_00755` 270 degrees at 0.97 with its left page upright at 1.00, the same with clamps and edges cropped away, and 211 of 240 on text-only crops. On 20,000 pool pages: 34 turned at confidence >= 0.9 (0.17 %; of 24 looked at, about 20 truly turned, 2 typed pages wrongly called upside down), 5.7 % under 0.9 (looked at: pages with little writing, covers, folded letters with mixed directions). 7 ms a page | an `orientation` column and its confidence; low confidence marks sparse or mixed pages |
| A 0.21 M-parameter CNN on a 384 x 256 greyscale thumbnail of a scan saying, a column, where the gutter is; trained on 13,000 openings with both pages labelled (the cut between their text blocks, from `pages.lance`) and 6,000 single sheets | 99.3 % of 1,032 held-out openings found, 3.8 % false alarms on single sheets, 90 % within 2 % of the width of its label; of 16 disagreements looked at, the network is on the fold in about 12 and the label (the middle of the gap between text blocks) is off, so its true accuracy is nearer 97 %. Several "false alarms" are a strip of the facing page it rightly finds. On the flagged rows (3,000 scans each, 16 looked at each): `alone` a gutter in 92 %, on the fold in about 14 of 16, and the line-based cut off by more than 2 % of the width in 83 % of them (median 5.9 %); `spread` a gutter in 94 % (85 % sure), on the fold in 15 of 16; nearly square whole scans 67 %, right in about 13 of 16 (2 pairs of loose sheets missed); tall single sheets 2.6 %. 14 ms a scan on CPU, thumbnails at 260 scans a second: every scan of both pools in about 3.5 hours on CPU | the cut for the 870,000 doubtful pages, and to recover the ~310,000 pages of the "spread" scans |
| Lance for every table | the page table's ids join to the GT sets' | adopted; version tags to set up |
| A 10-token no-repeat guard when Nemotron reads | 13.6 % to 7.55 % on the 200 test pages | every Nemotron read |
| Training Nemotron longer on the GT (4 epochs) | 7.55 % to 6.73 %, though its dev loss was worse (1.167 against 1.155) | dev loss is a poor guide here; score test pages |

## What does not work

| Thing | Evidence |
|---|---|
| AnyUp (4x finer DINOv3 features) | writing-map IoU 0.803 against 0.855 for plain interpolation; worse on 30 of 30 pages. It follows object edges; a page of writing has none. Resolution is not the bottleneck (coarse patches 0.847) |
| INSID3 segmenting writing from a reference page | IoU 0.72 (prose reference), 0.67 (ledger), 0.73 (both) against the layer's ~0.85: blobs, not lines, and across the gutter |
| A plain ink mask | stains, bleed-through and faded ink |
| Page confidence as the filter | blind to writing the pipeline did not find |
| NeMo Curator as a tool | nothing for documents, polygons or handwriting (its stage pattern is borrowed) |
| A side job on a training GPU | slowed Nemotron from 11 s to 16-18 s a step and once left 211 MB free |
| Building a real set without `--split-from-gt-test-lists` | the trainer refuses it; cost the control experiment 45 minutes |

## Open

- **Old labels or our own reader's?** The control experiment (`nemotron-parse-svea-pretrain` on 53,490 unfiltered Svea
  pages, then `nemotron-parse-svea-finetune`): compare with 6.73 %. If it does not help, re-read the pool pages with
  our best reader and keep the old labels as a second opinion (CER-HV's method, arXiv 2601.16713).
- **The writing layer on ledgers and faint ink:** patchy, because it learnt from prose pages. Train it on ledger pages
  too, or a line finder fine-tuned on GT outlines (SAM 3: needs checkpoint access).
- **The gutter network into the package:** a `gutter` step writing the image's gutter and its score for every scan, the page rows re-cut from it, then the GT and the 200 test pages re-cut and the readers re-scored once. Loose sheets lying side by side are its weak case.
- **Orientation at scale:** run the small network over every page (thumbnails load at 260 scans a second); typed pages fool it, so apply it after the non-handwriting pages are set aside. PaddleOCR's classifier not compared: ours is at 99.9 % in-domain.
- **Page embeddings at scale:** 0.4 s a page on CPU is 25 CPU-days for 5.4 million pages, so a GPU pass, or a smaller DINOv3. Then the non-handwriting kinds named from a handful of examples each, and the duplicates.
- Not run: DINOv3's text queries (typed cards against handwriting), Upsample Anything, Embedding Atlas (the viewer's
  cluster browser), Taulu (pre-printed registers only, ~1.6 % of the pool; cells, not text), table models for
  calibration (KumoTabular, TabICLv2 against boosted trees).
- Is the goal a better score on the mixed test pages, or a better reader for these archives? The pools are mostly
  microfilm of a few series.

## Backlog, in order (agreed 14 Nov)

1. **Single pages first, as a Lance table of pointers.** Built; the image-based gutter to correct the 870,000
   doubtful cuts; then re-cut the GT and the 200 test pages and re-score the readers once.
2. **Calibration per single page:** each score against truth, combined into one estimate of a page's error.
3. **The pipeline as code:** drop non-handwriting and turned pages, score, calibrate, deduplicate and balance, tagged
   training tables, a viewer.

The two small networks' weights are in `runs/htr/curate/models/` (`small_gutter.pt`, `orientation.pt`). Prototypes not yet
in the package are kept in `runs/htr/curate/prototypes/` (`two_backbones.py`, `gutter.py`,
`small_gutter.py`, `orientation.py`, `page_embed.py`, `debias.py`, `anyup_test.py`, `insid3_test.py`): each moves into `diffu_curate`
when it has earned its place.
