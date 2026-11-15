# Curating the pools of old-labelled scans: what we know

The ledger of the curation work: what was measured, what was seen, what works, what does not, and what is open. Add to
it the day a thing is learnt. The code is `packages/curate` (`diffu-curate`); the reader runs are in
[HTR_RUNS.md](HTR_RUNS.md).

## Start here: state and task list (updated 15 Nov 19:00)

Read this section, then "Rules". Update it whenever a job starts or ends or a task changes state.

**Running** (all in tmux session `htr`; machine dmlpai01, GPUs 1 and 2 only, GPU 0 is another user's):

| Where | Window | What | Writes | When it ends |
|---|---|---|---|---|
| CPU | `gutters` | `diffu-curate gutters`: the gutter network over every scan (1,700,000 of 3,184,889 at 19:00, ~57 scans/s, done about 02:00 on 16 Nov); then `pages --gutters` writes `pages_cut.lance`, **not to be used as it is** (task 1) | `runs/htr/curate/gutters.lance`, `gutters.log` | a table only; nothing reads it |
| GPU 1 | `vlserve5` | Nemotron 3.5 Super VL served in Docker (`runs/htr/curate/serve_vl.sh 1 8011`; 96 GB of the GPU; stop with `docker stop nemotron-vl`) | `runs/htr/curate/vl/serve.log` | stays up while task 9 runs |
| GPU 1 | `vldescribe` | the model describing 45,000 scans (20,000 Svea, 20,000 andra arkiv, 5,000 GT; `prototypes/vl_describe.py`, 2.7 scans/s, ~4.7 h from 18:50) | `runs/htr/curate/vl/sample.jsonl`, `sample.log` | resumable: rerun the same command |
| GPU 2 | `poolread`, `poolread2` | Nemotron Parse reading 53,490 Svea pool pages, one from the start, one from the end (`runs/htr/verify/pool_read.py`) | `pool_read/svea_pilot.jsonl`, `svea_pilot_rev.jsonl` | resumable |
| (stopped) | | the andra arkiv pool read, stopped 15 Nov 18:40 at 14,400 of 56,130 pages for the VL model's GPU; `runs/htr/curate/gpu1_queue.sh` resumes it (a second reader: the `poolread3` command with REVERSE) | `pool_read/andra_pilot.jsonl`, `andra_pilot_rev.jsonl` | restart when GPU 1 is free |

**Tasks, in order.** 1-3 are the agreed backlog (below); 4-8 are experiments asked for. A GPU that falls idle takes the
next GPU task at once.

| # | Task | State | Next step |
|---|---|---|---|
| 1 | Single pages: every scan cut at its gutter from the image | **the cut is too coarse and clips writing (2.2 % of scans); to be redone after task 9, not used as it is**; gutter pass running; the cut looked at by group and one bug fixed (section "The image cut looked at before the re-cut") | when `gutters.log` says done: `pages --gutters`, sample 20 cut pages per group again on andra arkiv and the GT, re-cut the GT and the 200 test pages, re-score the readers once; re-read the pool pages whose rectangle changed |
| 2 | Calibration per single page: one estimate of a page's label error from all scores | not started | needs 1; inputs: page confidence, the writing layer's coverage (4), reader disagreement (5) |
| 3 | The pipeline as code: drop, score, calibrate, deduplicate, balance, tagged training tables, a viewer | not started | needs 2 |
| 4 | Writing layer (flags writing without a label) | tuned and checked (section "The writing layer, tuned and checked"): it separates pages with a quarter of their lines unlabelled from complete ones (AUC 0.95 on test), a tenth less well (0.88); DINOv3, TIPSv2 and both are equal, so DINOv3 alone, linear head, threshold 0.5; heads saved in `writing_tune/heads.pt` | move it into `diffu_curate` as a score column (`covered`: the share of predicted writing inside the labelled lines' closed region); run it on the 3,651 calibration scans and sample the lowest-scoring pool pages to see what real misses it finds |
| 5 | Reader disagreement with the old label, per page and per cluster | reads running (15 Nov 17:00: Svea 15,137 of 53,490, andra arkiv 11,069 of 56,130; ~2,300 and ~2,500 pages an hour); median disagreement 12 % and 11 % of characters; 10 % and 8 % of pages disagree by more than half, and 24 of those looked at (section "Pages where our reader and the old label disagree by more than half") | **the pilot sets use the old line-position cut, so part of the disagreement is bad cuts**: join the reads to the image gutter (`gutters.lance`) and redo the look on pages whose cut the image confirms, each text read against the image in full; then the bands under 5 %, 10-20 % and 20-50 %; then score clusters (MinerU2.5-Pro's method, below) |
| 6 | Map the pools: embeddings and EVoC | embedders compared (TIPSv2 best for page similarity); EVoC run on 3,007 pages | tune PCA/UMAP/EVoC settings; name clusters with Toponymy and a local model; balance by cluster size^0.2 |
| 7 | Curated pretraining: ~100,000 filtered, image-cut pages of both pools, then a 4-epoch GT fine-tune, against 6.73 % | not started | needs 1 and a filter from 4 and 5; the unfiltered control gave 7.18 % against 7.55 % |
| 9 | **Describe each scan before it is cut** (Gabriel, 15 Nov): a vision-language model answers a fixed form about a scan; the answers and captions name the clusters (Toponymy) and are spread to all scans by a head on the page embeddings; every answer a column, the filter chosen afterwards | pilot done (section "The vision-language model describing scans"): 267 of 268 scans answered, 11 of 12 looked at right, its fold position a guess (0.5 for three in four), 2.7 scans/s; 45,000 scans being described | when `sample.log` says 45000: look at 20 scans of every kind and layout it names; embed the same 45,000 scans whole (TIPSv2, DINOv3), cluster with EVoC, name the clusters with Toponymy from the captions (read its docs first: can it call this server?); train the head and score it on held-out scans; then the columns on `scans.lance` |
| 8 | Line finder test: PaddleOCR's text detector against RF-DETR-Seg and the old YOLO, on GT PAGE polygons | proposed, **not approved** | ask before starting |

**Every check is a column (Gabriel, 15 Nov evening).** What describes a scan before it is cut goes on `scans.lance`
(a row a scan): the gutter's score and place, the vision-language model's answers (kind, one page or opening, empty
side, columns, writing at the fold, turned writing, caption) with a column saying whether the model gave them or the
head on the embeddings did, the cluster, the embeddings. What describes a page goes on `pages.lance` (a row a page):
the writing layer's `covered`, our reader's disagreement with the old label, the old pipeline's confidence, and Jev's
score of the page's lines (mean, lowest tenth, share of lines it flags). Columns are added with Lance's merge, the
tables not rewritten. Jev first on the calibration scans, where the truth says whether its page score ranks the
label's error; over the pools only if it does (it chose the better of two readings 61 % of the time).

**Questions waiting on Gabriel:** is the goal a better score on the mixed 200 test pages, or a better reader for these
archives (the pools are mostly microfilm of a few series)? Start task 8? SAM 3 needs checkpoint access on Hugging Face.

**The artifact** (claude.ai/artifact/HXE6bQA4rhE5AkS9JvNkVm, version 9, 15 Nov 13:55) carries everything up to the writing layer's check; its source is a scratch file of one session, so a new session reads it with the Artifact tool before changing it.

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

## Correction (15 Nov): the ground truth's line outlines are in its PAGE files, not its ALTO files

GT-Lejonet ships each scan twice, as PAGE (`page/`, 30,470 files) and ALTO (`alto/`, 29,406), in 225 volumes. Its ALTO
lines are boxes only (21,515 of 21,515 sampled: `HPOS/VPOS/WIDTH/HEIGHT` and a baseline); its PAGE lines carry the real
outlines (16,012 of 16,136 sampled are polygons). The reader sets take PAGE where both exist, so they are right; but the
curation prototypes read the ALTO: the writing layer was trained and scored on **boxes**, not polygons, and the
calibration's outline overlaps compare old polygons with true boxes. So "IoU 0.93 with true outlines" is with true
boxes, and a normal page's "0.89 of its writing covered" is partly boxes being wider than writing. Line matching and
the error rates by line are unaffected (they use boxes on both sides by design). To redo: the writing layer and the
outline measures on PAGE polygons.

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
| The "spreads" the line-based cut throws away | the gutter network finds a gutter in 94.2 % of 4,000; of 24 it does not cut, 15 are not pages (microfilm reel targets "Volym 383", "Rulle 391", reel numbers, typed archive cards, a "TOMMA SIDOR" sheet), 8 are real openings it misses (dark or blurred microfilm), 1 a single letter: about 2 % of spreads are truly missed openings | recover them; the rest goes with the non-handwriting pages |
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

## The control experiment's answer (15 Nov 10:01)

Nemotron 3.5 Parse pretrained one epoch on 53,490 random Svea pool pages with their old labels unfiltered
(`nemotron-parse-svea-pretrain`, dev loss 2.07 to 1.81), then fine-tuned on the GT exactly as `nemotron-parse-all`
(`nemotron-parse-svea-finetune`, dev loss 1.139 at its end against 1.155), read the 200 test pages with the guard:

| Run | GT epochs | CER | WER |
|---|---|---|---|
| `nemotron-parse-all` | 2 | 7.55 % | 20.5 % |
| `nemotron-parse-svea-finetune` (pool pretraining first) | 2 | **7.18 %** | 19.4 % |
| `nemotron-parse-more` | 4 | **6.73 %** | 18.8 % |

- Pretraining on the old labels helps a little: 0.37 points, better on 123 pages and worse on 52, but a bootstrap
  over the pages puts the gain between -0.13 and +0.84 points (95 %): not shown beyond doubt on 200 pages.
- Two more epochs on the GT helped more (0.82 points) for about twice the GPU time (9.5 h against 5 h).
- By collection: the Svea hovrätt test pages 8.63 % to 6.25 % (6,863 characters only: few pages), andra arkiv's
  collections 6.09 % to 5.36 % (as much as 4 GT epochs), the other collections (SSA, FNA) unchanged (8.57 % to 8.60 %).
  The pool's pages teach the reader its own archives and nothing else.
- So unfiltered old labels are worth little, and the gain sits where the pool's archives are. Not yet tried: filtered
  and image-cut pages from both pools, and pretraining followed by the longer fine-tune.

## Open

- **Old labels or our own reader's?** Unfiltered old labels gave a small gain (above). Next: filtered, image-cut pages
  of both pools; or re-read the pool pages with our best reader and keep the old labels as a second opinion (CER-HV's
  method, arXiv 2601.16713).
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

## EVoC on DINOv3 page embeddings (15 Nov, first step of the queued experiment)

EVoC on the 3,007 pages embedded with DINOv3 (2,048-d: class token and mean patch token): four layers of 95, 55, 23
and 9 clusters; at 23 clusters 22 % of pages are noise; 27 duplicate groups (54 pages). Every cluster looked at (12
pages each):

| Cluster | Pages | What it is |
|---|---|---|
| 0 | 65 | microfilm reel targets and typed archive title cards |
| 4 | 62 | placeholder sheets: BLANK PAGES / BLANKA SIDOR, SLUT / END, BÖRJAN / BEGIN, FORTSÄTTNING / CONTINUED, Volym nn |
| 12 | 48 | blank or near-blank microfilm openings |
| 15, 18, 19 | 109 | microfilm pages with a few lines in a corner or folded letter covers |
| 16 | 59 | blank colour pages, covers, paper edges |
| 1, 2 | 155 | colour ruled registers, 57-82 % without a labelled line (Svea, 20th century) |
| 3 | 116 | ledgers and tables (median 41 lines) |
| 5, 8, 9 | 346 | colour prose |
| 7, 11, 13, 14 | 1,044 | microfilm prose, split by contrast and density |
| 17, 20, 21, 22 | 218 | one series in one hand with decorated headings (andra arkiv only) |
| noise | 675 | mixed |

So without labels, the non-handwriting kinds (clusters 0, 4, 12, 16: about 230 pages, 8 %), the unlabelled registers
and a single series' hand each come out as their own cluster. fast_hdbscan on a 2-D UMAP of the same embeddings: 36
clusters, 12 % noise.

## Which embedder (15 Nov): TIPSv2, then DINOv3; EmbeddingGemma 2 well behind

The same 49,998 pages (20,000 Svea, 20,000 andra arkiv, 10,000 GT) embedded by each model on GPU 1
(`runs/htr/curate/prototypes/embed_pages.py`, `embed_gemma.py`; `runs/htr/curate/embed/`). Scored on the 10,000 GT
pages, whose volume (219) and collection (57) we know: of each page's 10 nearest pages, how many share its volume or
collection (independent of any clustering), and how well EVoC's clusters match the collections (adjusted mutual
information).

| Embedding (class token + mean patch token; EmbeddingGemma 2 its own 768-d) | Same volume | Same collection | EVoC vs collection (AMI) |
|---|---|---|---|
| chance | 0.029 | 0.069 | |
| **TIPSv2-L** | **0.726** | **0.863** | 0.520 |
| DINOv3 ViT-L | 0.653 | 0.799 | 0.520 |
| EmbeddingGemma 2, image + label text | 0.434 | 0.648 | 0.329 |
| EmbeddingGemma 2, image | 0.424 | 0.570 | 0.216 |

- TIPSv2 places a page among its own volume's pages more often than DINOv3 (0.726 against 0.653); their EVoC clusters
  match the collections equally. EmbeddingGemma 2 is far behind on page images, and adding the label's text helps it
  only a little.
- A volume is a scribe's or a few scribes' series and also one scanning session, so "same volume" mixes hand with
  camera, paper and lighting. It measures how well a model groups pages that belong together, not hands alone.
- The first run of the DINOv3/TIPSv2 job wrote nothing: an edit deleted a constant and a blanket `except` turned the
  error into "skip this page" for all 50,000 while it printed 70 pages a second. Rerun with only image errors caught,
  after a 20-page check of the output.
- The page table's `collection` for GT pages and the calibration's were the GT root folder's name (it starts with GT-
  as the collections do); fixed (3417d95) and the GT rows and both calibrations rebuilt.

## The writing layer, tuned and checked (15 Nov)

A head on frozen patch features says "writing here"; trained on 1,600 GT pages' PAGE line polygons, chosen on val
volumes, confirmed on test volumes (400 pages, 92 of them colour ledgers and tables drawn on purpose).
`prototypes/writing_tune.py` (two resolutions, results only) and `writing_eval.py` (1,024 px, heads saved in
`runs/htr/curate/writing_tune/heads.pt`, rows a page in `eval.json`).

- **The embedder does not matter here.** Test IoU against the line polygons: DINOv3 + TIPSv2 0.729, DINOv3 0.723,
  TIPSv2 0.713 (both − DINOv3 +0.006, 95 % 0.001 to 0.010); MLP against linear and 1,536 against 1,024 px: no
  difference. So DINOv3 alone, a linear head, 1,024 px.
- **IoU's ceiling is 0.84, not 1.** The line polygons themselves put on the 16-pixel patch grid and back overlap the
  polygons by 0.841 on test (prose greyscale 0.899, prose colour 0.826, ledger colour 0.793). The heads' 0.72 is 86 %
  of what any map on this grid can reach.
- **What it is for: does it notice missing labels?** `covered` = the share of the predicted writing that lies inside
  the labelled lines' region (polygons closed over the gaps between lines). On complete GT pages its median is 0.95
  (ledger colour 0.92). With a run of consecutive lines' polygons removed, how often a degraded page scores below an
  intact one (AUC over test pages; 0.5 is chance), DINOv3 linear at 0.5:

  | Lines removed | All | Prose greyscale | Prose colour | Ledger colour (92) | Ledger greyscale (8) |
  |---|---|---|---|---|---|
  | 10 % | 0.883 | 0.955 | 0.873 | 0.836 | 0.938 |
  | 25 % | 0.950 | 0.984 | 0.954 | 0.901 | 1.0 |
  | 50 % | 0.972 | | | | |

  Every feature set and head is within 0.01 of these. Recall of the polygons 0.88, 0.91 of the predicted writing
  inside the closed region.
- **What this does not show.** The removed lines are made up: a run of neighbouring lines of a page whose labels are
  otherwise complete. The old pipeline's real misses (a whole margin, a faint hand, a register's columns) have not
  been scored; next is the calibration scans and a sample of the pool pages with the lowest `covered`. Ledgers on
  greyscale rest on 8 pages.
- The earlier figure of 0.93 was against the ALTO boxes and is void.

## Pages where our reader and the old label disagree by more than half (15 Nov, 24 looked at)

Of the pool pages read so far, 1,549 of 15,137 Svea pages (10 %) and 841 of 11,069 andra arkiv pages (8 %) have an
edit distance between Nemotron's reading and the old label above half the label's length. By length alone:

| | Svea | Andra arkiv |
|---|---|---|
| old label under 100 characters | 549 | 277 |
| reader over 1.5 times the label | 314 | 160 |
| about the same length | 416 | 242 |
| reader under two thirds of the label | 218 | 115 |
| reader under 100 characters, label longer | 52 | 47 |

12 drawn at random from each pool and looked at, image beside both texts:
- **The label is incomplete, the reader reads the page** (about half): registers and ledgers where the old pipeline
  labelled a heading or a few cells and the reader returns the whole table (a label of 98 characters against 775 on a
  full register page; 15 against 1,157 on a full page of prose).
- **Pages that are not a page of handwriting**: typed archive title cards and reel targets (2 of 12 in andra arkiv),
  near-blank sheets with a stray word, a letter cover with an address, a page that is mostly a drawing.
- **Hard pages where neither text can be trusted**: heavy bleed-through, faint microfilm; both texts are noise.

**Corrected the same day (Gabriel, looking at the 24): the sample itself is partly wrong pages.** The two pilot sets
were built before the image gutter, with the cut from the old labels' line positions; 15,084 of the 15,137 Svea pages
read are a `_left` or `_right` half. At least 6 of the 24 are bad cuts: a strip of the neighbouring page inside the
crop (3 Svea), a page cut through its text so the label holds only the margin's words (the "15 against 1,157" page
above), and two typed title cards (a volume's scan 00001) cut in half. On those the disagreement comes from the cut,
not from the label or the reader. And the readings were not checked line by line against the images: "the reader is
right" was judged from length and the first 90 characters.

So what stands: a disagreement above half marks a page not to train on as it is. What does not stand: the shares of
each kind, and any claim about which text is right. To redo on image-cut pages (task 1), reading each sampled page's
two texts against the image in full.

## The image cut looked at before the re-cut (15 Nov evening, ~170 Svea scans by group)

Each scan drawn with the image's gutter and the old line cut on it, 12 a group
(scratch scripts; groups from `gutters.lance` joined to `pages.lance`).

| Group (Svea, share of scans) | What the 12 showed |
|---|---|
| score ≥ 0.9 (66 %) | 12 of 12 on the gutter |
| score 0.7-0.9 (28 %) | on the gutter or within a few % of it in a blank band |
| score 0.5-0.7 (3.6 %) | 11 of 12 on the gutter or a sheet's edge; one loose letter cut through |
| image sure, lines' cut elsewhere (5 %) | the image right in ~11 of 12 |
| image sure, facing page unlabelled (`alone`, 15 %) | 12 of 12 on the gutter; the old cut off in all |
| image sure, scan taller than wide (0.4 %) | 12 of 12 are openings of narrow books: the old table took each for one page |
| image sure, no labelled line (3.7 %) | openings cut right; one placeholder card cut at 0.18; two blank openings 5-8 % off |
| **image unsure (< 0.5), lines say opening (1.4 %)** | **36 of 36 hold two pages or sheets** |
| image unsure, no line, wider than tall (1.1 %) | reel targets, title cards, placeholders: one image each, rightly whole |
| image unsure, taller than wide, lines (0.2 %) | 12 of 12 single pages |

- **Bug, fixed (d1dd487):** a scan the network was unsure of became one whole "page". 17,082 Svea scans have lines
  that say opening there. Now: cut at the image's gutter when the lines' gutter is within 4 % of the width of it
  (5,783 scans; 24 of 24 looked at right), else whole with `spread` set, not a page (11,299: the right cut was the
  lines' in some and the image's in others).
- **Measured:** on 6,000 Svea scans the image cut passes through 3 or more labelled lines in 0.4-0.5 % of scans, at
  every score from 0.5 up.
- **Known, not fixed:** on blank or faint openings the gutter can sit 5-8 % of the width off (3 seen); the strip it
  moves holds no writing. A loose sheet lying over an opening is cut through.
- **Corrected the same evening (Gabriel: the cuts still go through text).** The groups above were judged on
  thumbnails, where a cut 1-2 % of the width off looks right. At full resolution, on 5,940 Svea scans with a gutter
  found, the cut lies inside one or more label boxes in 2.2 % of scans, two or more in 0.9 %, three or more in 0.5 %.
  Of 12 of the two-or-more looked at as strips: 4 cut letters or a column of numbers (tight bindings with writing
  up to the fold, registers), the rest are label boxes that reach over the fold. Of 12 random scans: none cut text, 2
  sit 2-5 % off in a blank band. The cause is how the network is made: 0.2 M parameters trained from nothing on
  384x256 thumbnails, answering on 96 columns (1 % of the width each, 40-60 scan pixels), taught the middle of the
  gap between the two pages' labelled text, not the fold, and never scored against folds marked by hand. It finds
  that a scan is an opening; it does not place the cut to the pixel. To do: a second step that places the cut at
  full resolution near the network's answer, scored against folds marked by hand on a few hundred scans.
- **What it means for the pilot sets and the pool reads:** both use the old line cut, so they hold `alone` pages
  with a strip of the facing page, tall openings as one page, and title cards cut in half. Pages whose rectangle
  changes in `pages_cut.lance` are to be read again; the rest of the reads stand.

## The vision-language model describing scans (15 Nov evening, task 9's pilot)

NVIDIA Nemotron 3.5 Super VL (120 B parameters, 12 B active, NVFP4, 75 GB; C-RADIO vision encoder) on GPU 1, asked
to fill a form about a whole scan: caption, kind, layout, written side, columns of running text, table across the
fold, writing at the fold, turned writing, where the fold is (`runs/htr/curate/prototypes/vl_describe.py`).

- **Setting it up** (each cost a start): our vLLM environment cannot serve it (it holds no built flashinfer kernels
  and cannot build them: its nvcc and CUDA headers disagree), so it runs in the card's container
  (`vllm/vllm-openai:v0.32.0`, 33 GB on the root disk, which has 40 GB left) with the flags of NVIDIA's cookbook for
  one RTX Pro 6000 (context 16,384, 128 sequences, 5 draft tokens). Scans are sent at 1,536 px on the longer side
  (the cookbook's size for documents), thinking off, the JSON asked for in the prompt. **A schema forced by the server
  gave broken JSON and "both pages empty" for nearly every opening**; the cookbook asks in the prompt, and so do we.
- **Pilot: 268 Svea scans, 20 from each of the groups looked at today.** 267 answered; 2.7 scans a second with 32
  requests in flight (2,270 tokens in, 104 out a scan): 45,000 scans in under 5 hours, all 3.15 million in 13 days,
  so a sample and a head on the embeddings, not every scan.
- **Its answers follow what the groups are:** the unsure wide scans without lines are signs and cards (15 of 20) and
  "not a page"; the unsure tall ones covers and spines (10 of 20); the tall scans the gutter network cuts are
  openings (19 of 20); the scans with one labelled side are "one page empty" (12 of 20). Of 12 looked at with their
  answers, 11 right; the miss a single register page with a strip of the next page, called an opening with both
  pages written.
- **Its fold position is a guess:** 0.50 for 149 of the 198 scans it sees a fold in. It says that there is a fold and
  which side is written, not where to cut. The cut's place still needs its own step.
- Table across the fold: 1 % and turned writing: 0 % in the pilot: too few to judge those two answers; to check on
  scans known to be such (ledgers over both pages; the orientation network's turned pages).

## Experiments queued (asked for)

1. **Map the pools with image embeddings and EVoC** (asked for 15 Nov). Embed single pages with DINOv3 (the page
   embedding above), TIPSv2 and EmbeddingGemma 2 (`google/embeddinggemma-2`, Oct 2026: one 768-d space for text,
   images, video and audio, interleaved; 0.74 B parameters), cluster each with EVoC (TutteInstitute/evoc: UMAP +
   HDBSCAN specialised to embedding vectors, CPU, several granularities, near-duplicate detection built in), and
   compare: do the clusters separate the kinds of page we know (prose, ledgers, registers, letters, typed cards, reel
   targets, placeholder sheets, blanks, turned pages) and the hands, and which model separates them best? fast_hdbscan
   (TutteInstitute/fast_hdbscan) as the clustering under a 2-D map. EmbeddingGemma 2 can also embed a page with its
   old label's text interleaved (`"<|image|> <label text>"`), which may separate pages whose label does not match the
   image. Then name the clusters with Toponymy (TutteInstitute/toponymy: names clusters at several scales with an
   LLM; local models through Hugging Face or LlamaCpp, so no outside service) from each cluster's label texts or
   captions. Start on the 3,007 pages already embedded with DINOv3, then a stratified sample of about 50,000 pages on
   a GPU when the queue allows. Looked at by sampling every cluster, not by its size alone.

   Research found 15 Nov (each read at its source) to fold in:
   - **Score clusters, not only pages** (MinerU2.5-Pro, arXiv 2604.04771, 2026): cluster page embeddings, score a
     sample a cluster by how far several readers agree, drop clusters of invalid content, weight clusters of easy
     pages down and mixed ones up. Ours: our readers against the old label a cluster; a cluster with a high median
     error is a kind of page or a labelling failure, a lone page with a high error a bad label (CER-HV's method).
   - **Balance by cluster size to a power** (DynamiCS, arXiv 2604.27932, ECCV 2026, code released): sample a
     cluster in proportion to its size^α (α = 0.2 in the paper), drawn afresh every epoch; k-means on DINOv2 there.
   - **PLSCAN** (fast_plscan, BSD-3; arXiv 2512.16558): HDBSCAN without the minimum cluster size, giving the most
     persistent layers; on 5-10-d reductions, compared with fast_hdbscan and EVoC.
   - **SCALAR** (TutteInstitute/scalar, 2026): cluster representatives and spread the labels back by kNN, for the
     full 5.4 million pages without a UMAP step.
   - **HiRank** (TutteInstitute/hirank, 2026): an outlier score from reverse-kNN ranks: the pages least like any
     other, to sample for non-handwriting and odd scans.
   - **Settings**: DINOv3 was the best of five embedders for zero-shot clustering in a 2026 benchmark (Aalborg,
     arXiv 2602.03894): label about 1,000 pages by kind and hand by hand and score the embedders and reductions by
     V-measure the same way.
   - **The nearest precedent**: HAICu (Twente, Groningen; Zenodo 17606825, poster, no code) mapped ~4 million Dutch
     colonial archive scans with DINOv2 + openTSNE and saw hands, tables and density separate.

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
