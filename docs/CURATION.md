# Curating the pools of old-labelled scans: what we know

The ledger of the curation work: what was measured, what was seen, what works, what does not, and what is open. Add to
it the day a thing is learnt. The code is `packages/curate` (`diffu-curate`); the reader runs are in
[HTR_RUNS.md](HTR_RUNS.md).

## Start here: state and task list (updated 17 Nov 20:15: the first curated set is training, E38)

**Handover, 18 Nov 15:00 (read this first if the session or the account changed).** Everything below runs in tmux `htr` and needs nobody connected.
- Running: E40's vectors on GPUs 1 and 2 (`embedpool0`, `embedpool1`, done about 16:40); the folds on the CPU (`placepool`, about 02:00).
- Starts by itself: the columns for every scan (`poolcolumns`), the heads' check (`headscheck`), the second fine-tune from checkpoint 250 and its test read (`trusted`, score about 01:30 on 19 Nov), the fold join (`foldjoin`), the exact yield (`poolcount`, `runs/htr/curate/pool_funnel.txt`), then the trusted pages read through vLLM with the DFlash2 draft and the repetition detector (`aftertrusted3`, `pool_read/trusted_v1_vllm.jsonl`).
- In hand, not queued: **E52, our own DFlash2 draft.** vLLM's `extract_hidden_states` mode refuses Parse's config ("Multiple valid text configs ... ['decoder', 'text_config']"). Next: read speculators' offline hidden-state format and how it treats image positions, write an extractor with Transformers (layers 1, 4, 8 of the decoder over the reader's own answers), then a smoke of `speculators train --speculator-type dflash2`. Untried: whether speculators loads Parse's tied, scaled embeddings and head.
- Owed: the results page (https://claude.ai/artifact/98423D2V376HLajHfbhpS2) lacks E43's old-pipeline scores, the block scores and E50 to E54; stopped pages of the vLLM read need a second read by the guarded reader; E50's speed on a quiet GPU; E54 (the repository's postprocessing in place of our regex). Asked 18 Nov and queued: publishing the reader and its draft to Hugging Face (E55); a table model of its own, tables never mixed into the running-text sets (E56).
- Waiting on Gabriel: the size of the next pretraining set (1 million pages recommended) and whether the goal is the mixed test pages or the Svea and andra arkiv material.


Read this section, then "Rules". Update it whenever a job starts or ends or a task changes state, and give every
experiment a row in the "Experiment register" below the day it starts.

**If you pick this up (17 Nov 20:15):** the queue in tmux `trusted` (`runs/htr/trusted_pretrain.sh`, log
`runs/htr/trusted_pretrain.log`) runs by itself: the pretraining on the trusted cuts ends about 22:30 on 17 Nov, then
the GT fine-tune (2 epochs) and its test read, then 2 more epochs and their test read. When a score is written
(`runs/htr/nemotron_parse/test_read_trusted_norepeat10/cer.txt`, then `test_read_trusted_more_norepeat10/cer.txt`),
put it in E38's row and in both artifacts, against 7.55 %, 7.18 % and 6.73 %. After the last test read the GPUs are
free: read the 139,937 trusted pages with our reader for the label filter (`runs/htr/verify/pool_read.py` with
`SET=/mnt/scratch/gabriel/htr/trusted_v1_nemotron/pages`), before going back to the old pool read. The VL server is
stopped (`docker start nemotron-vl` brings it back; it holds all of GPU 1).

**Running** (all in tmux session `htr`; machine dmlpai01, GPUs 1 and 2 only, GPU 0 is another user's):

| Where | Window | What | Writes | When it ends |
|---|---|---|---|---|
| GPUs 1 and 2 | `embedpool0`, `embedpool1` | **E40's vectors, started 18 Nov 09:17 on Gabriel's word** (the second fine-tune stopped for them at its checkpoint 250): DINOv3 and TIPSv2 vectors and the saturation of the 3,068,985 pool scans with labelled lines, one shard a GPU. **Restarted 10:47 and 10:54 with `OMP_NUM_THREADS=1`** (each right after a saved part; the queue kept waiting, one shard always held a GPU): 62 and 69 scans a second (47 each before), 131 together. Checked on the GPU before the start: 40 scans of the 45,000 reproduce their stored vectors (cosine 0.9999 and 1.0000); in the run's first 40,000 scans, 541 known ones at cosine 0.9998 or more | `/mnt/scratch/gabriel/htr/embed_pool/part-*.parquet` (600,000 scans at 11:00); logs `runs/htr/curate/embed_pool_shard{0,1}.log` (a log's count starts again at a restart: count the parts, 20,000 scans each) | about 16:20 on 18 Nov; resumable |
| GPUs 1 and 2, waiting | `trusted` | E38's second fine-tune (`nemotron-parse-trusted-more`), stopped at step 316 and **resuming by itself from checkpoint 250** when both shards have ended (the queue waits while a process computes on the GPUs), then its test read | `runs/htr/nemotron-parse-trusted-more`, `runs/htr/trusted_pretrain.log` | resumes about 16:20, about 8.6 hours, the score about 01:10 on 19 Nov |
| CPU, when the vectors are in | `poolcolumns`, then `headscheck` | `pool_columns.py apply`: the heads' columns for every embedded scan joined to `scans.lance` (under the lock `scans.lock`), then `pool_report.txt` and `column_state.txt`; then each head scored with volumes held out | `scans.lance`: `cut_class`, `cut_p`, `is_table`, `table_p`, `kind`, `kind_p`, `layout`, `layout_p`, `saturation`, `colour`, `pca`; `pool_columns.log`, `heads_check.log` | about 16:30 on 18 Nov (8 minutes; tested at 860,000 scans) |
| GPUs 1 and 2, after the training queue | `aftertrusted3` (in place of `aftertrusted2`, 18 Nov 13:45) | `runs/htr/after_trusted3.sh`: the ground-truth scans without patch vectors embedded and given their columns; then our best reader reads the 139,937 trusted pages **through vLLM with the DFlash2 draft**, one server a GPU (E50) | `/mnt/scratch/gabriel/htr/embed_gt`, `pool_read/trusted_v1_vllm.jsonl` | starts about 01:30 on 19 Nov; resumable; stop it for the next training with `tmux kill-window` on `parse1`, `parse2`, `aftertrusted3` |
| CPU, when the columns and the folds are in | `poolcount` | `stage_pool.py count`: the funnel of the larger set over every pool scan, and the tables | `runs/htr/curate/pool_funnel.txt` | about 02:15 on 19 Nov |
| GPU 2, paused | (stopped by PID for E38) | E12: the andra arkiv pool read, at 24,325 of 56,130 pages; resumable | `pool_read/andra_pilot.jsonl`, `andra_pilot_rev.jsonl` | restart only after the trusted pages are read |
| CPU, 12 workers, nice 19 | `placepool`, then `foldjoin` | E40's folds: the placer over the 2,915,784 pool scans with labelled lines that the gutter network calls an opening, in shuffled order, **running since 18 Nov 09:15 beside the training** (50 scans a second; the training's steps 10.95 s with it, 10.8 to 11.0 s before); a lock (`folds.lock`) makes a second placer do nothing; when it ends `foldjoin` runs `marks_to_lance.py` (fold_x, fold_found into `scans.lance`) and writes `column_state.txt` | `folds.lance`, `place_folds_pool.log`, `fold_join.log` | about 02:00 on 19 Nov (44 scans a second beside the embedding; 266,000 of 2,801,436 at 11:00 on 18 Nov), then the join |
| queued behind `trusted` | `aftertrusted` | E40: every pool scan with labelled lines embedded by DINOv3 and TIPSv2 (GPUs 1 and 2, one shard each) and every candidate opening's fold placed (CPU, 16 workers) | `/mnt/scratch/gabriel/htr/embed_pool/part-*.parquet`, `folds.lance`; logs `runs/htr/curate/embed_pool_{0,1}.log`, `place_folds_pool2.log` | about 13 hours from the end of the training; resumable. (The placer was tried beside the training on 17 Nov evening and stopped after 22,000 scans: training steps read 10.1 s before it and 10.3 to 10.4 s with it, but also 10.3 s after it was stopped, so it may not have been the cause; it saves no time before the vectors exist) |
| ports 7861-7863 | `markfolds`, `markgallery`, `markmap` | the marking pages (127.0.0.1 only); 225 scans drawn for fold marks are unmarked | `fold_marks.lance`, `cut_marks.lance`, `region_marks.lance` | stay up |

**Tasks, in order.** 1-3 are the agreed backlog (below); 4-8 are experiments asked for. A GPU that falls idle takes the
next GPU task at once.

| # | Task | State | Next step |
|---|---|---|---|
| 1 | Single pages: every opening cut on its fold | **solved for about nine openings in ten (17 Nov)**: the placer searching 6 % either side of the middle is off by more than 1 % of the width on 3 % of openings drawn at random (E31, E33); whether to cut is answered by the head on 15,288 hand marks (E34 to E37). `pages_cut.lance` (the gutter network's cut) stays unused | the openings without a fold line (about one in ten): the trained placer (E26); then re-cut the GT and the test pages the same way |
| 2 | Calibration per single page: one estimate of a page's label error from all scores | not started | needs 1; inputs: page confidence, the writing layer's coverage (4), reader disagreement (5) |
| 3 | The pipeline as code: drop, score, calibrate, deduplicate, balance, tagged training tables, a viewer | not started | needs 2 |
| 4 | Writing layer (flags writing without a label) | tuned and checked (section "The writing layer, tuned and checked"): it separates pages with a quarter of their lines unlabelled from complete ones (AUC 0.95 on test), a tenth less well (0.88); DINOv3, TIPSv2 and both are equal, so DINOv3 alone, linear head, threshold 0.5; heads saved in `writing_tune/heads.pt` | move it into `diffu_curate` as a score column (`covered`: the share of predicted writing inside the labelled lines' closed region); run it on the 3,651 calibration scans and sample the lowest-scoring pool pages to see what real misses it finds |
| 5 | Reader disagreement with the old label, per page and per cluster | paused 17 Nov for E38. Readings of 64,950 pool pages are in `pages.lance` (median disagreement 12 % and 11 %); they are of pages cut the old way and of another sample: only 1,881 of the 139,937 trusted pages have one (median 10.6 %, 5.7 % above half) | read the trusted pages themselves once the GPUs are free; then the filter and the look by band |
| 6 | Map the pools: embeddings and EVoC | embedders compared (TIPSv2 best for page similarity); EVoC run on 3,007 pages | tune PCA/UMAP/EVoC settings; name clusters with Toponymy and a local model; balance by cluster size^0.2 |
| 7 | Curated pretraining: ~100,000 filtered, image-cut pages of both pools, then a 4-epoch GT fine-tune, against 6.73 % | **running as E38 (17 Nov 09:42)**: 139,937 pages of 75,776 scans whose cut is trusted, labels unfiltered; then the GT fine-tune 2 + 2 epochs | the scores against 7.18 % and 6.73 %; second round: labels filtered by reader disagreement, more tables, the openings E26 recovers |
| 9 | **Describe each scan before it is cut** (Gabriel, 15 Nov): a vision-language model answers a fixed form about a scan; the answers and captions name the clusters (Toponymy) and are spread to all scans by a head on the page embeddings; every answer a column, the filter chosen afterwards | done on 130,666 scans (100,083 of the pools, the rest ground truth), two passes, every one embedded with Qwen3-VL-Embedding 8B; the head on hand marks carries cut-or-not (E36) | describe and embed more scans if E38 shows a gain; Toponymy names for the clusters |
| 8 | Line finder test: PaddleOCR's text detector against RF-DETR-Seg and the old YOLO, on GT PAGE polygons | proposed, **not approved**, but Nemotron OCR v2's detector is asked for (18 Nov, E47a) | ask before starting |

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

### Columns asked for (Gabriel, 17 Nov): the one dataset every training set is filtered from

One Lance dataset over every scan, every check a column, so that a training set, and a set of tables apart, is a
filter. **The list of columns is `WANTED` in `runs/htr/curate/prototypes/column_state.py`; the table below is that
script's output, read from `scans.lance` itself (18 Nov 11:45, with the 860,000 scans of E40's run embedded by then). Run it at the start of a session and after every
join, and paste its output here; add a line to `WANTED` the day a column is asked for.** Order of work: vectors and
folds for every scan (E40), then the heads' columns (`head_cut`, `table`, kind), orientation, the writing layer, Jev
on the calibration scans, clusters and names for every scan, the vectors' place (read the Lance skill first).

| Column | What it says | Pool scans with a value (of 3,154,257) | Made by |
|---|---|---|---|
| `page_confidence` | the old pipeline's confidence | 3,153,409 | diffu-curate features |
| `polygon_cover` | statistics of the old labels (lines, digits, cover ...) | 3,154,257 | diffu-curate features |
| `gutter_score` | the small gutter network's opening score | 3,154,180 | diffu-curate gutter |
| `saturation` | greyscale microfilm (near 0) or colour scan | 860,000 | embed_pool.py (E40), with the vectors |
| `vl_caption` | the VL model's caption | 100,083 | vl_describe.py (a sample: too slow for every scan) |
| `vl_kind` | the VL model's kind of scan | 100,083 | vl_describe.py, then a head for every scan |
| `vl_writing_layout` | table or running text, by the VL model | 99,831 | vl_ask2.py |
| `is_table` | table or running text for every scan, by a head | 889,169 | pool_columns.py (E40), with table_p |
| `kind` | the kind of scan for every scan, by a head | 889,169 | pool_columns.py (E40), with kind_p |
| `layout` | opening, single page ... for every scan, by a head | 889,169 | pool_columns.py (E40), with layout_p |
| `cut_class` | the hand marks' class for every scan, by a head on the patch vectors | 889,169 | pool_columns.py (E40), with cut_p |
| `colour` | a colour scan or greyscale microfilm | 860,000 | pool_columns.py (E40), from saturation |
| `vl_turned` | turned writing, by the VL model | 99,831 | vl_ask2.py |
| `orientation` | which way up the writing is, by the small network | **missing** | to do: orientation.py over the pools |
| `mark` | the class marked by hand (cut, no_cut, single, not_a_page, odd) | 13,481 | marks_to_lance.py |
| `head_cut` | the hand marks' class by the head on Qwen's vectors (described scans only) | 100,148 | cut_head.py |
| `fold_x` | where the fold is | 468,105 | place_folds.py; for every scan after E40 |
| `fold_found` | whether a fold line was found | 468,105 | place_folds.py |
| `cluster` | the cluster on the map of 14,985 scans (50 clusters on Qwen's vectors) | 13,290 | scalar_hirank.py; for every scan by a head once the clusters are named |
| `cluster_name` | the cluster's name | **missing** | to do: Toponymy |
| `covered` | writing layer: share of the writing under a label | **missing** | to do: writing_eval.py's layer over the pools |
| `jev_page` | Jev's score of the page's lines | **missing** | to do: first on the calibration scans |
| `read_disagreement` | our reader against the old label | **missing** | pages.lance holds it for 64,950 pages; to do on the trusted pages |
| `dinov3` | DINOv3's vector of the scan | **missing** | embed_pool.py (E40) makes it (Parquet parts); a column when the run has ended, 13 GB a model |
| `tipsv2` | TIPSv2's vector of the scan | **missing** | embed_pool.py (E40) |
| `qwen8b` | Qwen3-VL-Embedding 8B's vector | **missing** | scan_embeddings.lance holds 130,733 |
| `pca` | the first 64 principal components of the two models' vectors (82 % of the variance), 16-bit | 889,169 | pool_columns.py (E40), heads/pca.npz |


**The artifacts.** Results: claude.ai/artifact/VFK6knSmeTYdhGXcE1Yzsq (17 Nov; source a scratch file of one session, read it with the Artifact tool before changing it). Verdicts of all 38 experiments with two archify diagrams: claude.ai/artifact/98423D2V376HLajHfbhpS2 (17 Nov 20:15); its sources are kept in `runs/htr/curate/artifact/verdicts/` (the two diagram specs, `build.py`, `template.html`, `snapshots.sh`). The first artifact (HXE6bQA4rhE5AkS9JvNkVm) is not writable from this account.

**Two kinds of embedding, two jobs (Gabriel, 16 Nov).** They are not rivals and are not to be ranked in one table.
- **DINOv3 and TIPSv2** give a vector for every patch of a page. Their job is where the writing is: the writing
  layer, its overlap with the labels, the layout, the PCA pictures.
- **Nemotron Embed, Qwen3-VL-Embedding, EmbeddingGemma** give one vector for a page. Their job is making sense of the
  collection: clusters with EVoC, oddities with HiRank, names with Toponymy.
The "Which embedder" table below scores both kinds by one yardstick (does a page's neighbour come from its own
volume), which rewards pages that look alike; it says nothing of which gives the more useful clusters. For that the
clusters are judged against what the scans are (the VL model's answers), E21.


## One table: features are columns of `scans.lance` (16 Nov)

Read first: Lance's guides and format spec at the installed version (13.0.0): data evolution, merge, JSON, row ids,
scalar indexes, maintenance. What follows from them, and what is built:

- `scans.lance` holds every scan, both pools and the ground truth (3,184,889 rows), keyed by `image` (unique). The
  version before this work is tagged `before-features`.
- A feature becomes a column by `diffu_curate.table.add_features(table, features, key="image")`. A new column goes
  in by Lance `merge`, which attaches it as its own files without rewriting the rows; rows without a value hold null.
  A column that exists is updated in place, fragment by fragment (`LanceFragment.update_columns`, committed as one
  `Update`): rows named take the new value, the others keep theirs, no row moves, and the indexes are brought up to
  date. `merge` itself refuses a column that exists, and the transaction spec calls it the operation that conflicts
  with every other write, so nothing else may write the table during a join. A refresh of the 79,000 described scans
  takes 12 s.
- The VL model's whole answer is the JSON column `vl` (stored as JSONB), filtered with
  `json_get_string(vl, 'writing_layout') = 'table'`; JSON functions filter but cannot be selected, so the fields
  filtered on are typed columns too (`vl_kind`, `vl_layout`, `vl_written_side`, `vl_text_columns`,
  `vl_table_across_fold`, `vl_writing_at_fold`, `vl_fold_x`, `vl_writing_layout`, `vl_turned`, `vl_caption`).
- Also joined: `gutter_score`, `gutter_x` (every scan), `fold_*` (the full-resolution placer, 19,660 scans).
- Bitmap indexes on `pool`, `vl_kind`, `vl_layout`, `vl_writing_layout`, `vl_turned`; B-tree on `image`.
- `runs/htr/curate/prototypes/to_lance.py` rebuilds the `vl_*` and `fold_*` columns from every result file found; run
  it again when a run ends. Checked 16 Nov: 77,440 described scans in the table, none differing from the files.
- Still files, to move: the pool reads (to `pages.lance`), the embeddings, clusters, maps, HiRank scores. The run
  scripts still write JSONL and are changed to write Lance when the running VL job has ended.
- From the docs, for what comes next. Vectors: a fixed-size-list column (float16 is supported), exact search without
  an index, `IVF_RQ` as the first index to try, `prefilter=True` to search within a kind of scan; scans of wide
  vectors need a smaller `batch_size` (the default 8,192 rows of 4,096 float16 is 64 MB a batch). Images: plain
  binary inline is right below a few tens of KB (thumbnails), a blob column from about 1 MB (page crops); neither
  is used yet, the tables point at the scans on disk. Text: an `INVERTED` index gives BM25 search over captions and
  readings. Every write makes a version and old column files stay until `cleanup_old_versions` (default: older than
  two weeks; tagged versions are kept), so the table grows with each refresh until cleaned.
- Read in full (16 Nov, Lance repo at f7ef212): every guide but the cloud sections of `object_store`; of the format
  spec `table/index`, `schema`, `transaction`, `table_config`, `clustering`, `versioning`, and the configuration part
  of `file/encoding`. Not read: the index file layouts (`format/index/*`), `mem_wal`, `fragment_metadata`,
  `data_overlay_file` (not yet supported by any library), `row_id_lineage`, `layout`, `branch_tag`, the blob spec,
  and the page layouts of `file/encoding`.

## Readers by kind of page (E22, 16 Nov)

The 200 test pages joined to their scans' kinds in `scans.lance` (every page matched), each reader's stored reading
scored as before (edits summed over characters of truth).

| Kind (VL model) | Pages | Nemotron Parse | LightOnOCR-3 |
|---|---|---|---|
| all | 200 | 6.73 | 8.32 |
| running text | 174 | 5.73 | 5.24 |
| table or ledger | 26 | 14.08 | 30.82 |
| opening, both pages written | 181 | 6.54 | 8.25 |
| opening, one page empty | 10 | 16.79 | 16.70 |
| single page | 9 | 4.40 | 4.30 |
| left page | 93 | 5.47 | 5.51 |
| right page | 97 | 8.03 | 11.50 |

- **Tables.** The gap to running text is 8 points for Nemotron Parse (bootstrap 95 % interval 2 to 14). Within tables
  the right-hand pages read at 21 % and the left at 5 % (12 and 10 pages): too few to say why; to look at.
- **LightOnOCR-3 reads running text a little better** (0.5 points, interval 0.04 to 0.96) and tables twice as badly.
- **The errors are concentrated:** ten pages carry 44 % of Nemotron Parse's edits.
- **Two test pages have a partial truth** (looked at): `e6e45ddcae8c_right` holds a full page and its truth is the last
  line (51 characters); `4c7c6e97ab16_left` holds a full page and its truth starts at the session heading low on the
  page. Both readers read the whole page, 4 % and 8 % apart from each other, 1,225 to 1,273 edits from the truth.
  Without the two: Nemotron Parse 5.45 %, LightOnOCR-3 7.07 %. The sign is cheap and is consensus entropy's (E28):
  two readers agree with each other and both read far more than the truth holds. To run over the whole ground
  truth, where the same pages would teach a reader to stop early.
- "One page empty" (10 pages, 17 % for both readers) is the next to look at: a cut or a truth problem, not known yet.
- Unlimited-OCR 2's stored reading scores 63.7 % as before and is left out.

Not yet done by eye: the table pages and the one-page-empty pages.

## Folds marked by hand: the yardstick (E31, 16 Nov)

Gabriel marked 200 scans in 17 minutes (`prototypes/mark_folds.py`; marks in `runs/htr/curate/fold_marks.lance`, the
sample in `fold_marks_sample.lance`): 40 from each of four hard kinds, 20 single pages the network would cut, 20 plain
openings, each kind a third Svea, andra arkiv and ground truth. 167 got a fold, 27 "must not be cut" (one table or
text across the fold), 6 "single page". The sample is of hard kinds, not of the pools: the shares below are for
choosing a cutter and for finding what breaks it, not rates for the pools.

Distance from the marked fold, in percent of the scan's width, on the 167 scans with a fold:

| Cutter | Median | Off by more than 0.5 % | more than 1 % | more than 2 % |
|---|---|---|---|---|
| Gutter network | 1.48 | 69 % | 60 % | 41 % |
| The middle of the scan (0.5) | 0.65 | 59 % | 37 % | 12 % |
| Placer, ±3 % around the network (as built) | 0.39 | 43 % | 35 % | 30 % |
| Placer, ±6 % around the middle | 0.26 | 23 % | 8 % | 5 % |
| Placer, ±10 % around the network | 0.23 | 24 % | 10 % | 6 % |
| VL model's `fold_x` (115 scans) | 0.49 | | 25 % | |

- **The gutter network places the cut worse than the middle of the scan does** on these scans. The marked folds lie
  between 0.48 and 0.52 of the width in nine scans of ten; the network's answers between 0.42 and 0.55. Its earlier
  0.44 % median was against the labels' gap, on scans where the labels show one. Looked at: the worst misses are the
  network's, the marks sit on the fold.
- **The placer's band was too narrow.** With ±3 % around a network answer that is itself off by more, the fold lay
  outside the band in 62 scans. Searching ±6 % around the middle, without the network, finds a fold line in 145 of
  167 scans; **where it finds one, 2 % are off by more than 1 %** (median 0.22 %, about the precision of a mark);
  where it finds none (22 scans), half are off by more than 1 %. By kind, off by more than 1 % where found: cutters
  disagree 7.5 % (40 scans), writing at the fold 3 % (33), the other kinds 0 %. By pool 2 to 4 %.
- **So a cut can be trusted where the placer, searching around the middle, finds a fold line.** That is the rule for
  the first curated training set; the rest wait for the trained placer.
- **Whether to cut at all is the open half.** Of the 33 scans not to be cut the network says "gutter" for 32. The VL
  model calls all 6 single pages single; of the 27 not to be cut it calls 30 % single and 52 % a table across the
  fold; but of its 40 "table across the fold" scans 26 can be cut. Next: the marked scans as reference examples in
  the VL model's question (cut / must not be cut), and the same decision spread by nearest neighbours in the
  semantic embedding, both scored on these 200 leaving each scan out in turn.
- Scored rows: `runs/htr/curate/fold_marks_scored.json`, `fold_marks_placer.json` (to move into `scans.lance`).


### The same on 150 scans drawn at random (E33, 16 Nov)

Gabriel marked 150 more, drawn at random from the described scans, 50 from each pool and the ground truth
(`mark_folds.py random`; `fold_marks_sample_random.lance`): 142 with a fold, 4 not to be cut, 4 single pages. These
rates hold for described scans as they come; the 200 above were hard kinds.

| Cutter, on the 142 with a fold | Median | Off by more than 0.5 % | more than 1 % | more than 2 % |
|---|---|---|---|---|
| Gutter network | 0.72 | 61 % | 36 % | 20 % |
| The middle of the scan | 0.49 | 49 % | 23 % | 6 % |
| Placer, ±6 % around the middle | 0.25 | 16 % | 3 % | 0 % |
| the same, where it finds a fold line (133) | 0.23 | 12 % | 1 % (1 scan) | 0 % |

Whether to cut, on all 150 (8 not to be cut):

| Rule for "do not cut" | Found, of 8 | Openings held back, of 142 |
|---|---|---|
| First form (single page, or table across the fold) | 4 | 1 |
| The three answers defined (E32's question) | 7 | 6 |
| Either | 7 | 7 |
| Either, or the placer finds no fold line | 7 | 15 |

- **The rule for a trusted cut:** the VL model answers "cut" to the defined question, the first form does not say
  single page or table across the fold, and the placer finds a fold line. It keeps 128 of the 150 scans; of those,
  1 was not to be cut (a Svea register across the fold the model called "cut") and 1 has its cut more than 1 % off:
  2 of 128 wrong, which on this sample puts the true share under about 5.5 % with 95 % confidence.
- The gutter network is off by more than 1 % on a third of ordinary openings too; it is kept only for finding
  candidate openings, not for placing or deciding.
- Scans not to be cut are about 5 % of described scans (8 of 150).
- Rows: `runs/htr/curate/fold_marks_random_scored.json`, `vl/cut_or_not_random.jsonl`.

## Marks in bulk, and a head on the embeddings that spreads them (E34 to E36, 16 Nov)

Gabriel marked every one of the 14,985 scans on the embedding map in the gallery (`mark_gallery.py`, cluster by
cluster, select then answer; `cut_marks.lance`): cut 13,454, odd layout 428, do not cut 419, single page 380, not a
page 304. He also marked 962 by region on the map page (`region_marks.lance`). The gallery's 14,985 are every scan
of a random 15,000 of the 45,000 first described, so their shares hold for those: **10 % are not to be cut.**

- **The first form is wrong on most of them.** Of the 1,531 scans not marked "cut" it calls 889 an opening to cut:
  335 of the 419 "do not cut", 404 of the 428 odd layouts, 114 of the 304 that are not pages. It finds 42 % of the
  scans not to be cut. The 150-scan random sample (E33, 8 such scans) was too small to show this.
- **A head on the embeddings does it** (logistic regression on Qwen3-VL-Embedding 8B's vector, five answers). With
  whole volumes held out, on the first 7,418 marks: 94.1 % right against the first form's 85.5 %; it finds 95.4 % of
  the scans not to be cut and holds back 4.8 % of those to cut. Asking for more certainty (P(cut) under 0.8 counts as
  not to cut) finds 98.3 % and holds back 13.8 %.
- **Checked on marks made one scan at a time**, since the gallery was marked a cluster at a time and its marks
  follow the clusters the head also sees: trained on the gallery marks, tested on the 350 scans of the fold page
  (none in the training): it finds 39 of the 41 not to be cut and holds back 21 of 309; the first form 32 and 33.
  On the 150 drawn at random among them: 6 of 8 found, 10 of 142 held back.
- **So bulk marking and the embeddings together answer "cut or not" for the pool:** the head runs on any scan with a
  vector. Every described scan is being embedded (`embed_images.py`, GPU 2, 6 scans a second; vectors in
  `scan_embeddings.lance`, 4,096 float16, by `embeddings_to_lance.py`).
- The trusted-cut rule of E33 changes: the head's answer takes the place of the first form's, beside the VL
  model's defined question and the placer's fold line.

**16 Nov, 12:00: the root disk filled (21 MB free) and every job stopped.** Not this work: two Docker volumes of the
rask setup hold 923 GB and 432 GB. Each output file lost one half-written last line, removed; the jobs were
restarted at 12:20 and the disk had 938 GB free by 13:20. Follow-ons queued on a log line must match what a
restarted run writes: E24's did not and was started by hand.

## Experiment register

Every experiment has a row here, added the day it starts and closed the day its result is known. The result's detail
is in the section named; scripts are under `runs/htr/curate/prototypes/` unless a path says otherwise.

| # | Experiment | State | Result in one line | Script; section |
|---|---|---|---|---|
| E1 | Old labels as a page reader, on the test pages | done 14 Nov | 10.7 % CER on 207 pages | "What the old labels are worth" |
| E2 | Calibration: old label against truth on scans with both | done 14 Nov | Svea 259 scans 7.4 % CER, andra arkiv 3,392 scans 4.8 %; confidence ranks error (-0.60 to -0.76), blind to unlabelled writing | `diffu-curate calibrate`; same section |
| E3 | Pretrain on unfiltered Svea pool pages, then fine-tune (control) | done 15 Nov | 7.18 % against 7.55 %, gain only on the pools' own archives | `runs/htr/svea_pilot.sh`; "The control experiment's answer" |
| E4 | Two more epochs on ground truth | done | 6.73 % CER, the best reader | `configs/nemotron_parse_more.yaml`; HTR_RUNS.md |
| E5 | Small gutter network from the image | done, **found wanting 15 Nov** | finds openings; places the cut to 1 % of the width, inside a label box in 2.2 % of scans | `diffu_curate.gutter`; "The image cut looked at before the re-cut" |
| E6 | Orientation network | done 15 Nov | 99.94 % in-domain; typed pages fool it | `orientation.py`; "What works" |
| E7 | DINOv3 page embeddings + EVoC on 3,007 pages | done 15 Nov | kinds of page come out as their own clusters | `page_embed.py`; "EVoC on DINOv3 page embeddings" |
| E8 | Which embedder groups pages (TIPSv2, DINOv3, EmbeddingGemma 2) | done 15 Nov | TIPSv2 0.726 same volume, DINOv3 0.653, EmbeddingGemma 2 0.42-0.43 | `embed_pages.py`, `embed_gemma.py`, `compare_embedders.py`; "Which embedder" |
| E9 | The same for Qwen3-VL-Embedding 2B and 8B | done 16 Nov | 2B: 0.381 same volume, 0.531 same collection (0.430, 0.647 with the label's text); 8B: 0.514, 0.676 (0.454, 0.677 with text): the 8B the best of the image-and-text embedders, still behind DINOv3 (0.653, 0.799) and TIPSv2 (0.726, 0.863) | `embed_qwen.py`, `compare_neighbours.py`; "Which embedder" |
| E10 | Writing layer tuned on PAGE polygons (embedder, head, resolution) | done 15 Nov | all within 0.01 IoU (0.71-0.73 of a possible 0.84): DINOv3, linear | `writing_tune.py`; "The writing layer, tuned and checked" |
| E11 | Does the writing layer notice missing labels | done 15 Nov | AUC 0.95 with a quarter of the lines removed, 0.88 with a tenth; real misses untested | `writing_eval.py`; same section |
| E12 | Our reader against the old labels on pool pages | paused 17 Nov for E38: Svea read, andra arkiv at 24,325 of 56,130 (resumable) | median disagreement 12 % and 11 %; the pages are cut the old way, so part of it is the cut. Readings of 64,950 pages in `pages.lance`; only 1,881 of the 139,937 trusted pages have one (median 10.6 %, 5.7 % disagree by more than half): the read is to be redone on the trusted pages | `runs/htr/verify/pool_read.py`; "Pages where our reader and the old label disagree" |
| E13 | The image cut looked at by group, at thumbnail and full size | done 15 Nov | one bug fixed (unsure scans kept whole); the cut clips writing | scratch scripts; "The image cut looked at before the re-cut" |
| E14 | Vision-language model describing scans: pilot, 268 scans | done 15 Nov | kind and layout follow the groups, 11 of 12 looked at right; fold position a guess | `vl_describe.py`; "The vision-language model describing scans" |
| E15 | The same on 45,000 scans | done 15 Nov 23:05 | 44,932 answered; not a page 0.4 %, single page 3.5 %, opening with a page or both empty 18 %, both written 78 %; non-handwriting 1.7 % | `vl_describe.py`; "The vision-language model describing scans" |
| E16 | A head on the embeddings giving the model's answers, class by class | done 15 Nov on 44,932 scans | handwriting against everything else: finds 98.8 % of the other scans, 74 % of what it flags is other; single page: recall 0.91, precision 0.81; both pages written: precision 0.993, recall 0.93 | `scan_head.py`; "A head on the embeddings, class by class" |
| E17 | AnyUp, INSID3 segment-by-example, ink mask, NeMo Curator | done, **do not work** | 0.803 against 0.855 bilinear; 0.72 against 0.85 | "What does not work" |
| E18 | Placing the cut at full resolution near the gutter network's answer, from the image alone | done 15 Nov (prototype) | the column dark down its whole height (the fold) beats the network's place in 14 of 40 scans judged blind, loses in 2, ties in 24; a fold is found in 68 % of scans; "the column with the least writing" is worse than the network | `cut_place.py`; "Placing the cut on the fold" |
| E19 | Does a sharper question make the VL model tell tables from running text | done 15 Nov | yes: with each option defined, 33-35 of 35 scans judged by eye (alone, with thinking, or in a two-question form), against about half with the first form's bare option names; thinking adds nothing | `vl_table_ask.py`; "What the model's answers are worth, kind by kind" |
| E20 | A second pass over the 45,000 scans asking table or running text, and turned writing, with definitions | done 16 Nov 02:30 | tables 11 % of scans (Svea 17 %, GT 11 %, andra arkiv 5 %); of the first form's 7,911 "tables" 4,554 are tables; some turned writing 1.2 %, most turned 0.0 % | `vl_ask2.py` -> `runs/htr/curate/vl/sample_pass2.jsonl` |
| E21 | Which image-and-text embedder gives the best clusters of scans: Nemotron 3.5 Embed 8B (trained on document pages) against Qwen3-VL-Embedding 8B and EmbeddingGemma 2, each clustered with EVoC and judged against the VL model's answers; a map of each | done 16 Nov 11:35 | **Qwen3-VL-Embedding 8B is the best of the three; Nemotron Embed 8B the weakest.** Neighbours of the same kind of scan, classes balanced, chance 0.20: Qwen 8B 0.75, EmbeddingGemma 2 0.66, Nemotron Embed 0.60. All three alike on running text and tables (0.78-0.93); Nemotron Embed fails on single pages (0.14, Qwen 0.61) and half-empty openings (0.37, Qwen 0.59). Best EVoC agreement with the kinds (AMI): 0.31, 0.27, 0.29. Numbers only: the clusters are not yet looked at by eye
| E22 | The VL model's two passes over the remaining 25,632 ground-truth scans, so every GT page (the 200 test pages among them) has its kind and readers can be scored by kind of page | done 16 Nov | **tables are the hard kind; two test pages have a truth that covers only part of the page.** Nemotron Parse (6.73 % overall) reads running text at 5.7 % (174 pages) and tables at 14.1 % (26 pages; difference 8 points, 95 % interval 2 to 14). LightOnOCR-3 is ahead on running text (5.2 %, by 0.5 points, interval 0.04 to 0.96) and far behind on tables (30.8 %). Right-hand pages of tables read at 21 %, left at 5 % (12 and 10 pages). Ten pages carry 44 % of all errors; the two worst are not reading errors: the truth transcribes one line, or the lower part, of a full page that both readers read whole and alike (4-8 % apart). Without those two pages Nemotron reads at 5.45 %, LightOnOCR-3 at 7.07 %. See "Readers by kind of page"
| E23 | EVoC tuned, not run at its defaults: noise level (0, 0.25, 0.5), neighbours (15, 40) and smallest cluster (5, 25) varied for each one-vector embedder, every layer scored on all scans with the unclustered counted against it | done 16 Nov 11:33 for all four embedders | so far: no setting brings the unclustered share under about 40 % (noise_level 0: 48 % and 45 %), and the agreement with the kinds of scan hardly moves (0.18-0.19; 0.23-0.24): EVoC's settings are not the lever; the unclustered scans need assigning to their nearest cluster, or another clusterer | `evoc_tune.py` -> `embed_scans/evoc_tune_<name>.json` |
| E24 | The VL model's two passes on 60,000 more pool scans (30,000 from each pool, none described before), for more examples of the rare kinds and a better head | done 17 Nov (both passes; the second ended overnight) | 130,666 scans described in all, 100,083 of them pool scans; with them the trusted set doubled (77,258 pool scans pass the rule, E37)
| E25 | Full coverage for the clusters (SCALAR), and oddities (HiRank), on Qwen3-VL-Embedding 8B's vectors of 14,985 scans | first run 16 Nov 10:00; **to repeat on Nemotron Embed and to tune** | EVoC alone leaves 54 % unclustered; SCALAR (4,000 k-means representatives, EVoC on them, labels spread back by 5 neighbours) leaves 7 % with the same agreement with the kinds of scan (0.237 against 0.232); voting EVoC's unclustered scans in covers all but swells one cluster to 14 %. HiRank: the oddest scans are not the non-handwriting (signs and reel targets are many alike, so dense: AUC 0.18); 1,465 scans tie at the lowest score, the 18 looked at mostly ordinary openings with a few one-offs (a letter with a drawing, a fold-out) | `scalar_hirank.py`; "SCALAR and HiRank" |
| E26 | A trained cut placer: the full-resolution placer's confident folds as labels (20,000 scans of both pools and the GT), a model on a high-resolution strip around the fold learning to place the cut where no dark line shows; scored against folds marked by hand, by kind of scan | labels collected from 19,660 scans; 350 folds marked by hand (E31, E33) and 225 more scans drawn for marking, unmarked; **not trained**: the placer around the middle covers about nine openings in ten, so this is for the rest | | `cut_place.py` -> `runs/htr/curate/cut/placed_<pool>.jsonl` |
| E27 | Can the VL model judge or place the cut when shown a zoomed strip | done 16 Nov | **as a judge between two drawn cuts: agrees with the eye on 14 of its 16 decided scans (three runs; with thinking 14, one opposite, one unanswered), not by colour; it never says "no difference"**. As a locator against a ruler: median error 0.6 % of the width (1.5 % with thinking), no better than the gutter network (0.44 %). So: the judge, without thinking | `vl_fold.py`; next: the judge as the yardstick for cut placers on a few thousand scans, 100 of its calls checked by eye |
| E28 | Consensus entropy (Zhang et al., CVPR 2026, arXiv 2504.11101; `consensus-entropy` on PyPI, MIT): each page read by several readers, a reading scored by its mean normalised edit distance to the others; the page's lowest score says how far the readers agree, the reading with it is the label to keep. Ours: Nemotron Parse, LightOnOCR-3 and the old label; on the calibration scans first, where the truth says whether low entropy means low error and whether the chosen reading beats each reader | **queued 16 Nov** (needs LightOnOCR-3's readings of the calibration scans: a GPU job) | | to write |
| E29 | NeMo Data Designer (NVIDIA-NeMo/DataDesigner, Apache-2.0) as the harness for the VL model's judging and describing: seed rows with images, `LLMJudgeColumnConfig` with named scores whose every option is described, an OpenAI-compatible provider (our server), parallel requests, traces. Pilot: the cut judge of E27 rebuilt in it, the same 40 scans, the same answers expected; whether its judge column forces a JSON format, which broke this model's answers once | **queued 16 Nov** | | to write |
| E30 | One Lance table of scans with every feature as a column, built from the Lance docs read at their source | **first columns in 16 Nov**; pool reads, embeddings, clusters and scores to move; run scripts to write Lance | `scans.lance`: 3,184,889 scans (pools and GT) keyed by `image`; gutters, the VL model's answers (typed and as JSON) and fold placements joined; a refresh of 79,000 described scans takes 12 s; none differs from the result files | `diffu_curate.table.add_features`, `prototypes/to_lance.py`; "One table" |
| E31 | Folds marked by hand on 200 scans of hard kinds, every cutter scored against them | done 16 Nov (Gabriel's marks, 17 minutes) | **the gutter network is worse than cutting at the middle (60 % against 37 % off by more than 1 % of the width); the placer searching ±6 % around the middle finds a fold line in 87 % of scans and is then off by more than 1 % in 2 %**; whether to cut at all is not solved: the network says gutter for 32 of the 33 scans not to be cut | `mark_folds.py`; "Folds marked by hand" |
| E32 | Does the VL model decide better whether a scan is to be cut (cut / must not be cut / single page) when shown marked scans as examples; scored on the 200 marked scans, each left out of its own examples | done 16 Nov | **marked examples do not help the VL model overall (86 % right with definitions only, 88 % with six examples; 13 scans gained, 9 lost); they make it more cautious about saying "do not cut".** Scans not to be cut found, and openings wrongly held back, of 33 and 167: the first form's columns 28 and 32; the three answers defined 28 and 16; with examples 22 and 6; definitions or the first form 33 and 41. No rule finds all 33 without holding back a quarter of the hard openings. A missed one, looked at: a register whose rows run across both pages. The sample is of hard kinds, so the rates are not the pools'
| E33 | The cutters and the cut-or-not rules on 150 scans drawn at random and marked by hand, for rates that hold for the pools | done 16 Nov (Gabriel's marks) | placer ±6 % around the middle: 3 % of folds off by more than 1 %, 1 of 133 where a fold line is found; gutter network 36 %. **Trusted-cut rule (VL "cut" to the defined question, no objection from the first form, a fold line found) keeps 128 of 150 scans, 2 of them wrong.** Scans not to be cut are about 5 % | `mark_folds.py random`; "The same on 150 scans drawn at random" |
| E34 | Marking cut-or-not in bulk in a gallery: 1,129 scans, twenty a screen, each starting at the first form's guess, only the wrong ones clicked | done 16 Nov: all 14,985 scans of the map marked | cut 13,454, odd layout 428, do not cut 419, single page 380, not a page 304; 10 % of scans are not to be cut, and the first form calls 58 % of those an opening to cut
| E35 | Marking by region of the embedding map (after marimo's bulk-labelling example): the 14,985 scans with a Qwen 8B vector on their UMAP map, a box dragged over a region, twenty of its scans shown, one button marking the box | done 16 Nov: 962 scans marked by region; the gallery by cluster proved the quicker page | the clusters do separate the question: 4 of the 50 clusters hold 68 % of the scans the first form calls single or a table across the fold; 36 clusters (11,749 scans) hold under 2 % of them | `mark_map.py`, `scan_map.lance` -> `region_marks.lance` |
| E36 | A head on Qwen 8B's embeddings trained on the bulk marks, to answer cut-or-not for every scan | done 16 Nov; every described scan embedded 17 Nov (130,733 vectors in `scan_embeddings.lance`) | volumes held out: 94.1 % right (first form 85.5 %), 95.4 % of the scans not to be cut found, 4.8 % of the others held back. On 350 scans marked one at a time, outside the training: 39 of 41 found, 21 of 309 held back | "Marks in bulk"; `embed_images.py`, `embeddings_to_lance.py` -> `scan_embeddings.lance` |
| E37 | The rule for a trusted cut applied: first form says opening and no table across the fold, defined question says cut (`vl_cut`), head on all 15,288 marks says cut (`head_cut`, `head_p_cut`), placer finds a fold line | done 17 Nov on all 100,083 described pool scans | head on all marks, volumes held out: 95.4 % right, 95.2 % of not-to-cut found, 3.7 % of cut held back. Rule keeps 37,222 of 48,142 scans (77 %): Svea 18,379, andra arkiv 18,843; running text 34,803, tables 2,330. 48 drawn at random and looked at with the fold drawn: 47 cut on the fold, 1 a little to the side. **All described pool scans, check by check:** 100,083 -> opening with no table across the fold 95,485 -> defined question says cut 87,366 -> head says cut 80,767 -> fold line found 77,495 -> labelled lines 77,258 -> outside ground-truth volumes 75,776 | `cut_head.py`; `cut_or_not.lance` -> `vl_cut`; `folds.lance` (123,838 openings) |
| E38 | Nemotron 3.5 Parse pretrained on the trusted cuts of both pools (old labels, 1 epoch, the control's recipe), then fine-tuned on the GT as `nemotron-parse-all` (2 epochs; compare 7.55 % and the control's 7.18 %) and 2 more as `nemotron-parse-more` (compare 6.73 %) | **first answer 18 Nov 08:14**; the 2 more GT epochs (`nemotron-parse-trusted-more`) stopped at step 316 on 18 Nov 09:13 for E40's vectors (Gabriel's choice), resuming by itself from checkpoint 250 when they end | pretraining on 139,937 trusted pages, then 2 GT epochs: **6.92 % CER, 18.61 % WER** against 7.55 %, 20.50 % without pool pages (gain 0.63 points, 95 % interval +0.14 to +1.16, better on 123 pages, worse on 58), 7.18 % after the unfiltered Svea pretraining (0.26 points, not shown) and 6.73 % for 4 GT epochs alone (not ahead). The gain is largest on pages of the pools' own archives. Section "The trusted-cut pretraining's first answer" | `stage_trusted.py`; `diffu-reader-set --cuts` (a scan the file names is cut at its fold); configs `nemotron_parse_trusted_{pretrain,finetune,more}.yaml` |
| E39 | Does a head on the patch models' whole-scan vectors (TIPSv2, DINOv3 at 448 px, already held for the 45,000 first described scans) answer cut-or-not as well as the head on Qwen 8B's, so that every pool scan can be embedded by a small model and the VL model left out at scale | done 17 Nov | on 15,110 marked scans with both kinds of vector, volumes held out: Qwen 8B 95.5 % right, 95.3 % of the scans not to be cut found, 3.9 % of the others held back; TIPSv2 + DINOv3 94.6 %, 96.2 %, 5.0 %; DINOv3 alone 94.0 %, 96.0 %, 5.6 %; TIPSv2 alone 93.1 %, 96.9 %, 6.8 %. Of the scans each head would cut, 0.4 to 0.6 % are not to be cut. **The patch models are enough for cut-or-not.** Both together embedded the 45,000 scans at 34.6 a second on one GPU (Qwen 8B: 10.5), so the 3.15 million pool scans take about 13 hours on two GPUs. The described pool scans are 3 % of the pools and the rule keeps 77 % of them: by it the pools hold about 2.4 million scans to cut, 4.4 million pages; one epoch of pretraining covers about 280,000 pages a day, which sets the size | `cut_head_embedders.py` |
| E40 | The whole pools through the cut rule without the VL model: every pool scan with labelled lines embedded by DINOv3 and TIPSv2 (`embed_pool.py`: as the 45,000 were, in shuffled order, scans of one shape batched), every candidate opening's fold placed, then the head on the hand marks and the placer's fold line decide | **running since 18 Nov**: the folds on the CPU since 09:15 (tmux `placepool`, about 01:00), the vectors on GPUs 1 and 2 since 09:17 (tmux `embedpool0`, `embedpool1`, about 18:30); the columns by `pool_columns.py` (heads fitted on the 45,000 first described scans; `apply` when the vectors are in) | checked on the CPU on 40 scans of the 45,000: the new vectors match the stored ones (cosine at least 0.9995 for DINOv3, 0.9999 for TIPSv2). Expected: about 13 hours on two GPUs, about 2.4 to 2.5 million scans to cut. To do when it has run: vectors into Lance, the head fitted on the marks, a sample of its cuts looked at by eye, then a set of the size chosen. **First results, 18 Nov 09:40.** The run embeds 47 scans a second on each GPU (the loaders, 20 a shard, are the limit, not the GPUs). Its first two files (40,000 scans): 541 of them are among the 45,000 embedded before and reproduce their vectors (cosine at least 0.9998), no NaN, 82 % greyscale. The heads (`pool_columns.py fit`: cut class from 15,110 marks, table from 44,269 answers, kind, layout) applied to them and joined (`scans.lance` tagged `before-pool-columns`; the column first called `table` renamed `is_table`, a reserved word in filters). **On 742 scans the VL model described and the heads never saw:** table agrees on 97.8 % (62 of the VL model's 63 tables found; 80.5 % of the head's tables are the VL model's), layout on 94.1 %, kind on 87.1 %; the cut class agrees with the head on Qwen's vectors on 96.5 %. **By eye, 8 scans of each cut class among 38,717 that no model described and nobody marked: all 40 as named** (openings; registers across the fold; single sheets; reel targets, cards and blanks; letters with seals and loose slips). Shares on those 38,717: cut 87.8 %, odd 6.0 %, single 2.6 %, do not cut 2.4 %, not a page 1.2 %; tables 12 %. **A lesson on speed, 18 Nov 10:40.** The machine's CPU was the limit (idle 15 %, the GPUs busy two thirds of the time), and each shard's main process held about 8 cores (880 % of one) though it only stacks batches: torch's threads, idle between batches. With `OMP_NUM_THREADS=1`, measured over the same three minutes with one shard each way: **70.6 against 47.1 scans a second**, the main process at 76 % of a core. Both shards restarted so (after a saved part each, nothing embedded twice but the scans in flight), and the placer rose from 35 to 44 scans a second on the freed cores. The setting is the script's default on a GPU now, and a part is written under another name and renamed, so a run stopped while writing leaves no part a later run would trip on. **The larger set as a filter, 18 Nov 11:45.** `stage_pool.py` takes the scans by the columns (lines labelled by the old pipeline, `cut_class = 'cut'`, `is_table = false`, a fold line found, outside the ground truth's volumes; `SINGLES=1` adds the scans the head calls a single page, uncut) and stages them for the set builder as `stage_trusted.py` does; `count` prints the funnel and the tables, the other filter. **Tested on scans no model described and nobody marked:** 300 staged, 554 pages built (1.847 a scan, none wider than tall). By eye, each scan with its fold line drawn: 16 at random and the 16 the head was least sure of (`cut_p` 0.50 to 0.77), all 32 openings with the line at the fold (the least sure are openings with one page empty); 24 page crops, all single pages. **Projected from the run's first 40,000 scans, a random sample of the pools:** 81.3 % are an opening to cut with running text, a fold line is found on 95.9 % of the 4,451 with a fold placed, 97.9 % of those lie outside the truth's volumes: 76.3 % taken, **about 2.34 million scans and 4.33 million pages**; single pages 2.3 % (about 71,000); tables 10.6 % (about 324,000 scans; andra arkiv 6.0 %, Svea 17.7 %); colour scans 17.5 %. The exact funnel is written to `runs/htr/curate/pool_funnel.txt` when the columns and the folds are in (tmux `poolcount`). **A second lesson on speed, 18 Nov 11:30: the column step.** Run on the 680,000 scans embedded by then, `apply` took 11 min 22 s, 10 min 37 s of it in the kernel: with the machine's memory full of file cache, numpy's and pyarrow's large fresh arrays stall there on huge pages (one 154 MB part read in 61 s with the defaults, in 2 to 4 s with `NUMPY_MADVISE_HUGEPAGE=0` or pyarrow's system memory pool; the placer and the loaders lose 15 % and 31 % to the kernel, no such stall). `apply` now sets both, takes the vectors as they lie in the part into one array kept for the whole run, and prints its times: 43 parts (860,000 scans) in 111 s, the join of 894,169 rows in 16 s (ten held columns updated in place, 3 fragments), about 8 minutes at 154 parts; `cut_class` is identical to the first version's on all 716,392 scans compared. `marks_to_lance.py` and `stage_pool.py` hold the same guard and were run once as the queue will run them (23 s, 27 s). **`pca`, 18 Nov 11:37:** the first 64 principal components of the two models' normalised vectors (fitted on the run's first 200,000 scans; 2, 8, 16, 32 and 64 components hold 27, 54, 65, 74 and 82 % of the variance), a 16-bit vector column of the scan table (the helper's join of such a column tested on a throwaway table, new and held). Of a scan's 5 nearest by `pca` among 860,000, 44.1 % lie in its own volume (chance 0.02 %) and 98.2 % share table or not; a search takes 0.27 s without an index | `embed_pool.py`, `place_folds.py ... pool` |
| E41 | Is the VL model needed on every scan? On the 15,110 marked scans with patch vectors: its answers against the head for cut-or-not; and a head on the patch vectors fitted on its defined table-or-running-text answer | done 17 Nov | **cut-or-not: the head on the marks is far ahead of the VL model's answers.** Scans not to be cut found, openings held back, share of the kept that are not to be cut: VL first form 45.6 %, 0.4 %, 5.9 %; first form and defined question 64.8 %, 6.5 %, 4.1 %; head on patch vectors (volumes held out) 96.1 %, 5.0 %, 0.47 %; head and the VL model's two answers 97.8 %, 10.3 %, 0.29 %. Of the 61 scans not to be cut that the head would cut the VL model's answers stop 26, and they hold back 721 of the 12,873 openings the head keeps. **Table or running text: a head repeats the VL model's defined answer** on 44,269 scans, volumes held out: 97.3 % agree, 96.2 % of its tables found, 82.9 % of what the head calls a table the VL model does too. So the VL model describes a sample and the heads carry its answers and the marks to every scan | `vl_or_head.py` |
| E42 | Table pages: what the ground truth holds for them, where the reader's errors on them are, and what the old labels are worth on them (asked 17 Nov: are tables good data to train on at all) | done 17 Nov | **The ground truth marks almost no tables as tables:** a table block (rows and cells) on 201 of 49,491 training pages, 20 of 974 dev, 5 of 2,971 test; none of the 26 test pages the VL model calls a table has one: they are transcribed as lines. Those 26 carry 3,278 of the reader's 13,004 edits (25 %). Four looked at with both texts: an inventory with dot leaders and amount columns (20 %), a ruled register of livestock (16 %), a list of goods without a grid (7 %), and one that is no table but a draft with struck-out words whose truth is in a scrambled line order (57 %, not the reader's fault). **The old labels are much worse on tables, Svea's most of all** (calibration scans, by the VL model's kind; error on matched lines, true lines missed, old lines matching no true line, lines with a number wrong): Svea running text 6.3 %, 2.6 %, 1.2 %, 1.7 % (239 scans); Svea tables 18.7 %, 4.2 %, 35.1 %, 45.9 % (20 scans); andra arkiv running text 4.8 %, 1.4 %, 3.1 %, 1.6 % (3,260); andra arkiv tables 6.9 %, 8.4 %, 12.3 %, 6.1 % (103). The pool set holds no table block at all (225,367 text blocks). So pool tables teach lines in the old pipeline's order with many wrong numbers: no extra weight for them; better tables come from ground truth, a fixed reading order, or the double pass | queries on `data/htr/nemotron_parse_pages.*.jsonl`, `e22_rows.json`, `calibration*/{pages,lines}.lance` |
| E43 | The old pipeline as one reader on the same 200 test pages (asked 18 Nov): YOLO regions and lines, then TrOCR, as htrflow runs them, on the 200 seeded `test_0050` pages (the page images the readers are given) and scored on whole-page CER and WER as Nemotron Parse is. E1's 10.7 % was on 207 other pages (11 shared) and is not a comparison | **done 18 Nov 12:50** | Whole page, the same 200 pages, lines top-down: **YOLO + TrOCR swe-2 (the 2024 run's public recogniser) 16.55 % CER, 33.31 % WER; YOLO + the Swedish Lion 2026 (large, character) 14.47 %, 27.26 %; Nemotron Parse (4 GT epochs) 6.73 %, 18.83 %** (better on 138 pages, worse on 57 against the Lion pipeline; gain 7.74 points, 95 % 5.26 to 10.58). Running text (174 pages) 10.77, 8.81, 5.73 %; tables (26) 58.71, 55.82, 14.08 %. Of the swe-2 pipeline's 16.5 %: 8.3 missing, 4.3 extra, 4.0 wrong characters; on tables 46.7 of 58.7 is missing; the median page is at 6.9 %, the 20 worst pages hold 52 % of the edits (13 tables). Order is not the cause: the lines as exported, regions top-down, and all lines top-down give 16.5 % alike; htrflow's `ReadingOrderMarginalia` gives 23.2 % (21.3 % with the Lion). The label files name htrflow 0.1.3, `yolov9-regions-1` and `yolov9-lines-within-regions-1` with their revisions, the recogniser only as `./model`; run with htrflow 0.2.6 in an environment of its own (`transformers<5`: 5.23 cannot run the TrOCR decoder). TrOCR chose its checkpoint on these pages. Not the MiniCPM comparison: that was 5,000 given lines (TrOCR 3.58 % corpus CER) | `runs/htr/verify/old_pipeline_test_read.py`, `runs/htr/old_pipeline_test.sh`; `runs/htr/test_read_old_pipeline/` |
| E44 | Why the trusted-cut pretraining helps so little (asked 18 Nov): the GT dev loss of the three fine-tunes step by step, and the gain on the test pages by scan colour and by likeness to the pool scans | done 18 Nov | **The pools do not look like the test pages, and the GT catches up.** Greyscale test pages (49; the pools are 72 % and 88 % greyscale microfilm in a 600-scan sample) go from 6.99 to 5.78 % (17 % of their error), colour pages (151, 73 % of the characters) from 7.76 to 7.35 % (5 %). By the mean cosine of a page's ten nearest pool scans (Qwen 8B) there is no clear relation: thirds 0.32, 1.22 and 0.38 points, rank correlation 0.11. The pretraining's head start on the GT dev loss shrinks as the fine-tune goes on: 1.423 against 1.732 at step 250, 1.321 against 1.432 at 1,000, 1.134 against 1.155 at the end. 4,374 steps on the pool pages bring the GT dev loss to 1.78; 250 steps on the GT alone reach 1.73: the pool's targets are far from the GT's. Colour scans are about 580,000 of the pools' 3.15 million; `saturation` is added to E40's pass (scans are stored as RGB whether greyscale or not, so only the pixels tell) so that a set can be chosen by it. Not tested: whether colour pool pages help colour test pages more | `compare_reads.py`, `gain_by_likeness.py` |
| E45 | **NVIDIA's table model on our tables (asked 18 Nov; the first of the three to do).** Nemotron Table Structure v1 (`nvidia/nemotron-table-structure-v1`, Oct 2025: YOLOX, 54 M parameters, finds cells, rows and columns in an image of a table fitted to 1,024 px; NVIDIA Open Model License, commercial use allowed; public, 434 MB) run on our table pages by kind: ruled registers, inventories with amount columns and dot leaders, lists without a grid, tables across the fold | **first look and a first score done 18 Nov** (CPU: 26 pages in under two minutes, 228 tables in four); its uses not yet tried | read at its source: trained on 23,977 printed pages of Digital Corpora labelled by Azure Document Intelligence; AP on its own set: cell 58 %, row 77 %, column 85 %; its card says it may not generalise to unknown table formats, is not robust to turned tables, misses cells at a table's bottom, and wants the table cut out first (E46). First look: its boxes drawn on the 26 table test pages and about 100 GT and pool table scans, judged by eye by kind. If it finds rows and columns: (a) our reader's lines put in row order and scored against the truth on table pages (E42: part of the table error is reading order); (b) the GT's line transcriptions set into the cells it finds, as table targets for the reader (the GT marks 201 of 49,491 pages as tables). A small detector: the first look can run on the CPU. **First look, by eye, on the 26 table test pages, the model run on the whole page, boxes scored above 0.3** (`table_models_look.py`): without any training on our pages it lays a row on nearly every written line of the list-like pages (median 18 rows a page, up to 39) and finds believable columns on many (item and amount, name and page numbers, two lists side by side; median 2.5 columns, up to 7), with doubled column boxes at page edges on some; on dark microfilm and on a densely ruled register the rows are few (6); cells are patchy (median 5.5 a page). Two lists side by side get rows across both. Of the 26 pages about 5 are no tables at all (running text, a near-empty page): there it gives one column and a row a line. No figure yet: the test pages hold no table markup to score against. **Scored on the ground truth's own tables** (`table_model_score.py`: the 228 table regions of 139 training and dev scans with 2 or more rows and columns, each cut out and given to the model; its rows and columns paired one to one with the truth's where their spans overlap by half or more): **columns 89.3 % found, 87.5 % of those found right; rows 67.2 % found, 77.4 % right** at score 0.3; every column found in 80 % of the tables, 90 % of the rows in 21 %. The score taken changes little: rows found 75.9 % (69.1 % right) at 0.05 and 49.3 % (82.4 %) at 0.5; columns 92.8 % (76.1 %) and 83.5 % (92.3 %). By collection: name lists and registers well (handelskollegiet rows 92 %, columns 95 %; politikollegiet, 134 tables, rows 70 %, columns 96 %), tables whose rows are paragraphs of several lines badly (Sollentuna häradsrätt rows 44 %, columns 75 %). Looked at, 12 drawn: on the lists its rows lie on the truth's. All of this without training it on our pages; its card names fine-tuning (the YOLOX code) for unknown formats, and the 228 tables are labels for that | `table_models_look.py`, `table_model_score.py`; models in `/mnt/scratch/gabriel/models/nemotron-{table-structure-v1,page-elements-v3}` (run with `CUDA_VISIBLE_DEVICES=`: their loaders take cuda:0) |
| E46 | Nemotron Page Elements v3 (`nvidia/nemotron-page-elements-v3`, Oct 2025: YOLOX, 54 M parameters; table, chart, infographic, title, header or footer, text on a page fitted to 1,024 px; same licence; public, 434 MB): the step that cuts a table out for E45, and as a table finder for every scan against the VL model's answer and the head's (E41) | **first look done 18 Nov** with E45 | read at its source: trained on 36,093 pages of enterprise documents; AP for tables 45 % (recall 62 %) on its own set; its card says it may not generalise to documents unlike those on the web. **First look on the 26 table test pages: it calls 7 of them a table** (and 10 an infographic, the rest text), so it is no table finder for our pages; the head on the patch vectors finds 96 % of the VL model's tables (E41). The table model does as well on the whole page as on the table it cuts out | to write |
| E47 | Nemotron OCR v2 (`nvidia/nemotron-ocr-v2`, Apr 2026: a text detector (RegNetX-8GF, 45 M), a recogniser (36 M in the multilingual variant, lines) and a relational model for grouping and reading order; 34.7 pages a second on an A100 by its card; English, Chinese, Japanese, Korean, Russian; NVIDIA Open Model License; public): (a) its detector and reading order on our pages, against the GT's line outlines and the old YOLO's lines: a fast line finder, and a check for writing without a label on every pool page (it takes the place of task 8's first candidate; the other line finders stay unapproved); (b) its recogniser on the 200 test pages, for the record; (c) NeMo Curator's stages built on it (`NemotronOCRV2Stage`, the readers, `SkipProcessedStage`, `ResultWriterStage`: Curator issue 2121, PR 1899, June 2026), read for the pipeline as code | **queued 18 Nov** | read at its sources (model card, the blog of 17 Apr 2026, the issue): trained on about 12 million images, 680,000 of them real (handwritten pages among them) and the rest rendered; Swedish is not among its languages and neither page gives fine-tuning code, so (b) is expected to be poor on our hands; needs Linux with an NVIDIA GPU, Python 3.12 and a build of its C++ and CUDA parts. **E17's line "NeMo Curator has nothing for documents" is to be looked at again:** Curator holds an OCR pipeline since PR 1899 | to write |
| E48 | The second round of pretraining (Gabriel, 18 Nov: pretrain on synthetic and machine-labelled pages, then fine-tune): what the pretraining set holds and how the fine-tune keeps it. Three things to vary against E38, one at a time: (a) pool pages chosen from the whole pools to look like the target (colour, kinds, archives; E44) and in number; (b) pages drawn by the generator in the hands the GT is thin in, their labels exact (HTR_RUNS.md, "Next", item 4: the hand LoRAs); (c) a share of the pretraining pages kept in the GT fine-tune, against the fading E44 shows (the head start on the GT dev loss goes from 0.31 to 0.02) and Kiessling reports (arXiv 2609.20064, Sep 2026: knowledge the fine-tune data lacks fades); later (d) the pool pages' labels from our own reader where it agrees with the old label | **proposed 18 Nov**; waits for E38's second score (pretraining and 4 GT epochs against 6.73 %) and for E40's columns | no synthetic page has been used for a page reader yet; E38 is machine-labelled pages alone and gave 0.63 points. Kiessling's gain was largest where target data was scarce (a quarter of a collection with pretraining beat all of it without); the GT here is 49,491 pages, so the gain is to be looked for by collection and hand, not only overall | to write |
| E49 | **Tables as structure (Gabriel, 18 Nov: for a table the goal is image to structured HTML, rows and cells, not only its text in order).** In order: (a) a score: a held-out set of marked tables and a measure that compares tables as trees (TEDS), beside the CER of their text; (b) more table targets: the table model (E45), fine-tuned on the GT's 232 marked tables, sets the GT's line transcriptions into rows and cells on the GT pages that are tables but transcribed as lines; pages drawn by the generator with exact structure; a sample corrected by hand; (c) the reader trained to write those tables and scored on both | **decided 18 Nov, not started** | where it stands: the reader wrote a table block on 0 of the 200 test pages (best reader and E38's alike); the GT marks 232 tables on 139 scans (201 training pages, 20 dev, 5 test), 0.4 % of the training pages; a table target is LaTeX `tabular` (the reader's own form, `nemotron_parse._tabular`), which converts one to one into an HTML table; there is no table test set and no structure score. What a row is where there is no grid follows the GT's marked tables. Tables across the fold stay out (single pages only) | to write |
| E50 | Nemotron Parse as its repository runs it (asked 18 Nov; the whole of `nvidia/Nemotron-3.5-Parse` read: README, docs, serving reference, and every source file by two readers): vLLM with native Parse, the DFlash2 draft, its helpers | **first measure 18 Nov 13:40**; tonight's read of the trusted pages uses it | Our vLLM (0.33.0rc1) holds native Parse and `DFlash2DraftModel`; our fine-tuned checkpoint serves as it is (`/mnt/scratch/gabriel/htr/serve/parse-more`: links to the checkpoint and to `dflash2/`). **The text is the same**: 6.17 % (plain) and 6.20 % (draft) against 6.15 % through Transformers on the 197 test pages that do not loop; 3 pages loop to the token cap, vLLM having no repetition guard (E51). **Speed on a GPU shared with the embedding**: one page at a time 0.55 pages a second plain, 1.42 with the draft; 64 in flight 2.9; 600 trusted pages at 3.2 (Transformers, one at a time: 0.5). **The draft fits our weights badly**: 3.7 to 4.0 tokens accepted of 16 drafted (NVIDIA on its own weights: 10.7), first-position acceptance 0.78 to 0.83. FlashInfer's prebuilt kernels installed in `envs/vllm` (`vllm download-kernels`). Not yet measured on a quiet GPU, nor with NVIDIA's full serving profile at 128 or more in flight | `runs/htr/parse_serve.sh`, `runs/htr/verify/parse_vllm_read.py`, `parse_vllm_pool_read.py`; `runs/htr/nemotron_parse/test_read_vllm_{base,dflash16}` |
| E51 | The repository's repetition-stop processor in our vLLM (its file is written for the older runner: `ValueError: ... is not a subclass of vllm.v1.worker.gpu.sample.logits_processor.LogitsProcessor`) | **done another way 18 Nov 14:40**: vLLM's own repetition detector, not the port | Reducto's note (4 Sep 2026, read in full) uses vLLM's native detector (`repetition_detection`: `min_pattern_size`, `max_pattern_size`, `min_count`, a request field; our build has it) and tunes it by replaying settings over saved readings. Replayed here (`repetition_replay.py`): 14 looping readings (test pages and 700 trusted pages read without a guard) against 5,086 good pages (4,000 ground-truth targets, 1,086 readings that ended by themselves). All 14 settings catch 14 of 14 at a median of token 175 to 210; patterns from 1 token stop 6 to 31 good pages, the stricter ones 1. **Taken: 4 to 120 tokens, 6 copies.** Live with the DFlash2 draft on the 200 test pages: the 3 loops stop at tokens 144, 571 and 971 (`finish` "repetition"), none cut at the cap, 3.3 pages a second on the shared GPU. **A stopped page is a partial page** (7.87 % if scored as it is): with the 3 read again by the guarded Transformers reader, 6.76 % CER against 6.73 % for that reader alone. Not handled: loops that count (2074, 2075 ...) and very long repeated blocks, which the note catches with a detector of its own that it does not describe; the pool reader marks stopped pages (`finish`) and they are left out or read again | `runs/htr/curate/prototypes/repetition_replay.py`; `runs/htr/nemotron_parse/test_read_vllm_dflash16_stop` |
| E52 | A DFlash2 draft of our own for the fine-tuned reader (speculators' recipe, pasted 18 Nov: prepare-data on the reader's own answers, hidden states from vLLM, `speculators train --speculator-type dflash2`) | **decided 18 Nov 14:00 (Gabriel: we train our own)**; data from tonight's pool read, training after it | NVIDIA's draft reads layers 1, 4 and 8 of the decoder and shares the target's embeddings and head, all of which our fine-tune moved. Speculators calls its DFlash2 training experimental (the public objective is not available); to read its `dflash2.md` and whether it takes an image-prefix model before promising it. **Checked 18 Nov:** our speculators (0.10.0a32) has `dflash2` (its docs call the training objective experimental: the public one is not published; full verifier vocabulary required), its data pipeline takes messages with images, and vLLM's Parse decoder exposes the layer states a draft trains on (`EagleModelMixin`, `SupportsEagle3` in `nemotron_parse3_5.py`). **Not yet tried:** whether speculators loads this target's embeddings and tied head (mBART scales its embeddings), and a first end-to-end smoke. Training data must be the reader's own answers: tonight's vLLM read of the 139,937 trusted pages writes exactly those (image and raw answer). **18 Nov 15:10, the route without vLLM:** vLLM's `extract_hidden_states` mode refuses Parse's config (two text configs: `decoder`, `text_config`). Speculators' offline mode needs, a sample, one file `hs_<index>.safetensors` whose `hidden_states` is [tokens, layers, 1024] (the target layers, then the last layer; `train/data.py` `_build_sample`), beside a prepared dataset of `input_ids` and `loss_mask` (`DatasetRecord`); the loader checks the token count and finiteness. So the extractor is: our checkpoint in Transformers over (page image, its own answer), `output_hidden_states`, the decoder's layers 1, 4, 8 and the last at the text positions only (the image prefix is context, not a sample position), the loss mask on the answer. Open: whether the draft at serving also reads the image positions' states (vLLM warns that the drafter gets no multimodal embeddings) | |
| E53 | Double and triple pass (crops of Table and Chart blocks read again; a separate layout request) | **queued 18 Nov, behind E49** | they refine only blocks the first pass calls Table or Chart; our reader writes a table on 0 of 200 test pages, so today they would change nothing. NVIDIA's own score: 72.74 single, 74.92 double, 75.20 triple, with the `alternatives` merge that can return two readings of one table | |
| E54 | The repository's own postprocessing in place of our regex (`postprocess_document`: blocks with class, box, literal text; tables to HTML, CSV, hierarchical JSON with spans) | **queued 18 Nov** | to use `blocks[i]["text"]` with every presentation option off for the text score (the helpers drop an unclosed last block and text outside blocks, rewrite a list item's leading dash, and turn tables to HTML), and its HTML tables as E49's output. `block_score.py` (18 Nov) already scores the blocks: 94.1 % of 357 truth blocks found at IoU 0.5 (82.4 % at 0.75), 87.5 % of 384 read blocks are a truth block, class right on 99.1 %, order exact on 60 of the 64 pages with two or more found blocks | `runs/htr/curate/prototypes/block_score.py` |
| E55 | Publish to Hugging Face (asked 18 Nov): our Nemotron Parse reader and its DFlash2 draft, as the MiniCPM reader and its DSpark draft were (`publish_model.sh`, `Gabriel/diffu-minicpm-ocr-htr`, `-dspark`) | **queued 18 Nov** | which checkpoint (the best after E38's second score), the repository names and whether public or private are Gabriel's to say; the draft follows E52. A serving folder exists already (`/mnt/scratch/gabriel/htr/serve/parse-more`); the card must name NVIDIA's licence (OpenMDW-1.1) and the tokenizer's. **What the published repository must hold (asked 18 Nov):** a README with how to serve it in vLLM (`runs/htr/parse_serve.sh`'s command, with and without the draft), how to call it (the task prompt, temperature 0, top-k 1, repetition penalty 1.1, special tokens kept) and how to read the output (blocks, boxes, classes); **the repetition detector with its setting** (`repetition_detection`: patterns of 4 to 120 tokens, 6 copies; a stopped page is partial and is read again) and how it was tuned; **the evaluation and its numbers** (the 200 test pages: CER and WER, by kind of page, the block scores, against the old pipeline and the other readers, with what favours whom) and the files to reproduce them; **the training's TensorBoard logs**; speeds with the hardware they were measured on | |
| E56 | A Nemotron table model of its own, trained on all the table data (asked 18 Nov) | **queued 18 Nov, with E49** | **Tables and running text are kept apart**: the running-text sets take `is_table = false` only (`stage_pool.py` does), the table set is the other filter (about 324,000 pool scans projected, 232 ground-truth tables) and trains a second Nemotron Parse whose output is the table as HTML. The pools' old labels on tables are poor (E42), so its targets need E49's work first (structure score, targets from the table-structure model, a hand-corrected sample) | |
| E57 | The reader on our Blackwell GPUs (RTX PRO 6000 Blackwell; asked 18 Nov): kernels built for them, and NVFP4 or MXFP8 weights | **queued 18 Nov, behind E50's quiet-GPU speed** | Known: `vllm download-kernels` installed FlashInfer's prebuilt kernels for this GPU family (`flashinfer-jit-cache-sm120f`), and the server still reports FlashAttention 2 though NVIDIA's profile asks for 3; DeepGEMM stays off for want of `nvcc`. **Not known, to look up before promising:** whether this vLLM runs NVFP4 or MXFP8 for the Parse architecture (encoder and mBART decoder), what tool makes such weights for it, and what it costs in CER. To measure, on a quiet GPU, same 200 test pages: BF16 as now, each quantised form, pages a second and CER and WER; a form is kept only if the text holds. Goes in the published README (E55) with the commands | |

Queued, not started: Nemotron 3.5 Parse's double pass with our fine-tuned reader (the page parsed once, then each region it calls a table cut out and parsed again at higher detail and merged back: NVIDIA's card gives 72.7 -> 74.9 on ParseBench; our tables are 11 % of scans and the reader's weakest kind), on the test pages by kind once E22 gives every GT page its kind; Toponymy names for the clusters (needs E15); the rare kinds' recall (needs E15); a step placing
the cut at full resolution, scored against folds marked by hand; Jev's page score on the calibration scans; the
writing layer on the calibration scans; HiRank and duplicate columns; curated pretraining (task 7); the line finder
test (task 8, not approved).

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

## The trusted-cut pretraining's first answer (E38, 18 Nov 08:14)

Nemotron 3.5 Parse pretrained one epoch on 139,937 pool pages whose cut is trusted (both pools, old labels unfiltered;
dev loss on the GT 1.77 at its end), then fine-tuned on the GT as `nemotron-parse-all` (2 epochs, dev loss 1.134 at
its end against 1.155 without pretraining and 1.139 after the unfiltered pretraining), read the 200 test pages with
the guard. `prototypes/compare_reads.py` scores every run's stored reading page by page (it reproduces `htr-eval`'s
figures to the digit) and resamples the pages 10,000 times:

| Run | Pool pages before the GT | GT epochs | CER | WER | without the two pages of partial truth |
|---|---|---|---|---|---|
| `nemotron-parse-all` | none | 2 | 7.55 % | 20.50 % | 6.27 % |
| `nemotron-parse-svea-finetune` | 53,490, Svea, cut by the old labels' lines | 2 | 7.18 % | 19.43 % | 5.87 % |
| `nemotron-parse-trusted-finetune` | 139,937, both pools, trusted cut | 2 | **6.92 %** | **18.61 %** | 5.63 % |
| `nemotron-parse-more` | none | 4 | 6.73 % | 18.83 % | 5.45 % |

- **Against the same recipe without pool pages it helps, and this time beyond doubt on 200 pages:** 0.63 points
  (95 % interval +0.14 to +1.16; a gain in 99.5 % of the draws), better on 123 pages and worse on 58.
- **Against the unfiltered Svea pretraining it is not shown to be better:** 0.26 points (interval -0.39 to +1.05; a
  gain in 74 % of the draws), better on 93 pages and worse on 79. The two differ in three things at once (2.6 times
  the pages, both pools, the trusted cut), so the cut's own worth is not measured.
- **Against two more epochs on the GT alone it is not ahead:** 6.92 % against 6.73 % (interval -0.65 to +0.22 for
  the difference), for about the same GPU time (22 hours against 19). Words are a little better (18.61 % against
  18.83 %). The comparison that counts, the pretraining followed by 4 GT epochs, is `nemotron-parse-trusted-more`,
  started 08:14, ends about 18:00.
- **Where the gain is** (by the pool a test page's volume lies in; those volumes were left out of the pretraining):

  | Test pages | Pages | Characters | none | unfiltered | trusted | 4 GT epochs |
  |---|---|---|---|---|---|---|
  | volume in the andra arkiv pool | 23 | 20,073 | 4.17 | 2.89 | 2.77 | 2.90 |
  | volume in the Svea pool | 8 | 6,863 | 8.63 | 6.25 | 5.84 | 8.47 |
  | volume in no pool | 169 | 166,213 | 7.91 | 7.73 | 7.46 | 7.12 |

  The pool pages teach the pools' own archives most (1.4 and 2.8 points), and a little outside them (0.45 points,
  where the longer GT training gives 0.79).
- **By kind:** running text 6.24 to 5.83 % (unfiltered 5.67, 4 GT epochs 5.73); tables 17.06 to 14.85 % (unfiltered
  18.14, 4 GT epochs 14.08). The table figures hang on single pages: one inventory page alone is 583 edits between
  the two pretrained runs.
- **Looked at:** the page that got worst (`fbf7d0759ad2_right`, +213 edits) is an early stop: the reader read the
  first block, 383 of 635 characters, and ended. The readings' total length is the same in every run (194,458
  characters against 193,830 and 194,936), so stopping early is not general.
- **What it says about more pool pages.** From 53,490 to 139,937 pages, with a better cut, the gain was 0.26 points
  and not shown. The labels are the old pipeline's, about as wrong as the reader itself is on these pages (5 to 7 %
  on matched lines), so more of the same labels is unlikely to teach much more; better labels are the lever to try
  beside scale: the pages where our reader and the old label agree, or our own reader's readings.

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
| Qwen3-VL-Embedding 2B, image + label text (2,048-d) | 0.430 | 0.647 | not run |
| Qwen3-VL-Embedding 8B, image (4,096-d) | 0.514 | 0.676 | not run |
| Qwen3-VL-Embedding 8B, image + label text | 0.454 | 0.677 | not run |
| Qwen3-VL-Embedding 2B, image | 0.381 | 0.531 | not run |

- TIPSv2 places a page among its own volume's pages more often than DINOv3 (0.726 against 0.653); their EVoC clusters
  match the collections equally. EmbeddingGemma 2 is far behind on page images, and adding the label's text helps it
  only a little.
- Qwen3-VL-Embedding 2B (added 15 Nov evening, the same pages at the same 896 px, through Sentence Transformers
  with its default instruction): level with EmbeddingGemma 2 and as far behind the two vision models. The
  general-purpose image-and-text embedders place a page by what kind of document it is; the vision models by how it
  looks, which is what tells one volume's pages from another's. The 8B is running. `compare_neighbours.py` gives the
  two neighbour columns for every embedding in the folder and reproduces the four earlier rows to the digit.
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
- **Toponymy can name the clusters with this server** (read 15 Nov: toponymy 0.5.4, README and `llm_wrappers.py`):
  `OpenAINamer(model=..., api_base="http://127.0.0.1:8011/v1", use_json_object=False)` goes through LiteLLM to any
  OpenAI-compatible server, vLLM named; `use_json_object=False` because a forced JSON format broke this model's
  answers here. `Toponymy.fit(texts, vectors, map)` wants a text for each scan (the model's captions), the scans'
  embedding vectors, a low-dimensional map of them to cluster on (`ToponymyClusterer`, or our EVoC layers), and a
  sentence-embedding model to pick each cluster's most telling captions.
- **First 5,010 of the 45,000 described scans, with their embeddings** (`prototypes/scan_clusters.py`; scans whole at
  448 px, EVoC with its defaults, a linear head held out by volume, 5 folds):
  - What the model says the scans are: handwriting 81 %, table or ledger 17 %, sign or card 1.1 %, printed, cover and
    blank 0.2 % each; opening with both pages written 78 %, one page empty 18 %, single page 3.4 %, not a page 0.4 %.
    So four scans in five are openings with writing on both pages, and a single page is rare.
  - A head on the embeddings gives the model's answer for a scan of an unseen volume: kind 0.916 (always saying the
    commonest: 0.811), layout 0.940 (0.781), written side 0.941 (0.780), with TIPSv2 and DINOv3 together; either
    alone within 0.03. **Writing at the fold: 0.760 against 0.742, nothing**: a 448-px view of the whole scan does not
    hold it, or the model's answer is noise; to settle on scans where the labels' boxes reach the fold.
  - EVoC's own clusters carry less than the head: TIPSv2 32 clusters with 37 % of scans left as noise, a scan's
    cluster's commonest kind its own for 0.92; DINOv3 3 clusters at the chosen layer, no better than chance. The
    finer layers (1,100, 200, 75) not yet scored; the settings not yet tuned.
  - Not yet known: the recall of the rare kinds (signs, covers, blanks, single pages), which is what the filter needs;
    with 55, 8, 7 and 171 examples it waits for all 45,000.
- Table across the fold: 1 % and turned writing: 0 % in the pilot: too few to judge those two answers; to check on
  scans known to be such (ledgers over both pages; the orientation network's turned pages).

## Placing the cut on the fold (15 Nov, E18)

A second step after the gutter network, from the image alone (`prototypes/cut_place.py`): a band of 3 % of the width
to either side of the network's answer, read at 2,400 px of scan width.

- **First idea, wrong: the column with the least writing.** The fold is itself a dark line, so the emptiest column is
  the page's inner margin beside it; it cut into labelled lines twice as often as the network.
- **Second: the column that is dark down its whole height.** Writing is dark in some rows of a column, a fold in
  nearly all of them: thin dark vertical detail (a closing along the row minus the image), scored by a low quantile
  down the column, the band sheared a little each way for a leaning fold. A fold stands out clearly in 68 % of 1,453
  Svea scans (the score is two-humped: about 1.7 or about 30-46); in the rest the network's answer is kept.
- **The labels cannot judge it.** Against the old labels' boxes the new cut looks better on tight bindings (inside a
  box in 16 % of scans against 31 % where the pages' text is under 2 % of the width apart) and worse where the gap
  is 2-5 % (10 % against 6 %). Looked at, the "worse" ones have the new cut on the visible fold and the network's
  in the margin: the old boxes of the left page reach over the fold. A box is not truth at half a percent of the
  width.
- **Judged blind instead:** 40 random scans where the two cuts differ by more than 0.4 % of the width, each drawn
  with the two cuts in colours shuffled per scan, the one on the fold picked before the key was read: the new cut 14,
  the network's 2, no difference to see 24. The two losses are registers, where it took a ruled column line for the
  fold; both were unsure calls.
- **Open:** ruled lines in registers; the 32 % with no fold found (wide gutters, loose sheets) keep the network's
  place; not yet run on andra arkiv or the ground truth; the judging was by one pair of eyes on 40 scans.

## What the model's answers are worth, kind by kind (15 Nov, on the first 32,864 described scans)

18 scans drawn from each answer and looked at:

| The model says | Right of 18 | What the wrong ones are |
|---|---|---|
| not a page | 18 | (signs, reel targets, spines, covers) |
| single page | 18 | |
| opening, one page empty | 14 | two single pages with a strip of the next, one opening written on both pages, one with both empty |
| table or ledger (kind) | **8** | dense prose called a table |
| table across the fold | 10 | prose openings |
| turned writing | about 13 | it means "some writing is turned" (an address panel, a margin note), not most |

- **The form was the fault, not the model (E19):** asked "table or ledger, or running text?" with each option
  defined in a line, it is right on 33 to 35 of the 35 of those scans that the eye could call (three runs: the
  question alone, with thinking, and as the first of two questions). The first form gave bare option names.
  **Every option in a form gets a line saying what it means.** Thinking adds nothing here.
- So the first form's kind (handwriting against table), table across the fold and turned writing are not to be used;
  a second pass asks the two questions again with definitions (E20). Its "not a page", "single page", the signs and
  covers, and which side is written stand.
- What the first 32,864 scans are, by the answers that stand: not a page 0.4 %, single page 3.4 %, opening with one
  page empty 17.5 %, opening with both written 78 %; signs and cards 1.0 %, printed or typed 0.3 %, blank 0.2 %,
  cover or spine 0.2 %.

## A head on the embeddings, class by class (15 Nov 23:30, E16 on all 44,932 described scans)

TIPSv2 and DINOv3 embeddings of the whole scan (448 px), joined and reduced to 256 dimensions; a logistic head with
balanced classes, fitted on four fifths of the volumes and scored on the fifth, five times (`prototypes/scan_head.py`).
The truth here is the model's answer, which the eye found right for these answers in 14-18 of 18.

| Answer | Class | Scans | Recall | Precision |
|---|---|---|---|---|
| handwriting or not | not handwriting (sign, cover, blank, printed, not a page) | 770 | 0.988 | 0.739 |
| | handwriting | 44,162 | 0.994 | 1.000 |
| pages | opening, both pages written | 35,100 | 0.931 | 0.993 |
| | opening, one page or both empty | 8,073 | 0.947 | 0.770 |
| | single page (and loose sheets) | 1,571 | 0.905 | 0.808 |
| | not a page | 188 | 0.878 | 0.527 |
| written side | both / right / left | 35,106 / 6,958 / 2,052 | 0.92 / 0.91 / 0.94 | 0.99 / 0.77 / 0.64 |

- So the embeddings can carry the coarse description to every scan: what is not handwriting is found almost whole
  (a quarter of what is flagged is handwriting after all, 0.4 % of the handwriting scans), and "both pages written"
  can be trusted when the head says it.
- The kinds of non-handwriting are not told apart by the head (sign against cover against not a page: precision
  0.3-0.7); as one class they are.
- Where the head is unsure (a single page against an opening with an empty side), the scan can go to the model
  itself: the unsure share decides how many scans that is. Not yet measured: the head's probabilities and how many
  scans fall under a threshold.

## SCALAR and HiRank (16 Nov, E25, first run on Qwen3-VL-Embedding 8B's vectors)

- **EVoC does not cluster every scan, and its settings do not change that (E23).** At about 50 clusters it leaves 54 %
  of the 14,985 scans in none.
- **SCALAR does what was missing.** Its pipeline: k-means representatives of the scans, EVoC on the representatives,
  every scan given the label its nearest representatives vote for (`ClusterScalar`, `KMeansDownsampler`,
  `KNNUpsampler`; read at a566b3f). With 4,000 representatives and the noise vote at 0, 7 % stay unclustered, 51
  clusters, the largest 8 % of scans, and the agreement with the kinds of scan is the same as EVoC's on the scans it
  kept (0.237 against 0.232). With 1,500 representatives 14 %. It is also the form that reaches 3.15 million scans:
  EVoC only ever sees the representatives.
- The cheap alternative, voting EVoC's unclustered scans into their neighbours' clusters, covers every scan but one
  cluster swells to 14 % of them.
- **HiRank finds one-offs, not the non-handwriting.** `RankOD` (15 neighbours, rank up to 100, cosine): its score is a
  density, low for an odd scan (its README says the opposite of its docstring; checked on the numbers). Signs, reel
  targets and covers are many of a kind, so they are dense, not odd: scored against running-text openings their AUC
  as outliers is 0.18. 1,465 scans tie at density 0, so the "oddest 18" drawn are an arbitrary 18 of those: mostly
  ordinary openings, with a letter with a drawing, a fold-out and a sheet laid across. To use it: a longer rank list
  so the ties break, then the lowest hundred looked at; non-handwriting is the head's and the clusters' job.
- Not yet: either tool on Nemotron Embed's vectors; a look at SCALAR's clusters by sampling scans from each.

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
