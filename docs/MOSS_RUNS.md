# MOSS-Image runs on Leonardo: the experiment log

One entry per run: the question it answers, exactly how its cache and training were set, its Slurm jobs and its
folder, then its state and results. Paths are relative to the project's `$WORK` on Leonardo; each run folder also
holds a `run.json` with this entry's facts, beside the trainer's own records (a cache's `cache.json` holds its options,
each checkpoint's `state/progress.json` the training settings). How to run them: [MOSS_HPC.md](MOSS_HPC.md).

## The hand: what has been tested and what has not (8 Nov 2026)

The page's look (paper, ink, scan) follows the sample page; its handwriting does not. This is the record of every
test of the hand so far, on the Spark and on Leonardo, so the next one is chosen from it.

Tested:

| Test | Result | Where |
|---|---|---|
| One scribe's hand learned by an adapter (LoRA on Base: the 1715 letter book, Påhlman, landskapshandlingar) | **Works**: the clerk's letterforms on held-out windows at step 350, and on whole held-out pages with every line in place and the volume's look | `ea73d21`, `63f0092`; adapters `Gabriel/diffu-moss-hand-*` |
| Eight hands in one model, chosen by naming the collection (full_r3, 31 Oct, step 1,500) | **Half works**: writer ID of the drawn lines against the real pages' at the same size: Påhlman 88 % (81 %), Överståthållarämbetet 69 % (92 %), Kopparberg 2 % (68 %), Göteborg 0 % (57 %); the sample page ignored | `e0feb18` |
| The hand copied from the sample page: the content pilot (full_c1, 2,500 steps) and B at step 1,000 | **Fails**: one page drawn with four samples keeps the same hand letter for letter, only its paper changes; B's swap test H1 1/18 | TODO.md, the recipe; Results below |
| The VAE: does encoding a page lose its hand? | **No**: writer ID 0.59 on real lines, 0.57 on the same lines encoded and decoded (47 pages) | TODO.md, the recipe |
| The hand over training steps (the pilot's checkpoints) | No trend toward the sample's hand at any checkpoint | 8 Nov, `~/diffu_queue/hand_diag/hand_trend` |
| Reference guidance 2 (a third pass toward the sample at drawing time), B at step 1,000 on the swap set | **The look follows better, the hand not at all**: L1 6/6 (5/6 without), L2 7/7, L3 3/3, H1 1/18 (1/18 without); by eye the grey microfilm page now comes out grey from its own sample (beige without), and every page keeps one tidy cursive letter for letter whatever the sample | 8 Nov, Spark, `~/diffu_queue/eval_b/show_rg2` |

Not tested yet:

| Test | The question | State |
|---|---|---|
| A sample by the same scribe (A′) | Was the sample, often another scribe's page, the cause? | queued on Leonardo |
| Hand by the collection's name at full scale (N) | Does naming carry the hand for more collections with all the data? | queued on Leonardo |
| The swap test on windows | Every page of the set gives 2 windows (whole lines at ~2.5x the page's writing size: a 63 px line pitch against 23-25 px), each with a window of its sample at the same size, so B trains on them; the swap test has only been run on whole pages. Does the hand follow where the letters are large? If so the page's writing size is the cause (2 MP, or pages drawn region by region); if not, the way the sample is fed in | not run; drawing only, B's checkpoint |
| An adapter's hand with a sample page's look (recipe B's plan) | A hand LoRA trained on top of B, drawn with a sample page from another volume: the hand from the adapter, the look from the sample? | not run; needs B's checkpoint |
| Reference guidance 3 on the swap set | Does a stronger third guidance pass toward the sample carry the hand? | drawing on the Spark (B step 1,000) |
| The sample beside the page (its own position grid) | Does the hand follow when the sample's tokens no longer share the page's (row, column) positions? | not built |
| Training weighted toward low noise (`--sigma-shift`, `ea9fd2e`) | Does the hand carry when more steps fall where the letter strokes are decided? | code only, no run recorded |
| Writer-ID supervision | Does an explicit writer signal in training make the hand follow? | not built |

## Common to every run (7 Nov 2026)

- **Set**: `moss/page_pairs_full_1mp`, the whole GT at 1 MP (`diffu_page.data.page_pairs`, 2 windows a page, 10 % of
  volumes held out, seed 0), without `GT-RA_landskapshandlingar_20260227` (a copy of `_20260220`): 143,379 training
  and 11,920 validation rows, 54 collections, 203 volumes, every row in every prompt form, legible samples.
- **Training**: full fine-tune from MOSS-Image Base (`85975793`), 2 nodes (8 A100 64 GB), 16 samples a step, 9,000
  steps (~one pass), learning rate 5e-5 (the trainer's default, constant after its warmup), EMA 0.999, a checkpoint
  every 1,000 steps (with its state, to resume), validation every 500; two chained 24 h jobs (QOS `normal`), the
  second going on from the newest whole checkpoint (`RESUME=latest`).
- **Code**: main at `012d02d` or later, torch 2.14.1+cu126 (Leonardo's driver is 535).
- **Judged by**: the swap test and held-out pages beside the real scans (look, hand, realism; writer ID and CER as
  information only), on the held-out cache of the run's mix.

## Runs

| Run | Question | Cache (`moss/cache_full_*`) | Validation | Jobs | Folder | State |
|---|---|---|---|---|---|---|
| **B** | Look and text from the sample page, hands from per-hand adapters afterwards (the chosen recipe) | `b`: content mix (sample alone 60 %, sample with look words 20 %, no sample 20 %; nothing named), a look change for half the sample-form entries, no hand changes | `cache_full_b_val`, `_heldout` (content mix) | cache 63698295, trial 63750935 (100 steps, debug QOS), run 63744195 then 63750939 | `moss/runs/full_b` (trial: `trial_b`) | cache done 10:22 (143,379 entries, 1.5 TB); trial passed 10:51 (below); run started 7 Nov ~18:15 after 7.8 h in the queue |
| **A** | (cancelled 8 Nov before training: its samples, any page of the volume, are often another scribe's, so a hand that does not follow would say little; replaced by A′) | `a` (built, unused) | — | cache 63750290 (done), runs 63750297/63750306 cancelled | — | cancelled |
| **A′** | Does scale make the sample's hand follow when the sample is the same scribe's? (the pilot's recipe at full scale on nearest samples) | `a2`: the set rebuilt with `--sample-pick nearest` (`moss/page_pairs_full_1mp_near`: the other page of the opening, else the closest scan), content mix, a look change and a hand change for half the sample-form entries each | `cache_full_a2_val`, `_heldout` | set 63853412, cache 63853413, run 63853414 then 63853415 | `moss/runs/full_a2` | queued 8 Nov |
| **N** | Hand by name: the collection named in the prompt, the look from the sample | `n`: named mix (archive named in 60 %: no sample 20 %, sample plus archive 20 %, sample plus archive and look words 20 %; sample alone 40 %), a look change for half the sample-form entries, no hand changes | `cache_full_n_val`, `_heldout` (named mix) | cache 63750293, run 63750311 then 63750316 | `moss/runs/full_n` | cache started 7 Nov ~18:15 (time limit lowered to 10 h) |

Before these: the content pilot (`runs/moss_content_pilot/full_c1` on the Spark, 2,500 steps of one sample from the
same 55 collections at 80 scans each): the look follows the sample, the hand does not (TODO.md, the recipe).

## Notes

- 7 Nov, B: the cache job is marked FAILED (exit 1) though its cache is whole (`cache.json` written, 86,027 + 28,676 +
  28,676 entries, the look changes 14,338 of each kind): its last line, a `tail -1` of several logs, failed. Slurm then
  cancelled the trial waiting on it (`afterok`, 63745127) and the run's first job (63744194); the trial was submitted
  again without a dependency (63750935), the job that waited on the cancelled one (63744195) is now the run's first,
  63750939 its second. The A and N cache jobs (`build_cache.sbatch`) exit with the cache's own status.
- 7 Nov, throughput on 8 A100 64 GB (recipe B's cache, 2 nodes, 16 samples a step, debug QOS): with gradient
  checkpointing (as B runs) 11.57 s a step, 20.0 GB a GPU at most; without it every GPU ran out of memory in step 1's
  first forward pass (a 1 MP page and its 1 MP sample, ~9,000 tokens, keep more activations than the ~44 GB left), one
  rank then hung and the rest timed out in FSDP's all-gather after 10 min. So the free memory buys speed only in part:
  checkpointing some blocks rather than all, or keeping the gathered weights from the forward to the backward pass; a
  profile of the current setting (job 63806582) decides which.
- 7 Nov, profile of the B setting (job 63806582, ranks 0 and 4, steps 4-6, torch.profiler): the GPUs busy 84-98 % of
  each 11.5 s step; compute kernels 7.3-9.4 s; NCCL 4-10 s in all, 1.4-4.5 s of it not under compute (weights gathered,
  AllGather, ~885 calls a step; gradients reduced in float32, ReduceScatter); idle 0.3-1.9 s. Inter-node traffic goes
  over InfiniBand with GPUDirect RDMA (NCCL's log). Tested (30 steps, 2 nodes): NCCL_PROTO=Simple 11.64 s a step (no
  gain); keeping each block's gathered weights for its backward (--no-reshard-after-forward) 11.48 s (-0.8 %) at 28.9 GB
  a GPU. The step is bound by compute, a quarter of it the recompute of gradient checkpointing; checkpointing some
  blocks instead of all is the trade left for the free memory.
- 7 Nov, partial checkpointing (--uncheckpointed-blocks, 30 steps, 2 nodes): the last 12 of the 42 blocks keeping their
  activations 11.23 s a step (-2.9 %) at 49.2 GB a GPU, the allocator already retrying after out-of-memory; 24 blocks ran
  out of memory. A kept block costs ~2.4 GB a GPU for ~0.25 % of the step. Decision: A, N and B's second job keep B's
  setting (all blocks checkpointed, 20 GB a GPU, 11.6 s a step): the fastest that is safe, and the runs stay
  comparable. torch.compile (elementwise kernels ~8-10 % of GPU time) is the lever left, for a later run.
- 8 Nov, B at step 1,000 on the pilot's swap set (6 held-out pages, own sample and 3 others): L1 5/6, L2 7/7, L3 3/3,
  H1 1/18 (the pilot at step 2,000: 6/6, 5/7, 3/3, 2/18). By eye: text and layout exact and legible, the look follows on
  the colour pages (a grey microfilm page drawn beige even from its own sample), the hand one tidy cursive whatever the
  sample, neater than any real hand, no blots. Reference guidance 2 and 3 on the same set is drawing on the Spark.
- 8 Nov, why the hand may not follow (from reading the MOSS-Image report, its repo, KohakuMossTrainer and DiffSynth): a
  page's sample was any page of its volume (often another scribe's), and the sample's tokens share the drawn page's
  (row, column) positions (MOSS's reference axis turns only ~8 of 128 head dimensions at index 1), favouring the same
  place (paper, ink) over the same letters elsewhere. A′ tests the first; a pilot with the sample beside the page, the
  second.
- 8 Nov, Muon orthogonalised by owner ranks (e404761), 30 steps on 2 nodes: 11.07 s a step against 11.57 (-4.3 %), the
  same memory; its losses differ from the old code's by 2.6e-4 on average, as two runs of the old code do (2.5-3.7e-4:
  flash attention's backward is not deterministic), so the training is the same. Leonardo's checkout is at e404761 from
  8 Nov 08:21: N, A′ and B's second job run it; B's first job started on the old code. The stall fixes (3ac313c) wait
  for their own GPU test.

## Results

- **B trial** (7 Nov, 2 nodes, 100 steps of 16 on cache `b`): training loss 0.837 at step 1, 0.751 at 10, 0.60-0.65
  from 40 to 100; gradient norm from 3.3 to 0.2-0.5; validation loss at step 50 0.686 in every form; 10-17 GB a GPU;
  11.8 s a step (1.36 samples/s, 1.7x one node's), so the run's 9,000 steps take ~30 h; checkpoints at 50 and 100.

- **B, first 250 steps** (7 Nov): training loss 0.837 at step 1, 0.639 over 11-50, 0.617 over 201-250; gradient norm
  0.29-0.68 after step 10; ~21 GB a GPU with the EMA; 11.9 s a step, ~29 h for the 9,000.

(Filled in as each run is judged: validation loss by form, the swap test, held-out pages beside the scans.)
