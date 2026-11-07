# MOSS-Image runs on Leonardo: the experiment log

One entry per run: the question it answers, exactly how its cache and training were set, its Slurm jobs and its
folder, then its state and results. Paths are relative to the project's `$WORK` on Leonardo; each run folder also
holds a `run.json` with this entry's facts, beside the trainer's own records (a cache's `cache.json` holds its options,
each checkpoint's `state/progress.json` the training settings). How to run them: [MOSS_HPC.md](MOSS_HPC.md).

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
| **A** | Does scale alone make the sample's hand follow? (the pilot's recipe at full scale) | `a`: content mix, a look change and a hand change for half the sample-form entries each (as the content pilot) | B's (same mix) | cache 63750290, run 63750297 then 63750306 | `moss/runs/full_a` | cache started 7 Nov ~18:15 (time limit lowered to 10 h) |
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

## Results

- **B trial** (7 Nov, 2 nodes, 100 steps of 16 on cache `b`): training loss 0.837 at step 1, 0.751 at 10, 0.60-0.65
  from 40 to 100; gradient norm from 3.3 to 0.2-0.5; validation loss at step 50 0.686 in every form; 10-17 GB a GPU;
  11.8 s a step (1.36 samples/s, 1.7x one node's), so the run's 9,000 steps take ~30 h; checkpoints at 50 and 100.

- **B, first 250 steps** (7 Nov): training loss 0.837 at step 1, 0.639 over 11-50, 0.617 over 201-250; gradient norm
  0.29-0.68 after step 10; ~21 GB a GPU with the EMA; 11.9 s a step, ~29 h for the 9,000.

(Filled in as each run is judged: validation loss by form, the swap test, held-out pages beside the scans.)
