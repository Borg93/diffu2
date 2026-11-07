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
| **B** | Look and text from the sample page, hands from per-hand adapters afterwards (the chosen recipe) | `b`: content mix (sample alone 60 %, sample with look words 20 %, no sample 20 %; nothing named), a look change for half the sample-form entries, no hand changes | `cache_full_b_val`, `_heldout` (content mix) | cache 63698295, trial 63750935 (100 steps, debug QOS), run 63744195 then 63750939 | `moss/runs/full_b` (trial: `trial_b`) | cache done 10:22 (143,379 entries, 1.5 TB); trial queued |
| **A** | Does scale alone make the sample's hand follow? (the pilot's recipe at full scale) | `a`: content mix, a look change and a hand change for half the sample-form entries each (as the content pilot) | B's (same mix) | cache 63750290, run 63750297 then 63750306 | `moss/runs/full_a` | queued |
| **N** | Hand by name: the collection named in the prompt, the look from the sample | `n`: named mix (archive named in 60 %: no sample 20 %, sample plus archive 20 %, sample plus archive and look words 20 %; sample alone 40 %), a look change for half the sample-form entries, no hand changes | `cache_full_n_val`, `_heldout` (named mix) | cache 63750293, run 63750311 then 63750316 | `moss/runs/full_n` | queued |

Before these: the content pilot (`runs/moss_content_pilot/full_c1` on the Spark, 2,500 steps of one sample from the
same 55 collections at 80 scans each): the look follows the sample, the hand does not (TODO.md, the recipe).

## Notes

- 7 Nov, B: the cache job is marked FAILED (exit 1) though its cache is whole (`cache.json` written, 86,027 + 28,676 +
  28,676 entries, the look changes 14,338 of each kind): its last line, a `tail -1` of several logs, failed. Slurm then
  cancelled the trial waiting on it (`afterok`, 63745127) and the run's first job (63744194); the trial was submitted
  again without a dependency (63750935), the job that waited on the cancelled one (63744195) is now the run's first,
  63750939 its second. The A and N cache jobs (`build_cache.sbatch`) exit with the cache's own status.

## Results

(Filled in as each run is judged: validation loss by form, the swap test, held-out pages beside the scans.)
