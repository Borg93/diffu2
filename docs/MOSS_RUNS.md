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
| The same on windows: the swap test's six held-out pairs are windows (`*_w0`-`_w2`: whole lines at ~2.5x a page's writing size, a 63 px line pitch against 23-25 px), each sample cut at the same size | **Fails the same way**: the letters' size is not what stops the hand (corrected 8 Nov: an earlier version of this table listed the window swap test as not run) | `runs/moss_full_b/swap_step1000` on the Spark |
| The VAE: does encoding a page lose its hand? | **No**: writer ID 0.59 on real lines, 0.57 on the same lines encoded and decoded (47 pages) | TODO.md, the recipe |
| The hand over training steps (the pilot's checkpoints) | No trend toward the sample's hand at any checkpoint | 8 Nov, `~/diffu_queue/hand_diag/hand_trend` |
| Reference guidance 2 (a third pass toward the sample at drawing time), B at step 1,000 on the swap set | **The look follows better, the hand not at all**: L1 6/6 (5/6 without), L2 7/7, L3 3/3, H1 1/18 (1/18 without); by eye the grey microfilm page now comes out grey from its own sample (beige without), and every page keeps one tidy cursive letter for letter whatever the sample | 8 Nov, Spark, `~/diffu_queue/eval_b/show_rg2` |
| Reference guidance 3, the same | **The hand still not**: L1 6/6, L2 7/7, L3 3/3, H1 3/18; by eye the three passes keep the same cursive letter for letter as without guidance (a big loose sample hand gives the same tidy hand, the ink darker): the writer identifier is swayed by ink and paper tone, so H1 alone overstates | 8 Nov, Spark, `~/diffu_queue/eval_b/show_rg3` |
| An adapter's hand with a sample page's look (recipe B's plan), the adapter as it is: the 1715 clerk's (`Gabriel/diffu-moss-hand-overstathallarambetet-1715`, trained on Base) loaded onto B at step 1,000; two swap windows (Trolldomskommissionen, Svea hovrätt koncept), each with its own sample and 3 others, from the same noise as B's swap test | **The hand comes, the look mostly stays**: the writer identifier names 80 of 124 drawn lines (65 %) the clerk's against 3 of 124 for B alone (6/14 to 17/17 a drawing, whatever the sample); by eye a thinner pen, more slant, longer looped strokes, a lighter brown ink, toward the clerk's but larger and heavier than his. The look: colour samples give colour pages (all), grey samples grey pages in 2 of 5 strictly (B alone 2 of 5), one grey sample drawn beige that B drew grey; a few words and abbreviations scribbled that B wrote clean | 8 Nov, Spark, `runs/moss_full_b/swap_step1000_lora1715`, `~/diffu_queue/eval_b/lora_on_b.sh` |

Not tested yet:

| Test | The question | State |
|---|---|---|
| A sample by the same scribe (A′) | Was the sample, often another scribe's page, the cause? | queued on Leonardo |
| Hand by the collection's name at full scale (N) | Does naming carry the hand for more collections with all the data? | queued on Leonardo |
| The same with an adapter trained on top of B, in B's sample form | Does an adapter made for B give the whole hand, its words clean, without pulling the paper toward its own volume's? | training on the Spark (8 Nov 14:45): B at step 1,000 as `--transformer`, the pilot cache's 257 entries of Överståthållarämbetet without hand changes (`data_flux3/cache_moss_overstat_b`: 102 sample form, half with a look change, 78 with look words, 77 no sample), rank 32, AdamW 1e-4, 2 samples a step at ~31 s; stopped at step 250 for time, then drawn on the two swap windows as above (`~/diffu_queue/eval_b/lora_b_train.sh`, `stop_at_250.sh`, `lora_b_250.sh`; `runs/moss_full_b/lora1715_on_b`). **Step 250: the hand as the Base adapter's, the look better**: the clerk's named on 74 of 124 lines (60 %; the Base adapter 65 %, B alone 2 %); grey samples give grey pages in 4 of 5 (both others 2 of 5), colour samples colour pages; by eye the clerk's thin slanted hand, the abbreviated names line cleaner than under the Base adapter. Still cream from its own grey sample (saturation 30) as B alone. **With reference guidance 2 the look is right throughout**: every grey sample a grey page (5 of 5, saturation 5-7, the own-sample one too), colour samples colour pages; by eye the same hand letter for letter as without guidance, only the paper changed, though the writer identifier names fewer lines the clerk's (49 of 124, 40 %): it reads the paper's tone too (the clerk's real volume is beige), so on grey pages it understates the hand (`lora_b_250_rg2.sh`, `runs/moss_full_b/lora1715_on_b/swap_step_000250_rg2`) |
| The same on B at step 7,000 (its EMA, copied from Leonardo 8 Nov) | Does the route hold on a B trained 7x longer: B alone and B + the 1715 adapter trained on it (250 steps, as above), both at reference guidance 2 on the two windows; and the full swap test of B's raw weights at 7,000 against step 1,000's | **B alone at 7,000 (EMA), guidance 2: the look follows every sample, the hand still not**: grey samples give grey pages (saturation 3-4), colour samples colour pages (L1 2/2, L2 2/2, L3 1/1); text and layout exact; by eye the hand thinner, lighter and more slanted than at step 1,000, closer to the real windows, but the same letter for letter whichever sample is given. The writer identifier's H1 3/6 (0 at step 1,000) comes from grey tone, not letters: drawings identical in their strokes are named 0-93 % the sample's. **B + the 1715 adapter trained on it (250 steps), guidance 2: the look right everywhere, the hand weaker than on step 1,000**: grey samples grey (saturation 3-4), colour colour, the text clean; by eye the clerk's loops and flourishes and a lighter stroke over B's own hand, a smaller change than at step 1,000 (B's own hand is already thinner and more slanted). The identifier names 28 of 124 lines the clerk's (23 %; on step 1,000 with guidance 2, 40 %; B alone at 7,000, 3 %); counted on the beige drawings alone, where the clerk's beige volume does not bias it, 28 of 45 (62 %) against 39 of 45 (87 %) on step 1,000. By eye, though, the change from B's hand is as clear as on step 1,000 (a thin pen, lighter brown, the slant and looseness of the clerk's; side by side `~/diffu_queue/eval_b`): the identifier understates it. **500 adapter steps** (9 Nov, `lora1715_on_b7000_500`): the same as 250 by eye and by the identifier (29 of 124, 23 %; beige 29 of 45, 64 %), so 250 steps suffice. **The full swap test of B's raw weights at 7,000** (no guidance; `swap_step7000`, `~/diffu_queue/eval_b/show_7000`): L1 6/6, L2 7/7, L3 3/3 (step 1,000: L1 5/6), H1 8/18 (1/18), but by eye each page keeps one hand letter for letter whatever the sample: the identifier's H1 is the paper's tone again. Done 9 Nov 06:31 (`~/diffu_queue/eval_b/b7000_tests.sh`; `runs/moss_full_b/swap_step7000*`, `lora1715_on_b7000`) |
| The adapter route for three more scribes, on B at 7,000 (EMA) | Does an adapter per scribe carry other hands than the 1715 clerk's: Påhlman's travel journals, Kopparberg (Bergskollegium, E II b 57), Riksregistraturet, each from its pairs in the pilot cache without hand changes (`data_flux3/cache_moss_{pahlman,kopparberg,riksregistraturet}_b`: 245, 241, 261 entries), 250 steps, drawn on the two swap windows at guidance 2 and scored | running on the Spark from 9 Nov 09:04, ~3 h a scribe (`~/diffu_queue/eval_b/scribes_b7000.sh`; `runs/moss_full_b/scribe_*_on_b7000`). **Påhlman: no change** (12:11): the identifier names 0 of 124 drawn lines his, as for B alone, and by eye the hand is B's letter for letter; his adapter's loss barely moved (0.593 over steps 1-50, 0.578 over 151-250; the 1715 clerk's 0.639 to 0.601) with half the gradient (0.019-0.025 against 0.029-0.045): B already fits his pages, so the hand, a small part of the loss, gives the adapter little to learn. Queued after the three: Påhlman again with the noise levels shifted towards the clean end (`--sigma-shift 0.33`) and with 3x the learning rate (`~/diffu_queue/eval_b/pahlman_strong.sh`) |
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
| **B** | Look and text from the sample page, hands from per-hand adapters afterwards (the chosen recipe) | `b`: content mix (sample alone 60 %, sample with look words 20 %, no sample 20 %; nothing named), a look change for half the sample-form entries, no hand changes | `cache_full_b_val`, `_heldout` (content mix) | cache 63698295, trial 63750935 (100 steps, debug QOS), run 63744195 then 63750939 | `moss/runs/full_b` (trial: `trial_b`) | cache done 10:22 (143,379 entries, 1.5 TB); trial passed 10:51 (below); run started 7 Nov ~18:15 after 7.8 h in the queue; first job to step 7,050 (8 Nov 18:12); the second (63750939) cancelled 9 Nov before it started: validation flat since 6,000 at a constant learning rate, so its last 2,000 steps went to A′ from B instead. B ends at step 7,000 (`checkpoints/step_007000`, its EMA beside it) |
| **A** | (cancelled 8 Nov before training: its samples, any page of the volume, are often another scribe's, so a hand that does not follow would say little; replaced by A′) | `a` (built, unused) | — | cache 63750290 (done), runs 63750297/63750306 cancelled | — | cancelled |
| **A′** | Does scale make the sample's hand follow when the sample is the same scribe's? (the pilot's recipe at full scale on nearest samples) | `a2`: the set rebuilt with `--sample-pick nearest` (`moss/page_pairs_full_1mp_near`: the other page of the opening, else the closest scan), content mix, a look change and a hand change for half the sample-form entries each | `cache_full_a2_val`, `_heldout` | set 63853412, cache 63853413, run 63853414 then 63853415 | `moss/runs/full_a2` | queued 8 Nov; its two training jobs from Base (63853414, 63853415) cancelled 9 Nov for A′ from B (below); the set and cache jobs kept |
| **A′ from B** | The same question for a fraction of the cost: B at step 7,000 trained on A′'s cache, so it keeps what B learned and has only the hand from the same scribe's sample to learn | `a2` (as A′) | `cache_full_a2_val`, if built | run 63956218 after the cache 63853413 (`hpc_full_from.sbatch`: the repo's script with `TRANSFORMER`) | `moss/runs/full_a2b` | queued 9 Nov: from `runs/full_b/checkpoints/step_007000/transformer` (a fresh optimizer, 50 steps' warmup), 3,000 steps of 16, lr 5e-5, EMA 0.999, a checkpoint every 1,000, 12 h limit |
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
- 8 Nov, B at step 1,000 on the pilot's swap set (6 held-out windows, own sample and 3 others): L1 5/6, L2 7/7, L3 3/3,
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

- **B, first job** (63744195, ended 8 Nov 18:12 at its 24 h limit, as planned): step 7,050, the last whole checkpoint
  step 7,000 (with its state); training loss ~0.59-0.62 and gradient norm ~0.1 at the end, 12.7-13.3 s a step;
  validation loss flat from step 6,000 (sample form 0.6584-0.6585, with look words 0.6583-0.6586, no sample
  0.6598-0.6606). The second job (63750939, `RESUME=latest`) goes on from step 7,000; on 8 Nov 19:30 it and every
  queued MOSS job waited on priority (~135,000-137,000, the median of 10,613 jobs pending on `boost_usr_prod`), Slurm
  giving no start time. Its second job's limit lowered to 10 h (8 Nov 19:40; ~2,000 steps at ~13 s, the validations and two saves: ~8 h) so it fits a gap sooner.

- **B published** (9 Nov): the EMA weights at step 7,000 as a new commit of `Gabriel/diffu-moss-r6b` (`9da723d`; its first version stays at `726b9fe`), with a model card (data, recipe, validation by form, the swap test, the limitations) and `tensorboard/` (7,000 steps and 14 validations from `runs/full_b/train_log.jsonl`, converted by `~/diffu_queue/hf_publish/to_tensorboard.py`); `config.json` written as `MossTransformer` (the checkpoint's says `FSDPMossTransformer`, FSDP2's wrapper class, and names a cluster path) and `export.json` without cluster paths. README, INFERENCE.md and the staging Job pin the new commit.

(Filled in as each run is judged: validation loss by form, the swap test, held-out pages beside the scans.)
