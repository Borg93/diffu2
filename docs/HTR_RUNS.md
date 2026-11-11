# GT-Lejonet reader runs on dmlpai01: the experiment log

The full runs that are meant to beat Riksarkivet's Swedish Lion TrOCR
(`Riksarkivet/swedish-lion-trocr-large-handwritten-char`) on its own data. How the sets and the trainings are made, and
why each choice: [lion_data_prep/README.md](../packages/htr_train/lion_data_prep/README.md). The design behind
`htr-train`: [HTR_TRAIN.md](HTR_TRAIN.md). This page holds what is running, what is queued after it, what has been
measured, and what comes next. Update it when any of those changes.

Paths are relative to the checkout on dmlpai01 (`~/Desktop/diffu2`). `runs/` is not in git: the scripts named there
live only on that disk.

## Rules of the machine

- GPUs 1 and 2 only. **GPU 0 is another user's (marerle): never use it.**
- Every job runs in the tmux session `htr`, so it outlives a lost login or a new Claude session.
- Packing is off for MiniCPM-OCR and Jev: Qwen3.5's linear attention carries state from one packed sample into the
  next.

## Now: `minicpm-all` (MiniCPM-OCR, full fine-tune)

- **Spec:** `packages/htr_train/lion_data_prep/configs/minicpm_all.yaml`
  - **Data:** 1,481,877 training lines (scaled, white outside their polygon) and 83,072 training regions.
  - **Recipe:** ms-swift `swift sft` on GPUs 1 and 2, DeepSpeed ZeRO-2, 4 samples a GPU x 16 accumulation = 128 a step,
    gradient checkpointing in the language model and the vision encoder, lr 5e-6 cosine, 3 epochs = 36,681 steps.
  - **Checkpoints:** eval and save every 1,000 steps; the one of least eval loss is kept as the result.
- **Folder:** `runs/htr/minicpm-all/output`; its log is `runs/htr/pipeline.log`.
- **Stopped 8 Nov 21:55 after checkpoint-28000** (2.3 of its 3 epochs), by the user's call: a demo run, and the
  remaining 8,700 steps were worth about 0.1 point while TrOCR leads by 0.45. Checkpoint-28000 keeps its optimizer
  state, so the run can be resumed from it. The reader is checkpoint-28000 (dev lines 3.25 % corpus CER, the lowest). First started 4 Nov 23:07.
- **Hangs:** GPU 2 hung twice with Xid 8 ("GPU is probably locked", CUDA error 702), 7 Nov 21:38 and 8 Nov 04:36.
  It is a known, unfixed fault of RTX PRO 6000 Blackwell under sustained load (NVIDIA forum thread 382055: power limits,
  locked clocks and driver changes did not help there). Each time the run resumed by itself from its newest checkpoint,
  losing at most 1,000 steps (about 3 h).

### How it survives

| tmux window | What runs there | What it does |
|---|---|---|
| `htr:queue` | `runs/htr/minicpm-all/stop_and_queue.sh` | stopped the training and its resume loops (8 Nov 21:55), then runs `after_minicpm.sh` |
| `htr:18 gpuwatch` | `nvidia-smi -l 1` on GPUs 1, 2 | temperature, power, clocks, throttle reasons, once a second, to `runs/htr/minicpm-all/gpu_watch.csv` |

Each resume is `runs/htr/minicpm-all/run_resume.sh`:
- It waits until GPUs 1 and 2 are free.
- It resumes from the newest checkpoint that has its `trainer_state.json`; a crash while saving leaves one without.
- Its log is `runs/htr/minicpm-all/resume_loop.log`.
- When the training finishes it writes `runs/htr/pipeline_markers/minicpm_train.done` and starts the queue below.

### Queued after it: `after_minicpm.sh`

Started 8 Nov 21:55, when the training was stopped. Each step leaves a marker in `runs/htr/pipeline_markers`, so running the
script again resumes after the last finished step.

1. **Score the best checkpoint.** Of minicpm-all's checkpoints on disk, the one of lowest corpus CER on the 1,000 dev
   lines (not eval loss, which rose in epoch 3 while CER fell), linked as `runs/htr/minicpm-all/best`; read through vLLM on all 6,358 out-of-domain
   lines and 5,000 `test_0050` lines, beside TrOCR. Result: `runs/htr/final/result.txt`.
2. **Publish the reader:** `Gabriel/diffu-minicpm-ocr-htr` (private). **Done 8 Nov 22:42:** checkpoint-28000, its
   weights (3.1 GB) and configs, commit `7735d9ed`.
   - **Checked as its card says** (`runs/htr/verify/`): downloaded from the Hub, served with the card's command, asked
     with the card's request. 200 drawn test lines read at 4.19 % corpus CER, 10 dev regions at 3.58 %, and the three
     dotted leader lines end with `finish_reason=length` as the card says.
   - **Readings depend on batching by a few near-tie characters.** The same server read the same 200 lines 16 at a time
     and one at a time and differed on 9 (corpus CER 4.15 % and 4.40 %; the queue, 64 at a time, 4.33 %): vLLM is not
     batch-invariant. Scores on a few hundred lines carry about 0.2 points of this noise.
   - **Jev's card, checked without a GPU:** its `pip install` works in a clean environment, but pulls Pillow 13.0.0
     against the pairs' 12.2.0, so the card pins Pillow; both draw the same pixels (25 renders checked). The card's
     recipe redraws 200 of 200 smoke pairs' renderings pixel for pixel.
   - **Jev and the draft are checked the same way once published**, by `runs/htr/verify/watch.sh` (tmux `htr:verify`):
     Jev on 2,000 test pairs (AUROC, renders redrawn), the draft served beside the reader against the reader alone
     (identical readings, spec_decode counters, speed). Results in `runs/htr/verify/check_*.log`.
3. **Jev's sets.**
   - On the first GPU, the reader reads 100,000 training lines and every test line; the pairs follow on the CPU. They
     were estimated at 4.5 h and took about 10 h: about 148,000 lines an hour, one process writing a crop and two
     renders a line into folders of millions of files.
   - Meanwhile, on the second GPU, the reader answers 250,000 training lines and 50,000 training regions for the DSpark
     draft (`runs/htr/final/second_gpu.log`). **Done 9 Nov 00:15.**
4. **The DSpark draft, moved before Jev (9 Nov 07:32, at the user's request).** Its answers were ready at 00:15 and
   both GPUs sat idle while Jev's pairs built on the CPU. `runs/htr/dspark_first.sh` (tmux `htr:dspark`, log
   `runs/htr/dspark_first.log`) runs the queue's own two steps: `configs/dspark_all.yaml` on GPU 1, then publish as
   `Gabriel/diffu-minicpm-ocr-htr-dspark`. It leaves the queue's markers, so the queue skips both after Jev. A small
   process holds GPU 2 until the published draft has been checked with `check_dspark.sh`, which needs both GPUs.
   Until then the queue's wait for free GPUs keeps Jev back.
   - **Its training moved to two GPUs (9 Nov, `runs/htr/dspark_split.sh`, tmux `htr:dsparksplit`).** The run's
     own commands, timed on 2,400 of its samples on the idle GPU 2 (`runs/htr/dspark-speedtest/times.log`):

     | Layout | Prepare 2,400 | Train 390 steps |
     |---|---|---|
     | As configured: vLLM (1 API server) and the trainer on one GPU, 12 preparation workers, 2 loader workers | 90 s | 153 s |
     | Same GPU, 4 API servers, 48 preparation workers, 8 loader workers | 34 s | 153 s |
     | vLLM on GPU 2 (4 API servers), the trainer on GPU 1, 8 loader workers | | 100 s |

     - Preparation is limited by vLLM's single API server, which formats each sample. Training is limited by the
       GPU the trainer shares with vLLM: alone on its GPU, the trainer keeps it ~94 % busy.
     - So 3 epochs take about 10.5 h instead of about 16 h.
     - The script waits for the run's preparation to save its data. It then starts vLLM on GPU 2, stops the run's
       training and vLLM on GPU 1, and trains on GPU 1 into `runs/htr/dspark-all/output`. Data, recipe and epochs
       are unchanged. `dspark_first.sh` then logs the stopped launch as failed, and this script does its remaining
       steps: the markers, the publish and the check.
     - The run's preparation itself was left alone: it was 45 % done, and starting it again with more workers
       would have saved about 20 minutes.
     - **speculators' own docs ask for this layout** (read at the installed commit `560f046`). The DSpark example
       (`examples/train/dspark_qwen3_0_6b_sharegpt_online.sh`) says "online training needs separate GPUs for vLLM
       and training" and serves vLLM through `scripts/launch_vllm.py`. That launcher adds `--api-server-count`
       (from the CPUs: 3 on this 64-CPU server) and `--renderer-num-workers 2`, and sets `OMP_NUM_THREADS=1`,
       because vLLM's stock front end serializes prepare-data's rendering. htr-train's `dspark` job calls
       `vllm serve` itself with neither, puts both on one GPU, and passes `--num-workers 2` (the default is 12).
       Their DSpark data ablation (`docs/research/dspark-qwen3-4b-data-ablation-meganeflynn.md`) finds steps
       trained matter more than which data, so the epochs stay at 3.
     - GPU 2 hung twice with Xid 8 in minicpm-all, and vLLM now runs there. The script starts a stopped training
       again, up to 10 times, after restarting vLLM if it is down. A small process holds GPU 1 from the switch to the
       end, so the queue never sees both GPUs free between attempts.
     - **Checkpoints every tenth of an epoch, not `--save-best`.** With `--save-best` speculators saves only an epoch
       that improves validation loss, and never mid-epoch: `maybe_save_checkpoint` returns early, and mid-epoch saves
       require `not self.config.save_best` (`speculators/train/trainer.py` at `560f046`). A hang would then cost the
       epoch in progress, about 3.5 h. `--checkpoint-freq 0.1`, as their Kimi-K3 DSpark example uses, saves ten times
       an epoch and resumes at the exact step, so a hang costs at most about 20 min. Their train CLI page still says
       the flag is an integer ≥ 1; the code takes a fraction of an epoch. `checkpoint_best` is still linked to the
       epoch of least validation loss, with its `val_metrics.json`, which the publish reads.
     - **Read in full, at `560f046`:** every page under `docs/` (tutorials, CLI reference, algorithms, features,
       hyperparameters, losses, research notes) and the DSpark, DFlash best-practices, Kimi-K3 DSpark and DSpark
       evaluation examples, and `scripts/launch_vllm.py`. Points that bear on this draft:
       - The answers must come from the reader with the sampling it is served with (`prepare-data` page): ours are
         greedy (`dspark-set`, temperature 0), as the reader's card asks.
       - For a multimodal target, generate the answers with a multimodal-capable workflow and give `prepare-data` the
         conversations (response regeneration page), as `dspark-set` does.
       - The reader has 24 layers, and `mtp_num_hidden_layers: 0`, so MTP fine-tuning (their route for Qwen3.5
         models with MTP heads) does not apply; DSpark's 15 draft tokens must equal its block size, as served.
       - The reader is hybrid (linear attention): with a draft, vLLM keeps one copy of the recurrent state per draft
         token, so each request costs more memory and fewer fit in a batch (throughput vs. interactivity page). The
         draft card says to measure at the concurrency one reads at.
       - "Long prompts with short answers may leave little decoding time to save" (RAG page): our lines.
       - `runs/htr/verify/check_dspark.sh` now follows their evaluation rules: prefix caching off on both servers,
         acceptance length (1 + accepted / drafts), and a second pass 64 at a time beside one at a time.
     - **Switched 9 Nov 09:51.** Preparation saved its data at 09:50 (two Arrow shards and the token counts); vLLM came
       up on GPU 2 at 09:51:22. The trainer runs on GPU 1, 96–100 % busy. Its tenth-of-an-epoch checkpoints fall every
       4,485 steps, 22 min apart (10:16, 10:38, 11:00), so an epoch is 44,850 steps (6.7 samples, about 3,800 tokens a
       step) at 3.4 steps a second, about 23 samples a second, 3.7 h. Three epochs end about 21:00, plus each epoch's
       validation. (An estimate at 10:00 of 37,800 steps at 2.8 a second was wrong in both figures; the hours stand.)
     - **The loops over 4,097 tokens are dropped.** The reader's looping answers (about 40, at the 4,096-token cap) with
       their image tokens exceed vLLM's `--max-model-len` 4,097: vLLM refuses them ("Input length (4266) exceeds"),
       and the trainer drops each after its retries (`failed_samples=1`), training on the rest.
     - **Another user's job on GPU 2 (9 Nov 18:33).** marerle started a 10 GB sampling sweep on GPU 2 beside our vLLM
       (training 3.3 → 3.1 steps a second). The queue's `gpus_free` counted every process on GPUs 1,2, so Jev would
       have waited for it. At 20:24 the waiting queue was stopped by PID and started again (its `.done` markers skip
       what is done) with `gpus_free` counting only our processes and asking `NEED_GB` (60) free on each GPU; Jev
       peaked at 21.4 GB a GPU in the 2-GPU smoke run at 16 a GPU (32 now). The draft check's two servers take 80 % of
       their GPUs (`--gpu-memory-utilization 0.8`), as vLLM's default 90 % no longer fits beside the 10 GB. The queue's
       draft step now serves on the first GPU and trains on the second, as dspark_split.sh does.
     - **Done 9 Nov 21:55, published 22:10** (`Gabriel/diffu-minicpm-ocr-htr-dspark`, private): epoch 3 the best on
       held-out answers. Loss 0.168 / 0.132 / 0.115 and acceptance length 6.76 / 7.98 / 8.65 after epochs 1 / 2 / 3,
       still rising, but by less each epoch. TensorBoard log from the console log pushed to the repo's `runs/`.
     - **Three-way test** (`runs/htr/verify/bench_three.sh`, 22:15–22:34; results in `runs/htr/bench/`). One RTX PRO 6000
       per model, prefix caching off, greedy; 1,000 test lines (the first of test5000) and 200 dev regions; a
       placeholder process on GPU 1 kept the queue's Jev back until it ended. Read per second:

       | | at a time | TrOCR | reader alone | reader + draft | speedup |
       |---|---|---|---|---|---|
       | lines | 1 (TrOCR batch 1) | 4.31 | 13.28 | 18.17 | 1.37× |
       | lines | 64 (TrOCR batch 32) | 6.54 | 74.61 | 82.17 | 1.10× |
       | regions | 1 | – | 1.95 | 4.08 | 2.09× |
       | regions | 64 | – | 12.35 | 13.21 | 1.07× |

       Acceptance length in vLLM: 7.5 on lines, 9.7 on regions. Corpus CER, lines: TrOCR 4.48 %, reader 4.99 %, with
       the draft 4.97 %; regions: 4.28 % and 4.27 %. 964 of 1,000 line readings are identical with and without the
       draft, and the reader alone is identical to itself at 1 and 64 at a time for 959: the differences are GPU
       rounding, not the draft. The draft card on the Hub carries both tables. `bench_three.sh`'s summary printed
       the acceptance length as nan: vLLM names the counters `*_total` and the script kept the suffix. Computed by hand
       from its `.counters` files.
     - **Jev started 22:45** on GPUs 1 and 2 (the queue, as soon as the placeholder ended).
     - **A speculators training stopped by a signal exits 0.** On SIGTERM it saves `output/interrupted` and returns
       success, so `dspark_first.sh` took the stopped launch for a finished one: it wrote `dspark_train.done` and
       then failed to publish (no `checkpoint_best`). The marker was removed at 09:53, so the queue would still
       train the draft if this run failed. A wrapper cannot tell a signalled stop from a finished training by its
       exit code: never stop the trainer by signal while `dspark_split.sh` waits on it, since it would go on to
       publish. A GPU hang raises inside the trainer and exits non-zero, which its retry loop catches.
     - **htr-train's `dspark` job now renders this layout** (9 Nov): vLLM on the run's first GPU and the trainer on its
       last (one GPU: both on it), vLLM's front end and prepare-data's workers from the run's CPU budget as
       `launch_vllm.py` computes them, a checkpoint every tenth of an epoch. `configs/dspark_all.yaml` asks for two
       GPUs and this server's 64 CPUs (3 API servers, 12 workers) with 8 loader workers. `after_minicpm.sh` still
       launches the draft on the first GPU alone: it is edited once the queue ends, since bash reads a running
       script as it goes; until then its draft step would be refused (two GPUs asked, one named), and it is skipped
       anyway once `dspark_split.sh` leaves its marker. (Done 20:24: the queue was restarted with that edit.)
     - **Done 9 Nov 22:10: `Gabriel/diffu-minicpm-ocr-htr-dspark`** (private), the end of epoch 3 (folder `2`), the
       best on held-out answers. Held out after each epoch: loss 0.168 / 0.132 / 0.115, acceptance length 6.76 / 7.98
       / 8.65, first guess right 92.5 / 94.2 / 95.1 %, fifteenth 65.4 / 74.1 / 78.6 %. Still improving at the end, by
       less each epoch: a longer run (5–6 epochs on the same data) is the next step if regions matter.
     - **Speed and accuracy (`runs/htr/verify/bench_three.sh`, 22:15–22:33),** the first 1,000 test lines and the 200
       dev regions, each server on its own GPU, prefix caching off, OpenBMB's greedy settings; lines a second:

       | | one at a time | 64 at a time |
       |---|---|---|
       | lines: Swedish Lion TrOCR (float32) | 4.31 (batch 1) | 6.54 (batch 32) |
       | lines: reader alone | 13.28 | 74.61 |
       | lines: reader + draft | 18.17 (1.37×) | 82.17 (1.10×) |
       | regions: reader alone | 1.95 | 12.35 |
       | regions: reader + draft | 4.08 (2.09×) | 13.21 (1.07×) |

       Acceptance length in vLLM 8.72 over all of it (8.75 in `check_dspark.sh`'s 330 asks). Corpus CER with and
       without the draft within 0.04 points (lines 4.97–5.01 %, regions 4.27–4.31 %); TrOCR 4.48 % on the same
       lines. With the draft 96–97 % of line and 75–77 % of region readings are identical to the reader's alone, and
       the reader alone is identical to itself between 1 and 64 at a time just as often (959 of 1,000, 155 of 200):
       batch-size arithmetic on near-tied tokens, not the draft. The card on the Hub carries both tables, setup and
       use, and TensorBoard curves (`runs/`); the raw output is in its `eval/`.
     - **Jev held back for the test:** a placeholder process of ours on GPU 1 kept the queue's gate closed until the
       test ended (22:33:49); `jev_train` started 22:34:41. marerle's sweep on GPU 2 ended about 21:31.
5. **Train Jev** (`configs/jev_all.yaml`, both GPUs), then publish it as `Gabriel/diffu-minicpm-ocr-htr-jev`.
   **Done 11 Nov 16:44:** 45,486 steps of 128 in 41.7 h (2.3 s a step, 23 GB a GPU), `checkpoint-45486` published
   (private) and checked from the Hub (it loads as its card says; the renderer redraws all 2,000 stored renderings).
   - **Dev pairs** (2 % of the training lines' pairs): accuracy 87.2 % at step 1,000, 95.6 % at the end, flat from
     about step 40,000 (loss 0.312 → 0.132).
   - **Test pairs** (2,000 drawn from `jev_lines.test.jsonl`; `runs/htr/verify/check_jev.py`, then
     `jev_by_origin.py` on its saved scores): AUROC 0.870, 77.8 % judged right at 0.5. **By kind of candidate the
     picture splits:** the correct text against rule-made wrong candidates (perturbations, another line's text) AUROC
     0.988 (96.9 % of the correct kept, 93.9 % of the wrong caught); against the reader's own wrong readings AUROC
     **0.624**, 6.6 % caught. Those misreadings are small (median CER 5 %) and were 35,578 of the 3.0 M training pairs
     (1.2 %): the reader had trained on those lines, so it misread few of them. A threshold of 0.97 catches 74 % of
     them and rejects 55 % of the correct texts.
   - **OpenBMB's checkpoint on the same pairs,** given our prompt and rendering (not its own): AUROC 0.594; it scores
     nearly every pair near 0. That is the starting point of the fine-tuning, not a measure of their model.
   - **So this Jev is a gross-mismatch detector, not yet a proof-reader of the reader.** A next one needs real
     misreadings in number: readings of training lines by a reader that has not seen them (folds, or an early
     checkpoint), TrOCR's readings, the reader's second-best readings.
   - Its card has the table, the limit, setup and a transformers example (scores within 0.01 of vLLM's on 60 pairs);
     TensorBoard curves are in the repo's `runs/`.
6. **A longer draft (11 Nov 17:43, `runs/htr/dspark_ft.sh`, tmux `htr:dsparkft`).** The published draft was still
   improving after its 3 epochs (held-out acceptance length 6.76 / 7.98 / 8.65), so it is trained 3 more, started from
   its weights (`--from-pretrained`, into `runs/htr/dspark-ft`), at a learning rate of 3.75e-5: speculators' own study
   of training a trained DSpark draft further (docs/research/dspark-qwen3-4b-code-finetune-lr-ablation, at the
   installed commit) found the from-scratch rate made it worse and about an eighth of it over 3 epochs best. Same data
   and layout (vLLM on GPU 2, trainer on GPU 1). Not published from the script: it must beat the published draft on
   lines and regions in vLLM first (their study found in-domain validation an unreliable judge). Ends about 05:30 on
   12 Nov. Paused 3 minutes at 19:50 (SIGSTOP, then SIGCONT) for clean timings; it went on from step 25,163.

Controls:
- A file `runs/htr/HOLD` holds the trainings back.
- `DSPARK=0` stops before the draft.
- A failed upload is logged and the queue goes on; running the script again retries it.
- Repos are private unless `HF_PUBLIC=1`.

## Measured so far

### Final score: checkpoint-28000 against TrOCR (the queue, 8 Nov 22:03)

`runs/htr/final/result.txt`; vLLM with OpenBMB's flags, greedy, at most 512 tokens a line; TrOCR greedy.

| | Ours | TrOCR |
|---|---|---|
| 5,000 `test_0050` lines: mean / corpus CER, as read | 5.22 % / 5.64 % | **4.17 % / 3.58 %** |
| the same, each looping tail cut | 4.46 % / 3.92 % | 4.17 % / 3.58 % |
| the same, `normalize()` on both sides | 4.42 % | **3.87 %** |
| 6,358 out-of-domain lines: mean / corpus CER | 7.21 % / 5.55 % | **6.67 % / 5.30 %** |
| speed, 5,000 test lines | **68.6 lines/s** (vLLM, 64 at a time) | 7.9 lines/s (transformers, float32, batch 32) |

- **TrOCR leads in domain and out of domain.** In domain by 0.34 points of corpus CER with loops cut; out of domain by
  0.25 (on the 500-line draw the two were level).
- **Three test lines loop:** dotted leader lines ("1: Ducath . . . . ."), where the model writes dots up to the token
  limit (1,025 characters). They alone take corpus CER from 3.92 % to 5.64 %. A guard at serving (stop a repeated
  " ." run, or cut it as `cut_loop` does) removes them.
- **We read about 9 times as fast,** though the two ran on different stacks.

### The in-training curve

At each eval the model in training reads, greedily, 1,000 `test_0050` lines and 500 out-of-domain lines, among other
sets; the readings are in `output/cer/step-<n>.<set>.jsonl`. CER is the per-line mean, raw text. The `normalize()`
column applies TrOCR's training normalisation to both sides; it changes the test CER by under 0.1 points.

| Step | eval loss | test CER | test CER, `normalize()` | OOD CER | dev regions CER |
|---|---|---|---|---|---|
| 15,000 | 0.2442 | 5.19 % | 5.13 % | 6.99 % | 4.56 % |
| 18,000 | 0.2349 | 5.00 % | 4.93 % | 7.53 % | 8.79 % |
| 21,000 | 0.2275 | 5.02 % | 4.97 % | 6.83 % | 4.46 % |
| 24,000 | 0.2235 | 4.91 % | 4.87 % | 7.07 % | 5.08 % |
| 25,000 | 0.2290 | 5.00 % | 4.95 % | 6.80 % | 4.92 % |
| 26,000 | 0.2272 | 4.84 % | 4.79 % | 7.04 % | 5.21 % |
| 27,000 | 0.2261 | 4.78 % | 4.73 % | 7.03 % | 5.41 % |
| 28,000 | 0.2269 | 4.77 % | 4.72 % | 7.14 % | 5.07 % |

- **Eval loss** still falls, but more slowly: down 0.0093 over steps 15-18k, 0.0074 over 18-21k and 0.0040 over 21-24k.
- **Test CER** gains about 0.04 points a 1,000 steps.
- **OOD CER** is flat within noise.
- **Dev regions CER** jumps at steps 14,000, 18,000 and 22,000. At each of those evals one of the 200 regions looped to
  the token limit, and at no other eval.

### Is it overfitting? (8 Nov, step 27,000)

**It has started to memorise, but held-out reading is not worse yet.** Over 1,000-step windows:

| Steps | Train loss | Eval loss | Eval token acc. | Dev lines CER | Test CER | OOD CER |
|---|---|---|---|---|---|---|
| 23-24k | 0.209 | 0.2235 | 0.9443 | 4.16 % | 4.91 % | 7.07 % |
| 24-25k (epoch 3 begins at 24,454) | 0.185 | 0.2290 | 0.9437 | 4.22 % | 5.00 % | 6.80 % |
| 26-27k | 0.167 | 0.2261 | 0.9444 | 4.15 % | 4.78 % | 7.03 % |

- **The memorising.** Train loss fell 20 % within 250 steps of the epoch boundary (0.211 to 0.170), while every
  held-out measure stayed put. That is the model recognising samples it has trained on.
- **The eval loss.** It rose 0.005 and is falling back, with token accuracy flat. The loss is the cross-entropy of
  the dev text, so it grows when the model is more sure of the tokens it gets wrong, without any more of them being
  wrong.
- **What the held-out sets say.** By the per-line mean, dev lines and OOD looked flat since step 21,000. That was
  wrong: a few short lines swing the mean (at step 28,000 five lines made 0.22 of a 0.28-point drop). By corpus CER
  (edits over characters) dev lines still gain about 0.02 points a 1,000 steps (3.43 % at 21k, 3.25 % at 28k) and test
  lines 0.018 (4.27 % to 4.05 %); OOD is flat within noise. So the run continues to its end.

**Resampling the collections now: considered and dropped (8 Nov).** Stopping at step 29,000 and continuing on the lines
drawn by collection size^0.7 (repeats capped at 4) was prepared, then dropped before it ran:
- The method is standard in language-model pretraining ([Chen et al., 2026](https://arxiv.org/pdf/2604.16380)), but
  no 2026 work tests it for HTR, and the 2026 advice is to tune the mix with small repetition-controlled runs
  ([Zhou et al., 2026](https://arxiv.org/abs/2606.07597)), not to take a default.
- That we lose in the small collections rests on 1,000 test lines (11 of GT-FNA_1500); "Next", item 1, comes first.
- The 1,000 dev lines that would have judged it hold no GT-FNA_1500 or landskapshandlingar line (37 lines of
  collections under 20k), so it could not have shown whether it worked. GT-FNA_1500 is one volume, so a whole-volume
  dev split cannot hold any of it out; testing a mix needs a page-level split there.
- **The learning rate** is down to 8e-7 (16 % of the peak) and still falling. What the rest of the run can change is
  small either way.

**The data order was uneven.** Two early resumes (from checkpoints 3,000 and 6,000) ran with `--ignore_data_skip true`.
In transformers an epoch ends only when its pass over the data does, so each of those resumes restarted epoch 1's
shuffle at its first batch. The resumes since (14,000, 15,000, 21,000, 23,000) skip the batches already trained.

How it came about, and why it was caught late:
1. **5 Nov, step 3,000.** The regions were swapped for their masked versions, and the order was restarted on purpose
   ("the data order starts afresh"). The masked file holds the same 83,072 regions in the same order, so with the same
   seed the shuffle was the identical one. This was not checked.
2. **6 Nov, step 6,000.** The pause for a CER check copied that resume, flag included.
3. **6 Nov, step ~10,500.** The train loss jumping at 9k gave it away. `resume.sh` was written without the flag, and
   was taken then for one replay (6-9k), not two. The warning had been visible since about step 7k: train loss falling
   much faster than eval loss, which rose at 8k.
4. **Recorded only in chat,** so after a summary it was lost. On 8 Nov the `lion_data_prep` README was written telling
   to resume *with* the flag (corrected the same day), and the replay was found again from the losses above.

| Steps | What they trained on |
|---|---|
| 0-9,000 | epoch 1's first 3,000 batches, three times over |
| 9,000-14,000 | epoch 1's batches 3,000-8,000 |
| 14,000-24,454 | epoch 2's batches 1,773-12,227 |
| 24,454-36,681 | epoch 3, all of it |

- **It explains every jump in the train loss.** It fell at steps 7-9k (a third pass: eval loss rose 0.303 to 0.309 then,
  overfitting on those 384,000 samples). It jumped up at 9k (unseen batches). It dropped at 14k (epoch 2's shuffle,
  65 % of it seen before). It dropped again at 24,454.
- **Exposure before epoch 3,** counting each shuffle once: 5 % of the samples in neither, 39 % in one, 56 % in both. A
  quarter of the samples (epoch 1's first 3,000 batches, 384,000 samples) count three times in epoch 1's.
- **Fixed for the next runs (8 Nov):** htr-train's `steps.sh` resumes a training from its newest whole checkpoint
  whenever it runs again, never with `--ignore_data_skip`; `train.sh` and `after_minicpm.sh` start a stopped training
  again up to 10 times (`trained`). Jev is the first run to use it; minicpm-all keeps its own loop (`resume.sh`).

### Against TrOCR on the same lines (step 24,000)

TrOCR's greedy readings come from `runs/htr/ckpt_eval/test_trocr.jsonl` and `runs/htr/eval/ood_trocr_greedy.jsonl`.

| | Ours | TrOCR | Best of the two, per line |
|---|---|---|---|
| 1,000 test lines: mean CER | 4.91 % | **4.54 %** | 3.24 % |
| 1,000 test lines: corpus CER | 4.21 % | **3.60 %** | |
| 500 OOD lines: mean CER | 7.07 % | 7.01 % | 5.36 % |
| 500 OOD lines: corpus CER | **5.55 %** | 5.63 % | |

- **Out of domain the two are level. In domain TrOCR leads**, by 193 edits over the 1,000 lines.
- **At step 26,000** (same lines): test mean 4.84 % against 4.54 %, corpus 4.13 % against 3.60 % (169 edits behind); OOD mean 7.04 % against 7.01 %, corpus 5.40 % against 5.63 %; best of the two 3.16 %. At 27,000: test mean 4.78 %
  against 4.54 %, corpus 4.10 % against 3.60 % (160 edits behind); OOD level (7.03 % / 5.60 % against 7.01 % / 5.63 %).
- Per line, on the test lines, we are worse on 231, better on 162 and the same on 607.
- **The gap is in the letters, not the conventions.** With case, punctuation, `¬` and spaces ignored, the lead grows
  (ours 4.14 % / 3.67 % mean / corpus against TrOCR's 3.25 % / 2.69 %).
- **The two readers err on different lines.** The best of the two per line is far below either reader alone. A
  verifier choosing between the two readings could approach that floor; see "Next", item 5.

### Where we lose: by collection

Corpus CER on the 1,000 test lines, our training lines from the training set:

| Collection | Training lines | Test lines | Ours | TrOCR | Extra edits |
|---|---|---|---|---|---|
| GT-FNA_finnish_court_records | ~146,000 | 98 | 4.23 % | 3.12 % | +42 |
| GT-FNA_1500 | 13,271 | 11 | 17.27 % | 9.81 % | +35 |
| GT-RA_landskapshandlingar | 19,214 | 23 | 13.96 % | 7.78 % | +27 |
| GT-SSA_sodra_forstadens_kamnarsratt | 58,521 | 32 | 4.87 % | 3.29 % | +13 |
| GT-RA_goteborgs_poliskammare | 312,769 | 177 | 2.68 % | 2.55 % | +7 |

- Over 19 collections, the fewer training lines a collection has, the larger our gap (Spearman -0.44).
- TrOCR saw the small collections 6 times; we see them 3 times by the end of this run.
- Large collections are level, and in some we lead.
- **Caveat:** these samples are thin (11 lines of GT-FNA_1500). Confirm on all 88,317 test lines before acting on them
  ("Next", item 1).

### Can TrOCR's batch output be trusted as labels? (the agreement filter)

Where our reading and TrOCR's are identical, how good is TrOCR's label (its CER against the human reference)?

| Lines | Kept | TrOCR label CER, kept lines | TrOCR label CER, all lines |
|---|---|---|---|
| 1,000 test | 55.2 % | 1.34 % | 3.60 % |
| 500 OOD | 40.0 % | 1.89 % | 5.63 % |

- So agreement alone cuts label noise by about two thirds, but drops about half the lines.
- Even identical readings were wrong on 124 test lines and 71 OOD lines: both readers make some of the same mistakes.
- Agreement says nothing about line detection. A page whose lines were cut badly gives bad text that both readers may
  read alike. Detection needs its own page-level check ("Next", item 3).

### Speed: why a step takes 11-12 s

Measured 4 Nov with the same batch shape, 2 GPUs; s/it is seconds a step on normal batches. The stress test runs the
longest regions.

| Setting | s/it | Stress test |
|---|---|---|
| Checkpointing in both (chosen) | 11.91 | fits (70.7 GB) |
| No checkpointing in the vision encoder | 10.67 | out of memory |
| No checkpointing in the language model | 10.2-10.7 | out of memory |
| No checkpointing, ZeRO-2 | 9.8 | out of memory |
| No checkpointing, DDP | 7.13 | out of memory |

- Grouping batches by length ran out of memory.
- `torch.compile` was 10x slower.
- The cards are Max-Q (300 W cap).
- Packing would be about 7x faster but is not allowed (see "Rules of the machine").

### The data itself (8 Nov, on the CPU while training)

One row per line for all 1,594,072 lines (train, dev, test) in `runs/htr/analysis/lines_v2_flags.parquet`. Each row
holds its meta, its text and measures of its crop: the paper's level, the amount of ink and where it runs, and an image
hash. Our model's readings at steps 15k-25k sit beside it.

**What the training set is made of**
- 57 collections, very unequal:
  - Göteborgs poliskammare alone is 21 % of the training lines.
  - The 5 largest collections hold 49.5 %; the 30 smallest hold 4.8 %.
- The test set mirrors that mix (Göteborg 20.6 %). So the test mean mostly measures the large collections, and the
  small ones barely move it.

**The remaining errors are persistent: 86 % sit on lines wrong at every eval from step 15k to 25k**

On the 2,000 test and dev lines read at each eval:

| Category | Lines | Share of errors at step 25k |
|---|---|---|
| Right at every eval | 40 % | 0 % |
| Wrong at every eval | 35 % | 86 % |
| Flipping | 15 % | 8 % |
| Getting worse | 3 % | 5 % |
| Improving | 7 % | 2 % |

The worst 10 % of lines hold 52 % of the errors. More steps on the same data will not fix these lines; the data has to
change. Lines wrong at every eval are most common in the small collections.

**What the errors are**

From the 2,090 edits on those 2,000 lines, and 48 of the worst or always-wrong lines checked by eye on contact sheets
(`runs/htr/analysis/sheets/`):

- **Scarce old hands** give the worst lines. GT-FNA_1500 and landskapshandlingar are 16th-century, with old spelling,
  `ø`, `·` separators and Roman numerals. CER is about 6.2 % in collections under 20k training lines, against 2.7 % in
  the largest.
- **The model's language prior.** It modernises spellings ("sät" to "sätt") and turns rare names into common words
  ("Duhn" to "Datum"). It turns Finnish words in the Finnish court records into Swedish-looking ones ("nimimerkit" to
  "bomurke").
- **Things a scan often cannot settle:**
  - punctuation: 17 % of edits;
  - case only: 8 %;
  - word spacing: 5.5 %;
  - a line-end mark missing or added: 5.9 %. 100 of the 2,000 lines are wrong only in their last character, and some
    of those marks lie outside the crop.
- **Label or crop faults:**
  - text in the label that is outside the crop;
  - a struck-through word transcribed;
  - a line rotated 90°;
  - possibly one misread label ("söisa" where the image shows "sökt bevise").

**Faults in the data, counted over every line**

| Fault | Lines | Where | Checked by eye |
|---|---|---|---|
| Thin-strip polygons: the crop cuts off the tops and bottoms of the letters | 5,638 (5,308 train) | Stockholm magistrat 7.2 %, Påhlmans resejournaler 5.9 %, most others about 1 % or less; 143 pages with over half their lines thin (91 of them Stockholm magistrat) | a flagged page confirmed whole-page |
| Text too long for its ink: label text outside the crop, rotated lines, two lines in one crop | 920 | spread | 3 of 8 real faults |
| Dark, blurred or faint crops | 378 | boskapslängder, riksregistraturet, bergskollegium (microfilm) | 1 of 8 truly empty; the rest hard but valid |
| Cyrillic `ӕ` or Greek `β` for Latin `æ` and `ß` | 1,496 lines | Stockholm city archive (SSA) collections | the same glyph under two code points |
| Five kinds of dash (`—` 42,641, `-` 15,345, `–` 9,390, `―` 169, `‒` 42) | about 67,000 characters | each collection its own convention | |
| No-break spaces, tabs, double spaces | 7,862 / 1,640 / 3,708 lines | tabs mostly jämtlands domsaga | scoring forgives them; training does not |
| Duplicate crops; train/test leaks | none | | exact image hash and text |

**What is worth changing, measured against the whole test set** (88,317 lines, 2.8M characters):

| Change | Its share of test characters | Upper bound on the test CER gain |
|---|---|---|
| Fix the code points (`ӕ`, `β`) | 0.002 % | none; it would even cost, since the test labels keep `ӕ` |
| Re-cut the thin-strip lines | 0.38 % | 0.04 points even at 10 points better on them; the test pages are thin too |
| Drop the lines too dense for their ink | 0.08 % | about 0.01 points. Only 3 of 8 checked were real faults, too few to trust the rate |
| Collections under 20k training lines brought from 6.2 % to 4.3 % CER (the 20-60k tier) | 16.1 % | **about 0.3 points** |

So the faults in the table above are recorded but are not worth fixing for CER. Thin crops matter for the batch
pipeline's detection check, not for this training. What is worth doing:
1. **Draw collections at n^0.7, capped at about 4 repeats.**
   - Göteborg falls from 21 % to 13 % of draws; collections under 20k lines rise from 15 % to 27 %.
   - FNA_1500 and landskapshandlingar are drawn 1.4-1.6 times as often.
   - The cap matters: uncapped, the tiniest collections (126 to 779 lines) would repeat 3.6-6.3 times.
2. **More of the scarce hands than the GT has:** synthetic lines from the diffu-moss hand LoRAs (one exists for
   landskapshandlingar), and curated machine-labelled lines from the batch.

Two things need the GPUs and come with the queue: our model's readings of every test line (in Jev's step), and of
100,000 training lines. The training lines that the final model still cannot fit are the best place to look for label
errors at scale.

### The demo Space, and DSpark decoding in PyTorch (11 Nov)

`Gabriel/diffu-htr-speed` (a ZeroGPU Space): the same dev region read by the Swedish Lion TrOCR (the region as its
text lines, cut along the GT's outlines), by the reader alone and by the reader with its draft (the region as one
image), replayed from the same start at their measured pace on the manuscript and as text; the same region's timings
on our server, where vLLM serves the reader; then Jev's scores of both readers' reading of each line. Its files are
`lion_data_prep/space/`, published by `lion_data_prep/publish_space.sh` (examples cut by `space/build_examples.py`).
The first version's notes come first; what changed the same evening follows them.

- **ZeroGPU cannot host vLLM** (a GPU exists only inside short function calls), so the draft is decoded in PyTorch:
  `space/dspark_hf.py`, written from speculators' DSpark model code and vLLM's DSpark speculator.
- **Profiled (`torch.profiler`):** eager decoding was launch-bound, about 1,500 kernels a reader pass, the GPU busy
  12 % of the time. So each reader pass is replayed from a CUDA graph on transformers' StaticCache, the draft's
  proposal is one graph, and a draft cycle takes one reader pass: the reader's linear-attention states cannot be cut
  back after rejected tokens, so the verifying graph keeps each such layer's block inputs and the states are rebuilt
  over the accepted tokens only. A cache of 4,096 positions, not 8,192: a pass costs 14 % more with the longer one.
  fla's kernels and its fused norms were tried and left out (no gain under graphs; the fused norms changed readings).
- **The same tokens:** the reader alone on graphs writes what transformers' `generate` writes (5 of 5 lines); with the
  draft 19 of 20 lines and 11 of 12 regions are identical, the rest near-tie flips; acceptance length 8.1 on regions.
- **Measured on a quiet GPU** (GPU 2, the draft's training paused for it), the same 12 dev regions and 20 test lines,
  one at a time (`space/check_decode.py`; vLLM as the draft's card serves it):

  | | reader alone | with the draft |
  |---|---|---|
  | regions, vLLM | 5.9 s | 2.2 s |
  | regions, this PyTorch code | 12.4 s | 3.2 s (3.9×) |
  | lines, vLLM | 1.43 s | 1.21 s |
  | lines, this PyTorch code | 2.80 s | 2.19 s (1.3×) |

  In the app, a 17-line region: TrOCR 1.49 s, the reader alone 1.20 s, with the draft 0.36 s. (The vLLM rows were a
  second pass over images its servers had just read: see the processor cache below. They flatter vLLM.)
- **A lesson:** timed beside the draft's training on the same GPU, the PyTorch reader alone took twice as long
  (10.5 ms a token against 5.0), and looked slower than TrOCR; vLLM and TrOCR slowed far less. Time on a quiet GPU.
- **The baseline's weights:** the 2026 Swedish Lion is not on the Hub; a private copy of our download is
  `Gabriel/swedish-lion-trocr-large-handwritten-char`, for the Space.

The same evening (11 Nov, 20:30–21:45):

- **It runs on ZeroGPU.** Without a secret the private TrOCR copy answered 401 at startup and the whole Space stopped
  (`RUNTIME_ERROR`). The app now starts without TrOCR and says so in its lane; with the user's read-only `HF_TOKEN`
  secret it loads all four models. On ZeroGPU (half an RTX PRO 6000 Blackwell) every call gets a fresh worker: a GPU
  after about 3 s, then about 5 s to capture the CUDA graphs. The 10-line region then takes TrOCR 2.10 s one line
  after the other and 1.30 s as one batch, the reader alone 1.24 s, with the draft 0.56 s (0.3 s before the first
  token, then 7.6 ms a token alone and 2.1 with the draft). The two regions compared there read the same with and
  without the draft.
- **One rule for everyone (the user's question).** The first version raced TrOCR on a batch of the region's lines
  against the reader's one sequence, and TrOCR looked as fast as the reader alone. Batching was the whole reason: 8 to
  18 lines decode side by side, while a region is one sequence with nothing to batch (the reader batches whole
  regions, in vLLM). The page now shows three views, each with one rule: the race with one stream each (TrOCR reads
  its lines one after the other); TrOCR's one-batch time as a second bar; and everything batched, from our server
  (TrOCR at batch 32 against vLLM with 64 requests at a time). All bars start together once every reader is timed;
  before, each lane started when its result arrived, and TrOCR, run first, seemed to get a head start.
- **The same regions on our server** (`space/measure_reference.py` → `space/reference.json`, shown under the race):
  GPU 2 with the trainer paused, TrOCR in transformers, the reader in vLLM, one request at a time, medians of 5;
  beside them this PyTorch code on the same GPU (the local app, medians of 3, timed from the decoded image). Seconds:

  | region | TrOCR line by line | TrOCR one batch | vLLM alone | vLLM + draft | PyTorch alone | PyTorch + draft |
  |---|---|---|---|---|---|---|
  | 10 lines, 124 tokens | 1.66 | 0.84 | 0.50 | 0.32 | 0.78 | 0.28 |
  | 8 lines, 76 tokens | 1.25 | 0.60 | 0.38 | 0.27 | 0.53 | 0.23 |
  | 9 lines, 96 tokens | 1.86 | 0.75 | 0.53 | 0.35 | 0.63 | 0.23 |
  | 11 lines, 129 tokens | 1.47 | 0.67 | 0.57 | 0.34 | 0.79 | 0.26 |
  | 17 lines, 209 tokens | 3.92 | 1.54 | 0.69 | 0.33 | 1.21 | 0.38 |
  | 18 lines, 198 tokens | 3.41 | 1.32 | 0.73 | 0.45 | 1.15 | 0.41 |

  vLLM alone is 1.2–2.2× faster than TrOCR's one batch and 2.6–5.7× faster than TrOCR line by line; with the draft
  2.0–4.6× and 4.4–11.7×. With the draft this PyTorch code is as fast as vLLM for one region at a time; alone it is
  1.2–1.8× slower. TrOCR line by line varied by up to 25 % between runs (a CPU-bound loop).
- **vLLM keeps the processed form of an image it has seen** (its multimodal processor cache, on by default). One
  request at a time, a region it had seen came back 0.05–0.1 s sooner on a quiet GPU (0.11–0.2 s beside the
  training). So the reference sends every request with one pixel, drawn at random, changed by one level; a first
  version changed fixed pixels, and its second run on the same servers was cache-assisted and 15–20 % too fast.
  That one-level change also flips a near-tied token now and then (2 or 3 readings of a region over 6 requests).
- **Do the 9 Nov "64 at a time" numbers stand? Yes.** They were read on servers that had just read the same images
  one at a time. `runs/htr/verify/bench_cold.sh` (both servers on GPU 2, asked in turn, the trainer paused) read the
  same 1,000 lines and 200 regions 64 at a time, first new to the servers, then again:

  | per second | new | seen | 9 Nov (seen, a GPU each) |
  |---|---|---|---|
  | lines, reader alone | 67.3 | 68.9 | 74.6 |
  | lines, with the draft | 77.2 | 84.6 | 82.2 |
  | regions, reader alone | 11.9 | 11.7 | 12.4 |
  | regions, with the draft | 12.7 | 12.7 | 13.2 |

  At 64 at a time the GPU is the limit, and the cache changes little. The page quotes the "new" column.
- **Near ties:** on GPU 2 the PyTorch reading with the draft differed from the reading alone by one token in 4 of
  these 6 regions ("dal.r" against "d.r", "Pohhieth" against "Pohlieth"); the page says so when it happens.
- **The page:** one region box (an example from a three-column `gr.Examples` table, or an upload; the app tells them
  apart by the file's bytes), and Jev's lines folded away under a sentence that reads them (both right, only one
  right and which one Jev scored higher, Jev's mean score on right and on wrong readings), with column headers and
  the characters that differ from the ground truth marked. On a wide page the bars stand beside their summary, our
  server's timings beside the batched rates, and each of Jev's lines beside its readings. On ZeroGPU the 17-line
  region: TrOCR 4.99 s line by line and 2.09 s as one batch, the reader 1.89 s alone and 0.76 s with the draft.
- **Public since 21:55** (the user's choice); the TrOCR copy stays private, read through the Space's secret.
- **Not done:** ahead-of-time compilation (AOTInductor, the route ZeroGPU supports), which could close the reader
  alone's gap to vLLM and shorten the 5 s a fresh worker spends on graphs. SGLang: OpenBMB's guide serves the reader
  and a draft through a pull request still open in October 2026, built from source with CUDA 13; not measured.

### Throwaway analysis scripts

In `runs/htr/analysis/` (on disk, not in git). Once a curation step is chosen, it moves into `lion_data_prep`.
- `features.py OUT.parquet WORKERS`: the line table (about 9 min on 24 low-priority workers; training did not slow).
- `sheet.py TABLE FLAG OUT.png`: contact sheets of flagged lines.
- `align.py`: the edit operations of a reading.
- `compare_trocr.py OURS.jsonl TROCR.jsonl`: by collection, by line, by edit operation.
- `agreement.py OURS.jsonl TROCR.jsonl`: the agreement bands.
- `loop_guard.py`, `cer_probe.py`.

Run them with `packages/htr_train/envs/minicpm/.venv/bin/python`.

## Next, in order

The order the user set: finish this run, then Jev, then the DSpark draft, each published to the Hub. That chain is
queued above. Then:

1. **Every test line, by collection and by page.**
   - TrOCR reads the 83,317 test lines it has not read yet (about 3.3 h on one GPU).
   - Ours reads all 88,317 (step 1 of the queue reads only 5,000).
   - Then CER by collection and by page, with each bad page's reason (hand, layout, scan, label).
   - This confirms or kills item 2.
2. **Upsample where we are weak.**
   - A short continued fine-tune from the final checkpoint, the small weak collections repeated about 3 times, low
     learning rate.
   - Measure each collection, and out of domain to catch forgetting.
   - About a day on two GPUs.
3. **A curation pipeline for the batch output** (about 1 million documents, PAGE/ALTO from a line detector and TrOCR):
   1. A page-level detection check: compare the batch's lines with the GT pages that the batch also processed; compare
      the region read whole with its lines joined; check ink coverage and geometry.
   2. A line filter: the agreement filter, then Jev.
   3. Coverage counted per collection.
   4. A human-review sample.
   5. A mark in the set builder for machine-labelled lines.
4. **Pretrain on the curated batch, then fine-tune on the human GT.**
   - Synthetic pages in the weak hands can join the pretraining: the user's diffu-moss hand LoRAs, e.g.
     `Gabriel/diffu-moss-hand-landskapshandlingar`. Their labels are exact.
   - The GT fine-tune comes last, so the last labels the model learns are clean. Keep a share of the pretraining data
     in the fine-tune against forgetting.
   - Evidence: [Kiessling, Sep 2026](https://arxiv.org/abs/2609.20064) pretrained on about 6.2M real and about 1M
     synthetic lines. Fine-tuning that on 25 % of the target collection beat training from scratch on all of it, on
     all seven test sets. It also saw knowledge that the fine-tune data lacked fade.
5. **Jev picks the better of our reading and TrOCR's.**
   - Best of the two is 3.24 % against TrOCR's 4.54 % on the test lines, and 5.36 % against 7.01 % out of domain.
   - Measure how much of that Jev recovers, once Jev is trained.
6. **How much faster the draft makes decoding** (served lines and regions, with and without it). The model card says
   it has not been measured.
7. **Research track:** self-supervised pretraining of the vision encoder on unlabelled scans.
   [Garrido-Munoz and Calvo-Zaragoza, NeurIPS 2026](https://arxiv.org/abs/2609.35473): masked-pixel pretraining gave
   the lowest CER on every benchmark, and it is the only kind tested that improved with real handwriting.

A 4th epoch of the same data has the weakest case: the curve is flattening, and item 2 targets the small collections
more cheaply.

## Waiting on the user

- Where the batch output (PAGE/ALTO) and its scans are, and whether `volym_batch_lista.db` lists that batch.
- Whether to add item 5 (Jev choosing between the two readings) to the queue, after Jev.

## Sources

Only 2026 work is cited here, by the user's rule: the field moves fast enough that older results do not settle choices
here.
