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
- **State (8 Nov 11:00):** step ~24,950 of 36,681 (epoch 2.04), about 11 s a step, ending around **01:00 Tue
  10 Nov**. First started 4 Nov 23:07.
- **Hangs:** GPU 2 hung twice with Xid 8 ("GPU is probably locked", CUDA error 702), 7 Nov 21:38 and 8 Nov 04:36.
  It is a known, unfixed fault of RTX PRO 6000 Blackwell under sustained load (NVIDIA forum thread 382055: power limits,
  locked clocks and driver changes did not help there). Each time the run resumed by itself from its newest checkpoint,
  losing at most 1,000 steps (about 3 h).

### How it survives

| tmux window | What runs there | What it does |
|---|---|---|
| `htr:16 resume21k` | `runs/htr/minicpm-all/resume_loop.sh` | the running training; up to 3 resumes (1 used) |
| `htr:17 resume_more` | `runs/htr/minicpm-all/resume_more.sh` | once `resume_loop.sh` gives up, 10 more resumes, 2 minutes apart |
| `htr:18 gpuwatch` | `nvidia-smi -l 1` on GPUs 1, 2 | temperature, power, clocks, throttle reasons, once a second, to `runs/htr/minicpm-all/gpu_watch.csv` |

Each resume is `runs/htr/minicpm-all/run_resume.sh`:
- It waits until GPUs 1 and 2 are free.
- It resumes from the newest checkpoint that has its `trainer_state.json`; a crash while saving leaves one without.
- Its log is `runs/htr/minicpm-all/resume_loop.log`.
- When the training finishes it writes `runs/htr/pipeline_markers/minicpm_train.done` and starts the queue below.

### Queued after it: `after_minicpm.sh`

Starts by itself when the training ends. Each step leaves a marker in `runs/htr/pipeline_markers`, so running the
script again resumes after the last finished step.

1. **Score the best checkpoint.** Linked as `runs/htr/minicpm-all/best`; read through vLLM on all 6,358 out-of-domain
   lines and 5,000 `test_0050` lines, beside TrOCR. Result: `runs/htr/final/result.txt`.
2. **Publish the reader:** `Gabriel/diffu-minicpm-ocr-htr` (private).
3. **Jev's sets.**
   - On the first GPU, the reader reads 100,000 training lines and every test line; the pairs follow on the CPU (about
     4.5 h).
   - Meanwhile, on the second GPU, the reader answers 250,000 training lines and 50,000 training regions for the DSpark
     draft (`runs/htr/final/second_gpu.log`).
4. **Train Jev** (`configs/jev_all.yaml`, both GPUs), then publish it as `Gabriel/diffu-minicpm-ocr-htr-jev`.
5. **Train the DSpark draft** (`configs/dspark_all.yaml`, first GPU), then publish it as
   `Gabriel/diffu-minicpm-ocr-htr-dspark`.

Controls:
- A file `runs/htr/HOLD` holds the trainings back.
- `DSPARK=0` stops before the draft.
- A failed upload is logged and the queue goes on; running the script again retries it.
- Repos are private unless `HF_PUBLIC=1`.

## Measured so far

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

- **Eval loss** still falls, but more slowly: down 0.0093 over steps 15-18k, 0.0074 over 18-21k and 0.0040 over 21-24k.
- **Test CER** gains about 0.04 points a 1,000 steps.
- **OOD CER** is flat within noise.
- **Dev regions CER** jumps at steps 14,000, 18,000 and 22,000. At each of those evals one of the 200 regions looped to
  the token limit, and at no other eval.

### Against TrOCR on the same lines (step 24,000)

TrOCR's greedy readings come from `runs/htr/ckpt_eval/test_trocr.jsonl` and `runs/htr/eval/ood_trocr_greedy.jsonl`.

| | Ours | TrOCR | Best of the two, per line |
|---|---|---|---|
| 1,000 test lines: mean CER | 4.91 % | **4.54 %** | 3.24 % |
| 1,000 test lines: corpus CER | 4.21 % | **3.60 %** | |
| 500 OOD lines: mean CER | 7.07 % | 7.01 % | 5.36 % |
| 500 OOD lines: corpus CER | **5.55 %** | 5.63 % | |

- **Out of domain the two are level. In domain TrOCR leads**, by 193 edits over the 1,000 lines.
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

### Throwaway analysis scripts

In `runs/htr/analysis/` (on disk, not in git):
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
- Whether the three Hub repos should be public. They are private by default.
- Whether to add item 5 (Jev choosing between the two readings) to the queue, after Jev.

## Sources

Only 2026 work is cited here, by the user's rule: the field moves fast enough that older results do not settle choices
here.
