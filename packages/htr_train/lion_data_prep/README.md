# GT-Lejonet: sets and training to beat the TrOCR baseline

The Riksarkivet GT-Lejonet HTR export (`GT-Lejonet_HTR_202602`: ~60 collections, ~30,500 scans with PAGE XML) and
the recipe our readers are trained on it with, so that they compare one to one with the TrOCR model trained on the
same GT ("swedish-lion": `microsoft/trocr-large-handwritten` fine-tuned, its code in RA's own repository). The
mechanisms are general options of `diffu-reader-set` and `htr-train`; what is specific to this dataset and this
comparison is the choice of them, here. The runs made with it on dmlpai01, their state, the queue and every result so
far against the baseline: [docs/HTR_RUNS.md](../../../docs/HTR_RUNS.md).

## The baseline, as its code trains and tests it

| | TrOCR baseline | here |
|---|---|---|
| test pages | `test_0050` of each volume (5 %, its `splits.py`) | the same: `--gt-test-list test_0050` |
| line image | the line's bounding box, white outside its polygon (`gt.py` `apply_mask`) | the same: `--mask-lines` |
| lines left out | text with `??`, `(?)`, `[` or `]`; a crop under 10 px | the same: `--reject-uncertain --min-line-side 10` |
| empty lines | kept as empty targets from `GT-RA_tomma_sidor` (10 pages) | left out, as every empty line |
| text | trained on its `normalize()`d text (fractions, quotes, dashes, NFKC, ...) | the raw transcription |
| second export of `GT-RA_landskapshandlingar` | included | left out (the same scans twice) |
| out-of-domain test | `Riksarkivet/eval_htr_out_of_domain_lines`, white-masked polygon line crops; no volume of it is in this GT | the same set |
| CER | mean of per-line `jiwer.cer`, beam 4 | the same mean and corpus CER (edits / characters), greedy |
| checkpoint | best validation loss on `test_0050` (its test pages) | best eval loss on a dev split of whole training volumes |

So both readers train on the same pages and are tested on the same pages, cut the same way. Scoring reports CER on the
raw text and with the baseline's `normalize()` applied to both sides, the per-line mean and the corpus CER, on the
`test_0050` lines and on the out-of-domain set. The baseline picked its checkpoint on its own test pages, which
flatters its test score slightly; ours never sees them.

## Reproduce

```bash
# every line and region cut, dev splits, then the training (MiniCPM-OCR, then Jev) on GPUs 1 and 2
GT=/mnt/work/GT-Lejonet_HTR_202602 WORKERS=56 GPUS=1,2 bash packages/htr_train/lion_data_prep/build_sets.sh
```

`build_sets.sh` cuts with every core given (`WORKERS`, ~500 lines/s on dmlpai01's network disk: ~1 h for the
~1.5 M lines) into `data/htr` (`SETS`), then runs `train.sh` (`NO_TRAIN=1` stops after the sets). `train.sh` trains
`configs/minicpm_all.yaml` with `htr-train launch --where local`, then hands over to `after_minicpm.sh`:
- **The model scored:** the checkpoint of least eval loss (linked as `runs/htr/minicpm-all/best`) is read through vLLM
  with OpenBMB's flags on all 6,358 out-of-domain lines and the 5,000 `test_0050` lines. The readings are scored beside
  the Swedish Lion TrOCR's greedy readings of the same lines in `runs/htr/final/result.txt`: as read, and with each
  reading's looping tail cut (`htr-eval cer`).
- **Jev's reads and pairs:** the same model reads 100,000 training lines and every test line, Jev's real errors
  beside its rule ones. Jev's pairs, dev split and test pairs follow on the CPU, about 10 h for the training lines
  (about 148,000 an hour).
- **The draft's responses:** meanwhile, on the second GPU, the same model answers 250,000 training lines and 50,000
  training regions, each a seeded draw: the DSpark draft's responses. A draw of the two together would hold only about
  16,000 regions, and regions' long answers are where a draft speeds decoding up.
- **Jev, then the draft:** `configs/jev_all.yaml` trains on both GPUs, then `configs/dspark_all.yaml` serves the reader on the
  first and trains the draft on the second (`DSPARK=0` stops before the draft). A GPU step waits until none of our processes
  holds the GPUs and each has `NEED_GB` (60) free; another user's small job there does not hold it back.
- **Each model to the Hub:** `publish_model.sh` uploads each model as soon as it is made. The reader goes up after its
  scoring as `Gabriel/diffu-minicpm-ocr-htr` (`READER_REPO`), and Jev and the draft after their training as its
  `-jev` and `-dspark`.
  - Only the weights and configs are uploaded, not the optimizer or trainer state. The card comes from
    `model_cards/` with the run's own numbers, and an upload is refused if a file names a local path.
  - A new repo is private unless `HF_PUBLIC=1`.
  - A failed upload is logged and the pipeline goes on; a second run retries it.
  - `DRY_RUN=1 bash publish_model.sh ...` stages and renders without uploading.

Each waits until no other process computes on its GPUs. The scripts leave a marker a step
(`runs/htr/pipeline_markers`), so a second run resumes after the last finished one; a file `runs/htr/HOLD` holds the
trainings back. Run them in tmux: they outlive the login.

**A training that stops** (on dmlpai01, GPU 2 hangs now and then with Xid 8, "GPU is probably locked") is started
again by `train.sh` (MiniCPM-OCR) and `after_minicpm.sh` (Jev): up to `RETRIES` times (10), 2 minutes apart, each
once the GPUs are free, from the run's own `command.sh`. Running a run's `command.sh` again is also the way to resume
one by hand.
- Its `steps.sh` (htr-train's) resumes from the newest checkpoint that holds its `trainer_state.json`. The trainer
  writes that file last, so a crash while saving leaves a checkpoint without it.
- The trainer then skips the batches already trained. **Never resume with `--ignore_data_skip true`**: it restarts the
  epoch's pass at its first batch, so those batches train again. minicpm-all's resumes from checkpoints 3,000 and
  6,000 did that ([docs/HTR_RUNS.md](../../../docs/HTR_RUNS.md), "Is it overfitting?").
- A run written before htr-train resumed this way (its `steps.sh` without the `RESUME=()` line) is refused: run again,
  it would train from the start into the same folder.

minicpm-all itself was written before that: the loop that keeps it going is in
[docs/HTR_RUNS.md](../../../docs/HTR_RUNS.md), "How it survives".

`configs/smoke_*_2gpu.yaml` are the same setups in small on the smoke sets, to check a machine first (minutes).

**The demo Space** (`space/`, published by `publish_space.sh` as `Gabriel/diffu-htr-speed`, ZeroGPU): one dev region
read with one stream each by the Swedish Lion TrOCR (its text lines, one after the other), by the reader alone and by
the reader with its draft, replayed from the same start at their measured pace; the same region's timings on our
server, where vLLM serves the reader; then Jev's scores line by line.
- `GT=... bash publish_space.sh` stages `space/`, the Jev prompt and renderer from `diffu_htr_train`, the font and the
  example regions (`space/build_examples.py`: the crops from the dev sets, the lines' outlines from the GT's PAGE XML),
  and uploads them; `DRY_RUN=1` stages only.
- ZeroGPU cannot host vLLM, so `space/dspark_hf.py` decodes the draft in PyTorch on CUDA graphs;
  `space/check_decode.py READER DRAFT LINES REGIONS` checks it against the reader alone on a GPU and times both.
- `space/measure_reference.py` times the example regions on a quiet GPU (TrOCR in transformers, the reader in vLLM
  alone and with its draft, each image new to the server) and writes `space/reference.json`, which the page shows
  under its live race; `publish_space.sh` refuses to publish an example that has no timings there.
- The Space reads the private TrOCR copy with a read-only `HF_TOKEN` secret, set in its settings; without it the app
  runs without TrOCR and says so.

## What the recipe settles, and why

- **Unpacked.** In `envs/minicpm`, Qwen3.5's linear-attention layers carry state from one packed sample into the next
  (a sample packed second differs by 0.82 relative from the same sample alone; the first by 0.03), so packing,
  ~7x faster, would train on mixed lines. OpenBMB trains MiniCPM-OCR unpacked too.
- **4 samples a GPU** for MiniCPM-OCR: 8 peaked at 86 of 96 GB with regions in the batch.
- **Regions at most 9 x 448^2 px**: MiniCPM-OCR sees no more of an image (9 slices of 448^2 px); at the scans'
  resolution the region crops alone would take ~375 GB.
- **CER while it trains** (`cer_sets`): at each evaluation, and once before the first step of a start or resume, the
  model in training reads 1,000 dev lines, 200 dev regions, 500 out-of-domain lines and 1,000 `test_0050` lines
  greedily (about 4 min on the two GPUs: 21 s, 188 s and 25 s at step 14,000), logged as
  `eval_cer_<set>` (per-line mean), `eval_cer_corpus_<set>`, `eval_cer_norm_<set>` (the baseline's `normalize()`)
  and `eval_cut_<set>` beside the eval loss; the readings go to `output/cer/step-<n>.<set>.jsonl`. It shows the
  trend; the scores against the baseline come from `htr-eval read` on a checkpoint through vLLM, and the checkpoint
  kept is still the one of least eval loss. The draws are `build_sets.sh`'s `cer_sets` step.
- **Loops cut when scored, both readers alike** (`cut_loop`): greedy decoding sometimes repeats itself to the end
  (`damn. damn. damn.`, `. . . .` until the token limit), so `htr-eval cer` also scores each reading with a looping
  tail cut to one copy. The rule: a unit of 2 to 60 characters repeated 3 or more times over 10 or more characters,
  holding a letter unless the reading hit the token limit. It cut 8 of 43,316 saved readings (ours from steps 6,000 and
  14,000-21,000, and TrOCR's), each closer to its reference afterwards, none further, and none of TrOCR's.
  Published OCR work treats such loops the same way: LightOnOCR detects them and trains them away, and DeepSeek-OCR
  blocks repeated n-grams when serving.
