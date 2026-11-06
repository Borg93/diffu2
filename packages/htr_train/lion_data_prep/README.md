# GT-Lejonet: sets and training to beat the TrOCR baseline

The Riksarkivet GT-Lejonet HTR export (`GT-Lejonet_HTR_202602`: ~60 collections, ~30,500 scans with PAGE XML) and
the recipe our readers are trained on it with, so that they compare one to one with the TrOCR model trained on the
same GT ("swedish-lion": `microsoft/trocr-large-handwritten` fine-tuned, its code in RA's own repository). The
mechanisms are general options of `diffu-reader-set` and `htr-train`; what is specific to this dataset and this
comparison is the choice of them, here.

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
`configs/minicpm_all.yaml` and then `configs/jev_all.yaml` with `htr-train launch --where local`, building
Jev's pairs on the CPU meanwhile; each waits until no other process computes on its GPUs. Both scripts leave a
marker a step (`runs/htr/pipeline_markers`), so a second run resumes after the last finished one; a file
`runs/htr/HOLD` holds the trainings back. Run them in tmux: they outlive the login.

`configs/smoke_*_2gpu.yaml` are the same setups in small on the smoke sets, to check a machine first (minutes).

## What the recipe settles, and why

- **Unpacked.** In `envs/minicpm`, Qwen3.5's linear-attention layers carry state from one packed sample into the next
  (a sample packed second differs by 0.82 relative from the same sample alone; the first by 0.03), so packing,
  ~7x faster, would train on mixed lines. OpenBMB trains MiniCPM-OCR unpacked too.
- **4 samples a GPU** for MiniCPM-OCR: 8 peaked at 86 of 96 GB with regions in the batch.
- **Regions at most 9 x 448^2 px**: MiniCPM-OCR sees no more of an image (9 slices of 448^2 px); at the scans'
  resolution the region crops alone would take ~375 GB.
- **CER while it trains** (`cer_sets`): at each evaluation, and once before the first step of a start or resume, the model in training
  reads 1,000 dev lines, 200 dev regions and 500 out-of-domain lines greedily (about 4 min on the two GPUs: 21 s, 188 s and 25 s at step 14,000), logged as
  `eval_cer_<set>` (per-line mean), `eval_cer_corpus_<set>`, `eval_cer_norm_<set>` (the baseline's `normalize()`)
  and `eval_cut_<set>` beside the eval loss; the readings go to `output/cer/step-<n>.<set>.jsonl`. It shows the
  trend; the scores against the baseline come from `htr-eval read` on a checkpoint through vLLM, and the checkpoint
  kept is still the one of least eval loss. The draws are `build_sets.sh`'s `cer_sets` step.
