# diffu-htr-train

Training our page and line readers from run specs: Unlimited-OCR 2 (the page reader; LoRA or full), MiniCPM-OCR 1.0
(the line and region recogniser; OpenBMB's full recipe), MiniCPM-OCR-Jev (the proof-reader that scores whether a crop
shows a text), a DSpark draft for our fine-tuned MiniCPM-OCR, and two more page readers: LightOnOCR-3-4B (its pages
grounded as LightOn's client asks for them: [lion_data_prep/README.md](lion_data_prep/README.md)) and Nemotron 3.5 Parse
(by our own trainer: no tool has the model). The tools (ms-swift, speculators, vLLM, our `htr-nemotron-parse`) each run
in their own environment under `envs/`; this package is a light workspace member that checks a spec, renders the
tools' exact commands, launches them on the Spark (under `jobs/spark_guard.sh`) or on Slurm, and builds the sets they
train on. The design, with every recipe and pin cited, and the hardware estimates: [docs/HTR_TRAIN.md](../../docs/HTR_TRAIN.md).
The runs on all of GT-Lejonet (what runs, what is queued, the results against the TrOCR baseline):
[docs/HTR_RUNS.md](../../docs/HTR_RUNS.md).

## The environments

Each is its own uv project and lock, excluded from the workspace (as `packages/sglang_moss`): their pins conflict with
each other and with the workspace's (transformers 5.22 vs 4.52.1 vs speculators' <= 5.19 vs vLLM's).

| Env | For | Pins |
|---|---|---|
| `envs/minicpm` | MiniCPM-OCR, Jev | Python 3.12, torch 2.14.1 (cu130), ms-swift @ `ef4953a4`, transformers 5.22.1, flash-linear-attention 0.4.2, peft < 0.21, datasets < 4.8.5, qwen-vl-utils 0.0.14, trackio; groups `multi-gpu` (deepspeed >= 0.19.3, < 0.20) and `x86` (flash-attn 2.8.3, causal-conv1d: packing) |
| `envs/uocr2` | Unlimited-OCR 2 | Python 3.12, torch 2.14.1, ms-swift @ `ef4953a4`, transformers 4.52.1, easydict, addict, peft < 0.21, datasets < 4.8.5, trackio; group `multi-gpu`. Its remote code runs from `baidu/Unlimited-OCR-2@0480966938291445591f69f08a13b041c073a96a` only |
| `envs/speculators` | the DSpark draft | Python 3.12, speculators @ `560f0465` (0.9.0 + the prepare-data render-route fix), torch 2.14.1, transformers overridden to 5.22.1 (speculators pins <= 5.19.0; `minicpm_ocr` is native from 5.21) |
| `envs/vllm` | serving the target for the draft | vLLM at OpenBMB's pin `fa3b9d13` from vLLM's own prebuilt wheels (linux aarch64 and x86_64), transformers overridden to 5.22.1; built and run on the Spark (GB10) |
| `envs/nemotron_parse` | Nemotron 3.5 Parse (`htr-nemotron-parse`) | Python 3.12, NVIDIA's validated Transformers stack for it (its `reference/parse_3_5/transformers.constraints.txt`): torch 2.11.0 (cu130), transformers 5.14.1, timm 1.0.22, accelerate 1.12.0, Pillow 12.2.0, beautifulsoup4 4.13.3; trackio. Its remote code runs from `nvidia/Nemotron-3.5-Parse@72ad4d71dd827135ee581502363e14748008900a` only |

```bash
uv sync --project packages/htr_train/envs/minicpm             # one at a time; --group multi-gpu on a cluster
uv sync --project packages/htr_train/envs/uocr2
uv sync --project packages/htr_train/envs/speculators
uv sync --project packages/htr_train/envs/vllm
uv sync --project packages/htr_train/envs/nemotron_parse
```

htr-train calls each tool by its path, `envs/<env>/.venv/bin/<tool>`, never imports it.

## The sets

A set is one variant of a reader's data, built once and named by the runs that train on it; its
`<stem>.options.json` records how it was built, and each run's `manifest.json` carries the options of every set it read,
so runs can be compared later.

```bash
# lines (MiniCPM-OCR), scaled to one 448 x 448 view, the GT's test pages apart, a dev split of whole volumes
diffu-reader-set --xml ~/Desktop/data/GT --model minicpm-ocr --task lines --max-crop-pixels 200704 \
    --split-from-gt-test-lists --out data/htr/minicpm_lines_scaled.jsonl
htr-train split-dev data/htr/minicpm_lines_scaled.jsonl --share 0.03
# variants: a cap a collection (--max-per-collection 20000 --seed 0), some collections (--collection GT-...,
# --exclude-collection GT-...), native crops (no --max-crop-pixels), regions (--task elements), drawn pages
# (--xml of a diffu-run-export); a set mixed with another holds out its volumes:
htr-train split-dev data/htr/minicpm_regions.jsonl --like data/htr/minicpm_lines_scaled.dev.jsonl
# whole pages for Nemotron 3.5 Parse (each written as its processor scales it, to fit 1664 x 2048), the regions' dev
# volumes held out
diffu-reader-set --xml /mnt/work/GT-Lejonet_HTR_202602 --model nemotron-parse --split-from-gt-test-lists \
    --gt-test-list test_0050 --workers 56 --out data/htr/nemotron_parse_pages.jsonl
htr-train split-dev data/htr/nemotron_parse_pages.jsonl --like data/htr/minicpm_regions.dev.jsonl
# Jev's pairs from the training lines (scaled to the generator's line height), our frozen renderer
htr-train jev-set --lines data/htr/minicpm_lines_scaled.train.jsonl --out data/htr/jev_lines.jsonl
# Jev's test pairs (to evaluate it on), from the test lines
htr-train jev-set --side test --lines data/htr/minicpm_lines_scaled.test.jsonl --out data/htr/jev_lines.test.jsonl
# a draft's responses: the target served by vLLM (htr-train serve-target SPEC prints the command) answers the prompts
htr-train dspark-set --prompts data/htr/minicpm_lines_scaled.train.jsonl --target <snapshot> --limit 300000 --seed 0 \
    --concurrency 64 --out data/htr/dspark_responses.jsonl  # a seeded draw of the set, 64 asked at once
```

### Scoring

`htr-eval` (run from `envs/minicpm/.venv/bin/`: it reads with that environment's torch and transformers) scores the
readers: `set-lines` and `ood-set` give lines to score, `draw` a fixed draw of them, `read` a reader's predictions
and speed (MiniCPM-OCR through vLLM as OpenBMB asks it; the Swedish Lion TrOCR through transformers), `cer` their error
rates as each side scores itself, as read and with each reading's looping tail cut, and `as-reads` a reader's
predictions as Jev's reads. A spec's `cer_sets` has ms-swift read draws like these at each evaluation, with the model
in training (`evaluate/swift_plugin.py`, passed as `--external_plugins` with `--callbacks cer`). Jev's AUROC and a
draft's served acceptance are not commands yet ([docs/HTR_HANDOVER.md](../../docs/HTR_HANDOVER.md), "Scoring").

### Smoke sets

The smoke runs (`configs/smoke/`) train on three small collections, five volumes and about 55 pages, the GT's test
pages apart: a dev split needs at least three volumes. The regions and pages hold out the lines' dev volume; Jev's
pairs come from the lines' training side, so they hold out one of their own.

```bash
GT=~/Desktop/data/GT-Lejonet_HTR_202602
SMOKE=(--collection GT-RA_vase_haradsratt_20260220 --collection GT-SSA_hall-_och_manufakturratten_20260218
       --collection GT-RA_svea_hovratt_koncept_20260220 --split-from-gt-test-lists)
uv run diffu-reader-set --xml $GT "${SMOKE[@]}" --model minicpm-ocr --task lines --max-crop-pixels 200704 \
    --out data/htr/smoke/minicpm_lines.jsonl
uv run htr-train split-dev data/htr/smoke/minicpm_lines.jsonl --share 0.2
uv run diffu-reader-set --xml $GT "${SMOKE[@]}" --model minicpm-ocr --task elements --out data/htr/smoke/minicpm_regions.jsonl
uv run htr-train split-dev data/htr/smoke/minicpm_regions.jsonl --like data/htr/smoke/minicpm_lines.dev.jsonl
uv run diffu-reader-set --xml $GT "${SMOKE[@]}" --model unlimited-ocr-2 --task parse --out data/htr/smoke/uocr2_pages.jsonl
uv run htr-train split-dev data/htr/smoke/uocr2_pages.jsonl --like data/htr/smoke/minicpm_lines.dev.jsonl
uv run htr-train jev-set --lines data/htr/smoke/minicpm_lines.train.jsonl --out data/htr/smoke/jev_lines.jsonl
uv run htr-train split-dev data/htr/smoke/jev_lines.jsonl --share 0.25
```

## Runs

A run is one YAML file (`configs/smoke/`, `configs/full/`): its name, a description of its variant, its job (one of
`minicpm-ocr`, `unlimited-ocr-2`, `jev`, `dspark`, `lightonocr-3`, `nemotron-parse`), its sets and its resources. It
is checked before anything starts: an unpinned Hub model, remote code for anything but Unlimited-OCR 2 and Nemotron 3.5
Parse at their revisions, a test page or a dev volume in training, a missing image, pairs of another renderer, a
draft's responses from another target, a batch that does not divide, packing without flash attention, a LightOnOCR-3
page larger than 2,048 px, packed or sized otherwise than LightOnOCR-3 reads it, and a Spark run of more than its one
GPU or memory are refused.

Nemotron 3.5 Parse (`nemotron-parse`) is trained by `htr-nemotron-parse` (`diffu_htr_train.nemotron_parse`):
transformers' Trainer around the model's remote code, a process a GPU under `torchrun --standalone`, one node. Its
spec's fields are that trainer's options (which parts learn, the layout tokens' weight in the loss, the longest answer,
activations recomputed) and TrainingArguments; it resumes like an ms-swift run, by running `command.sh` again. The why,
the recipe and what is not known yet: [docs/HTR_TRAIN.md](../../docs/HTR_TRAIN.md), "Nemotron 3.5 Parse". On dmlpai01:

```bash
HTR_TRAIN_LOCAL_GPUS=1,2 htr-train launch packages/htr_train/lion_data_prep/configs/smoke_nemotron_parse_2gpu.yaml --where local
HTR_TRAIN_LOCAL_GPUS=1,2 htr-train launch packages/htr_train/lion_data_prep/configs/nemotron_parse_all.yaml --where local
```

```bash
htr-train render packages/htr_train/configs/smoke/minicpm_lines.yaml        # the scripts, nothing started
htr-train launch packages/htr_train/configs/smoke/minicpm_lines.yaml --where spark
htr-train launch packages/htr_train/configs/full/minicpm_lines_scaled.yaml --where slurm
HTR_TRAIN_LOCAL_GPUS=1,2 htr-train launch packages/htr_train/lion_data_prep/configs/minicpm_all.yaml --where local
htr-train smoke                                                             # every smoke run, queued in ~/diffu_queue
```

`runs/htr/<name>/` holds `job.yaml`, `manifest.json` (the commit, each env's lock digest, the models, every set's
digest, lines and options, the commands), `steps.sh` and `command.sh` (the guard's call) or `job.sbatch`. A
`launch` refuses a folder that holds a run already. To resume a training that stopped, run its `command.sh` (or
`job.sbatch`) again: the steps resume an ms-swift training from the newest checkpoint holding its
`trainer_state.json`, and the trainer skips the batches it trained (`--ignore_data_skip` is never passed: it would
train them again). A run whose checkpoints hold the model only cannot be resumed: run again, it starts from the beginning. Settings come
from `HTR_TRAIN_*` variables (`diffu_htr_train.settings`): `HTR_TRAIN_RUNS`, `HTR_TRAIN_ENVS`, `HTR_TRAIN_GUARD`,
`HTR_TRAIN_FLOOR_FILE`, `HTR_TRAIN_SLURM_PARTITION`, `HTR_TRAIN_SLURM_ACCOUNT`, `HTR_TRAIN_JEV_FONT`.

## Tests

- default lane (`pytest packages/htr_train`): the rendered commands against OpenBMB's and ms-swift's published ones,
  every refusal, the scripts, the manifest, the sets.
- `envs` lane (`pytest packages/htr_train -m envs`, the envs built): a real `swift sft` of two steps on tiny models, on
  CPU; Nemotron 3.5 Parse itself (it has no small form: its code fixes its vision encoder's size) trained two steps on
  CPU in two processes, resumed by its run's steps, and a batch of pages of three sizes read as each page alone
  (`pytest packages/page -m envs`: our pages read back by the model's own postprocessing, tokenizer and processor).
- `gpu` lane (`-m gpu`): the smoke runs, through the guard, one at a time.

## GT-Lejonet

[`lion_data_prep/`](lion_data_prep/README.md) holds the sets and runs for the full GT-Lejonet HTR export, matched one
to one to the TrOCR model trained on it (the same test pages, lines cut and kept as it cuts and keeps them): its
`build_sets.sh` cuts every line and region and starts `train.sh`, which trains MiniCPM-OCR on two GPUs from its
`configs/`, then hands over to `after_minicpm.sh`: the reader scored beside TrOCR, Jev's reads and pairs and the
draft's responses, then Jev and the DSpark draft, each model published to the Hugging Face Hub as it is made. Where
those runs stand: [docs/HTR_RUNS.md](../../docs/HTR_RUNS.md).
