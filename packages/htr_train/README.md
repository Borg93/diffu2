# diffu-htr-train

Training our page and line readers from run specs: Unlimited-OCR 2 (the page reader; LoRA or full), MiniCPM-OCR 1.0
(the line and region recogniser; OpenBMB's full recipe), MiniCPM-OCR-Jev (the proof-reader that scores whether a crop
shows a text) and a DSpark draft for our fine-tuned MiniCPM-OCR. The tools (ms-swift, speculators, vLLM) each run in
their own environment under `envs/`; this package is a light workspace member that checks a spec, renders the tools'
exact commands, launches them on the Spark (under `jobs/spark_guard.sh`) or on Slurm, and builds the sets they train
on. The design, with every recipe and pin cited, and the hardware estimates: [docs/HTR_TRAIN.md](../../docs/HTR_TRAIN.md).

## The environments

Each is its own uv project and lock, excluded from the workspace (as `packages/sglang_moss`): their pins conflict with
each other and with the workspace's (transformers 5.22 vs 4.52.1 vs speculators' <= 5.19 vs vLLM's).

| Env | For | Pins |
|---|---|---|
| `envs/minicpm` | MiniCPM-OCR, Jev | Python 3.12, torch 2.14.1 (cu130), ms-swift @ `ef4953a4`, transformers 5.22.1, flash-linear-attention 0.4.2, peft < 0.21, datasets < 4.8.5, qwen-vl-utils 0.0.14, trackio; groups `multi-gpu` (deepspeed >= 0.19.3, < 0.20) and `x86` (flash-attn 2.8.3, causal-conv1d: packing) |
| `envs/uocr2` | Unlimited-OCR 2 | Python 3.12, torch 2.14.1, ms-swift @ `ef4953a4`, transformers 4.52.1, easydict, addict, peft < 0.21, datasets < 4.8.5, trackio; group `multi-gpu`. Its remote code runs from `baidu/Unlimited-OCR-2@0480966938291445591f69f08a13b041c073a96a` only |
| `envs/speculators` | the DSpark draft | Python 3.12, speculators @ `560f0465` (0.9.0 + the prepare-data render-route fix), torch 2.14.1, transformers overridden to 5.22.1 (speculators pins <= 5.19.0; `minicpm_ocr` is native from 5.21) |
| `envs/vllm` | serving the target for the draft | vLLM at OpenBMB's pin `fa3b9d13` from vLLM's own prebuilt wheels (linux aarch64 and x86_64), transformers overridden to 5.22.1; built and run on the Spark (GB10) |

```bash
uv sync --project packages/htr_train/envs/minicpm             # one at a time; --group multi-gpu on a cluster
uv sync --project packages/htr_train/envs/uocr2
uv sync --project packages/htr_train/envs/speculators
uv sync --project packages/htr_train/envs/vllm
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
# Jev's pairs from the training lines (scaled to the generator's line height), our frozen renderer
htr-train jev-set --lines data/htr/minicpm_lines_scaled.train.jsonl --out data/htr/jev_lines.jsonl
# Jev's test pairs (to evaluate it on), from the test lines
htr-train jev-set --side test --lines data/htr/minicpm_lines_scaled.test.jsonl --out data/htr/jev_lines.test.jsonl
# a draft's responses: the target served by vLLM (htr-train serve-target SPEC prints the command) answers every prompt
htr-train dspark-set --prompts data/htr/minicpm_lines_scaled.train.jsonl --target <snapshot> --out data/htr/dspark_responses.jsonl
```

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
`minicpm-ocr`, `unlimited-ocr-2`, `jev`, `dspark`), its sets and its resources. It is checked before anything starts:
an unpinned Hub model, remote code for anything but Unlimited-OCR 2 at its revision, a test page or a dev volume in
training, a missing image, pairs of another renderer, a draft's responses from another target, a batch that does not
divide, packing without flash attention, and a Spark run of more than its one GPU or memory are refused.

```bash
htr-train render packages/htr_train/configs/smoke/minicpm_lines.yaml        # the scripts, nothing started
htr-train launch packages/htr_train/configs/smoke/minicpm_lines.yaml --where spark
htr-train launch packages/htr_train/configs/full/minicpm_lines_scaled.yaml --where slurm
htr-train smoke                                                             # every smoke run, queued in ~/diffu_queue
```

`runs/htr/<name>/` holds `job.yaml`, `manifest.json` (the commit, each env's lock digest, the models, every set's
digest, lines and options, the commands), `steps.sh` and `command.sh` (the guard's call) or `job.sbatch`. Settings come
from `HTR_TRAIN_*` variables (`diffu_htr_train.settings`): `HTR_TRAIN_RUNS`, `HTR_TRAIN_ENVS`, `HTR_TRAIN_GUARD`,
`HTR_TRAIN_FLOOR_FILE`, `HTR_TRAIN_SLURM_PARTITION`, `HTR_TRAIN_SLURM_ACCOUNT`, `HTR_TRAIN_JEV_FONT`.

## Tests

- default lane (`pytest packages/htr_train`): the rendered commands against OpenBMB's and ms-swift's published ones,
  every refusal, the scripts, the manifest, the sets.
- `envs` lane (`pytest packages/htr_train -m envs`, the envs built): a real `swift sft` of two steps on tiny models, on
  CPU.
- `gpu` lane (`-m gpu`): the smoke runs, through the guard, one at a time.
