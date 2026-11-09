# Training the readers (htr-train)

Think of four people working on one archive. **Unlimited-OCR 2** is the scribe who copies a whole page in one go and
notes where each paragraph sits and what it is: our page reader. **MiniCPM-OCR** is a fast copyist handed one strip
of the page at a time (a line, or a paragraph): our line and region recogniser. **MiniCPM-OCR-Jev** is the
proof-reader: shown a strip, the text someone claims is written there and a typed print of that text, it answers only
"same or not", as a probability; it never writes text itself, which is why it suits checking our drawn lines.
**DSpark** is an apprentice who guesses the copyist's next 8-15 tokens at once so the copyist only checks them: copying
gets faster, the result does not change, but the apprentice must have learnt from this very copyist, so our
fine-tuned MiniCPM-OCR needs its own.

`packages/htr_train` (`htr-train`) trains all four from run specs; this page is the how and why. The package's own
[README](../packages/htr_train/README.md) has the commands. `htr-eval` scores MiniCPM-OCR (line sets, readings through
vLLM, CER as read and with loops cut), and a run reads fixed draws while it trains (`cer_sets`). Jev's AUROC, a draft's
served acceptance and the judge stage are not commands yet.

State as of 4 Nov 2026, what broke in the first real runs, and the run order on other hardware:
[HTR_HANDOVER.md](HTR_HANDOVER.md). The full GT-Lejonet runs on dmlpai01 (what runs, what is queued, the results
against the TrOCR baseline): [HTR_RUNS.md](HTR_RUNS.md).

## One tool where it fits

- **ms-swift** (`swift sft`) trains the three models: MiniCPM-OCR full, Unlimited-OCR 2 LoRA or full, Jev as a
  two-class sequence classifier. It is pinned to main at `ef4953a4` (4.7.0.dev0), the first with `minicpm_ocr` and
  Unlimited-OCR 2; release 4.6.0 has neither.
- **speculators** at main `560f0465` (0.9.0 plus #1317: `prepare-data` stops instead of silently dropping conversations when the render route is missing) trains the DSpark draft: ms-swift trains no draft of any kind for MiniCPM-OCR (MTP heads only,
  under Megatron, which has no `minicpm_ocr`).
- **vLLM** serves the target for the draft (its rendered conversations and hidden states). Unlimited-OCR 2 has no
  vLLM support yet, so it gets no draft.

## The recipes, as published

| Job | Recipe | Our deltas |
|---|---|---|
| MiniCPM-OCR on lines and regions | OpenBMB's French handwritten lines (`docs/swift_minicpm_ocr_french_en.md`): `<image>Text Recognition:`; full SFT of the vision encoder and language model, projector frozen; `--add_non_thinking_prefix true --loss_scale ignore_empty_think`; lr 5e-6 cosine, warmup 0.05, AdamW fused (beta2 0.95, eps 1e-8), wd 0.1, clip 1.0; global batch 128 (8 x 1 x 16); ZeRO-2; `DOWNSAMPLE_MODE=4x`; 3 epochs, the best by eval loss | ms-swift's built-in `zero2` for their `ds_zero2.json` (the same config); our sets and dev split; on dmlpai01 (`lion_data_prep/configs/minicpm_all.yaml`) lines and regions together, 2 GPUs x 4 x 16, gradient checkpointing in the vision encoder too |
| Jev | OpenBMB's `docs/finetune_jev_en.md`: `--task_type seq_cls --num_labels 2 --problem_type single_label_classification`, full, vision and projector frozen, lr 1e-5, packing 12,000 with flash attention, ZeRO-3, `IMAGE_MAX_TOKEN_NUM=2048` at training only | started from `openbmb/MiniCPM-OCR-Jev` itself (its head and comparison carry over), `--model_type qwen3_5` named; on dmlpai01 (`lion_data_prep/configs/jev_all.yaml`) unpacked with sdpa (packing leaks between samples in Qwen3.5, "Not known yet"), 2 epochs |
| Unlimited-OCR 2 | ms-swift's `examples/models/unlimited_ocr2/train.sh`: LoRA r8 alpha 32 all-linear, vision and projector frozen, lr 1e-4, accumulation 16 | the Hugging Face repo at its commit, `--model_type unlimited-ocr` named, rank 16; a full fine-tune variant |
| DSpark | speculators' `examples/train/dspark_qwen3_0_6b_sharegpt_online.sh`: online, vLLM serving the target's hidden states | OpenBMB's released draft's shape: 5 layers, 15 tokens a block, target layers 4 8 12 16 20 (their `[3, 7, 11, 15, 19]` read as Z Lab's `id + 1`, an inference), mask token 248076; the draft vocabulary is speculators' default 32,000, not the 114,688 tokens OpenBMB's draft keeps (to be tuned once a draft is measured) |

Every run also passes `--use_hf true`, `--split_dataset_ratio 0` (never a split by sample: lines of one page would land
on both sides) with the dev set given whole, `--add_version false`, and logs to the local trackio project `diffu` under
the run's name (Unlimited-OCR 2's transformers 4.52.1 predates TrainingArguments' trackio project, so its runs log to
tensorboard). `--freeze_vit` and `--freeze_aligner` default to true in ms-swift even for a full fine-tune, so a spec
must state both. The golden tests (`packages/htr_train/tests/test_render.py`) check each rendered command against the
published one, flag for flag, less exactly these deltas.

## Data, and variants of it

The sets come from `diffu-reader-set` (diffu-page owns the PAGE/ALTO files and the scans): `--task lines` for
MiniCPM-OCR (a sample per text line, its box cut as `<page id>_<k>_<j>.jpg`, the meta giving its kind, production,
element and line), `--task elements` for regions, `--task parse` for Unlimited-OCR 2. The GT's test lists send every
test scan to `<stem>.test.jsonl` (`--split-from-gt-test-lists`), never to training, and the GT's second export of
`GT-RA_landskapshandlingar_20260227` is always left out.

A set is a **variant**, built once, and several runs are meant to be compared:

- all lines, or at most N a collection (`--max-per-collection N --seed S`: the scans shuffled first, so the cap takes
  from all of a collection's volumes);
- some collections (`--collection`, `--exclude-collection`);
- lines, regions, or both mixed (two sets, mixed by ms-swift's sampling: each part's `samples` in the run spec);
- scaled crops (`--max-crop-pixels 200704`: a line in one 448 x 448 view, 256 visual tokens instead of 1,000-1,500) or
  the scans' resolution (no option);
- real pages only, or real and drawn (a second set from a `diffu-run-export`, mixed in the spec).

Each set records its options and counts in `<stem>.options.json`; `htr-train split-dev` (a dev split of whole volumes,
or another split's volumes with `--like`) and `htr-train jev-set` write theirs, carrying their parent's; a run's
`manifest.json` carries every set's options with its digest. A run spec's `description` says what the variant is.

No real sample is trained on unless its set was split by the GT's test lists and it is on the training side (a real
page of an unsplit set may be a test page): a run, a dev split and a Jev pair set refuse it.

**Jev's pairs** (`htr-train jev-set`), from the training lines (`--side test`: from the test lines, to evaluate
Jev on; a line of the other side is refused): each crop with its GT text (label 1) and, by default, one wrong
candidate (label 0, `--negatives`), drawn by the line's seed from the rules (confusable letters of period hands: n/u,
e/c, long s/f, a/ä/å, o/ö; a letter inserted or dropped; a word dropped or two swapped; a digit changed; the first or
last word cut) and another line of the same page (closest in length); and readers' reads (`--reads`), positive where
they are the GT after NFC and whitespace normalisation, negative where not, each with its CER to the GT. Every
candidate is drawn by our renderer (one line, black on white, at the crop's size, FreeType's basic layout; DejaVu Sans
by default, `HTR_TRAIN_JEV_FONT`), whose version (`htr-train-jev-render-1+<font digest>+pillow-<v>+freetype-<v>`)
each pair records; a Jev run refuses pairs of another. The prompt is ours, in English (`diffu_htr_train.jev.prompt`:
the crop, "OCR text:" and the candidate, its rendering, then the instruction to judge whether they match); each pair
records its fingerprint, and a Jev run refuses pairs built in another wording. Jev outputs a probability; the accept threshold is a setting of the judge (to come,
default 0.5), not calibrated here.

Our renderer is not OpenBMB's. Theirs (in their Docker image and render service) draws a candidate with a pinned
Chromium, CJK fonts and KaTeX, laid out by its category (`text`, `formula`, `table`), and they warn that a different
renderer changes the score ("a new renderer version is a new input distribution"). So base Jev's scores on our renders
are not comparable to their published numbers, and a Jev fine-tuned on our pairs reads our renders only: it is
evaluated, and later run as the judge, with the same renderer version. Their unit is a region (a text, formula or table
block, labelled by people); ours is a line of text, labelled by construction. A region-pair set (from
`diffu-reader-set --task elements`) is the obvious second variant. Jev's processor raises an image to at least 65,536 pixels (its
`preprocessor_config.json`, `shortest_edge`), so a 48 px line crop (48 x 600 = 28,800 px) and its render are both
enlarged about 1.5 times before the model sees them: the same for both images, but a native-height variant
(`--height 0`) is worth one run.

## A draft's data, and serving it

`htr-train serve-target SPEC` prints the vLLM command that serves a DSpark spec's target for its responses (by its
path, under that name, reading images below the spec's `media_root` only; no remote code); `htr-train dspark-set`
asks it every prompt, greedy, thinking off, and records the target in `<stem>.target.json`, which the draft's run
checks (a set answered by another target, or with no answers, is refused). vLLM runs in `envs/vllm`, built for x86
first, so the draft's smoke (`configs/smoke_vllm/dspark.yaml`) is an x86 run, not part of the Spark's smoke queue. The
trained draft is served beside its target as OpenBMB serves theirs (`docs/minicpm_ocr_tutorial_en.md`):

```bash
vllm serve <target snapshot> --mm-processor-kwargs '{"downsample_mode": "4x", "max_slice_nums": 9}' --trust-request-mm-kwargs \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --speculative-config '{"method": "dspark", "model": "<draft checkpoint>", "num_speculative_tokens": 15, "enable_adaptive_verification": false}'
```

and checked to be active through vLLM's `/metrics` `spec_decode_*` counters.

From speculators' docs and DSpark research notes (read in full at `560f0465`):

- The training command is speculators' DSpark example's (`examples/train/dspark_qwen3_0_6b_sharegpt_online.sh`) with
  OpenBMB's draft shape (5 layers, block 15, mask token 248076, their layers `[3, 7, 11, 15, 19]` in Z Lab's
  convention, which speculators' DFlash converter maps to `[4, 8, 12, 16, 20]`), plus the research runs' cosine
  schedule with 4 % warmup and 2 dataloader workers a rank (`lion_data_prep/configs/dspark_all.yaml` sets 8, as
  measured). vLLM gets the same layers and the last (24), as `scripts/launch_vllm.py` adds it.
- **vLLM on the run's first GPU, the trainer on its last** (`gpus_per_node: 2`; with one GPU both share it). Their
  DSpark example: "online training needs separate GPUs for vLLM and training". Sharing one GPU, the trainer waits on
  vLLM: an epoch's 390 steps on 2,400 of the 9 Nov draft's samples took 153 s shared and 100 s split
  (`docs/HTR_RUNS.md`).
- **vLLM's front end as their launcher starts it**, from the run's CPU budget (`cpus_per_task`; `render_budget` in
  `render.py`): `--api-server-count`, `--renderer-num-workers 2`, native thread pools bounded, and prepare-data's
  workers from the same budget. With vLLM's single API server, every conversation is rendered in turn: the 9 Nov draft
  prepared its 300,000 answers in about 1 h 50 min. An earlier reading here left these flags out as speed only.
- **A checkpoint every tenth of an epoch** (`checkpoint_freq: 0.1`, as their Kimi-K3 DSpark example saves), not
  `--save-best`: with it speculators saves only an epoch that improves validation loss and never mid-epoch
  (`trainer.py`), so a stopped run loses the epoch in progress. `checkpoint_best` is linked either way.
- A DSpark draft is served with `num_speculative_tokens` equal to its trained block size (vLLM asserts it), so 15.
- The draft trains with `speculators train` itself, not under `torchrun` (under `torchrun` it claims an accelerator
  first). Its trackio logs land in the local project `speculators` (speculators names it; no flag changes it), not
  `diffu`.
- Validation loss on the responses' own 10 % split ranks drafts poorly (speculators' research notes); a draft is judged
  by acceptance length on held-out OCR prompts, served as above, in the evaluation step.
- Cheaper routes, not built: fine-tuning OpenBMB's released draft (`--from-pretrained`) needs a converter of ours
  first (speculators converts DFlash checkpoints, not DSpark ones, and OpenBMB's `d2t` is delta-encoded with a
  114,688-token vocabulary), and a fine-tune wants a learning rate about 10x below the from-scratch 3e-4. Their MTP
  route (fine-tuning a target's own multi-token head) does not apply: MiniCPM-OCR's config has
  `mtp_num_hidden_layers: 0`. The first measurement is OpenBMB's draft as it is against our fine-tuned target: if its
  acceptance holds, no draft is trained at all.

## Where it runs

The Spark (GB10) is for smoke runs and small pilots, one guarded GPU job at a time; every run on all of the GT belongs
on HPC or the x86 Blackwell servers. Estimates (the design's arithmetic; only OpenBMB's RIMES run is measured: 1.86
line samples/s per A100):

| Job (all GT training data, one epoch) | GB10 | 8 x A100 | Where |
|---|---|---|---|
| MiniCPM-OCR lines, scaled, no packing (1.51 M x ~290 tokens) | ~750 h | ~28 h | HPC |
| same, packed (flash-attn) | ~70-135 h | ~2.5-5 h | HPC |
| MiniCPM-OCR regions, packed (85 K x ~2,800) | ~37-73 h | ~1.5-3 h | HPC |
| Jev, 1 M pairs, 2 epochs, packed | ~150-300 h | ~6-11 h | HPC |
| Unlimited-OCR 2 LoRA (53 K pages x ~1,250) | ~15-60 h | ~0.6-1.2 h | Spark over a weekend, or HPC |
| Unlimited-OCR 2 full | no | ~1-2 h | HPC |
| DSpark draft (~300 K responses, 3 online epochs) | ~1-2 days if vLLM runs there | ~2-4 h | HPC / x86 |

Measured on dmlpai01 (2 x RTX PRO 6000 Blackwell Max-Q, 96 GB, 300 W): MiniCPM-OCR on all lines and regions,
unpacked, ZeRO-2, gradient checkpointing in both, takes about 11.9 s a step of 128 samples (~10.8 samples/s). That is
~40 h for an epoch of 12,227 steps and about 5 days for the 3 epochs ([HTR_RUNS.md](HTR_RUNS.md)).

Memory per GPU (estimates): MiniCPM-OCR full ~25-30 GB alone, ~10-15 GB on 8 with ZeRO-2; Jev ~40-45 GB alone at 12 K
packed tokens; Unlimited-OCR 2 LoRA 26 GiB (ms-swift's example), full ~65 GB alone, ~20-25 GB on 8.

On Slurm, `htr-train launch --where slurm` writes an sbatch of one srun task a node; each task sets ms-swift's
`NNODES`, `NODE_RANK`, `MASTER_ADDR` (the first host), `MASTER_PORT` and `NPROC_PER_NODE`, with `HF_HUB_OFFLINE=1`
(models staged before). KubeRay is not built (no Ray work until testing is done).

## Evaluation, as OpenBMB's docs ask

MiniCPM-OCR's part is built (`htr-eval`). Jev's and the draft's are scored by scripts until they are commands
([HTR_HANDOVER.md](HTR_HANDOVER.md), "Scoring").

- MiniCPM-OCR is read as OpenBMB evaluates it: thinking off, greedy (`temperature` 0.01, `top_p` 0.001, `top_k` 1,
  `repetition_penalty` 1.0), `downsample_mode` 4x, `max_slice_nums` 9, at most 4,096 new tokens for a text region
  (512 a line, as on RIMES), `finish_reason` `length` counted as a failure; CER and WER on NFC text, stripped, with
  case, accents, punctuation and inner spaces kept (their RIMES numbers: base 5.67 % CER, fine-tuned 2.98-3.08 %).
- Jev is served by vLLM as a classifier: `--runner pooling --convert classify --hf-overrides '{"num_labels": 2}'`
  (its `config.json` has no `num_labels`), `--chat-template-content-format openai` (with the default, the images and
  text go in another order than in training, and the scores still look plausible), `--limit-mm-per-prompt
  '{"image": 2}'`; `prob_1` is `softmax(logits)[1]`; the prompt is ours (`diffu_htr_train.jev.prompt`), its last token id 271. No
  `IMAGE_MAX_TOKEN_NUM` at evaluation: the checkpoint's own image budget, as OpenBMB's guide says.
- Jev's metric is AUROC on test pairs (`htr-train jev-set --side test`); a threshold, if one is ever needed, is chosen
  on labelled samples of our own pages by the accept rate and the accuracy of what is accepted.

## Not known yet

- Whether packing can be made safe for Qwen3.5 here: in `envs/minicpm` a packed sample's linear-attention state leaks
  into the next (HTR_HANDOVER.md), so every run is unpacked until a check of the same sample packed and alone agrees.
- Whether flash-attn and causal-conv1d build for aarch64 / sm_121 (plan: `sdpa`, no packing, on the Spark).
- Whether Jev's `score.weight` loads into ms-swift's classifier head (check the load log for a newly initialised
  `score`).
- Scaled or native line crops: the same 2,000 lines read by base MiniCPM-OCR both ways settles it.

Settled since: vLLM's aarch64 wheels and speculators with transformers 5.22.1 both ran the DSpark pilot on the Spark
(GB10), and the whole DSpark route ran on dmlpai01 (HTR_HANDOVER.md).
