# Handover: training MiniCPM-OCR, Jev and a DSpark draft on other hardware

For a session (or a person) picking up the reader training on another machine. [HTR_TRAIN.md](HTR_TRAIN.md) has
the design (recipes, data variants, why each tool); this page has the state as of 4 Nov 2026, what broke in the first
real runs and how to get around it, and the order to run things in. The full GT-Lejonet runs since then (what runs on
dmlpai01, what is queued, the results against the TrOCR baseline, what comes next) are in [HTR_RUNS.md](HTR_RUNS.md).

## Where it stands

Pilots on the Spark (GB10, one GPU), 2,000 steps of 8 samples each, on `data/htr/pilot` (at most 2,000 GT training
lines a collection, scaled to one 448 x 448 view; dev 5 % by volume; Jev pairs built from those lines):

| Model | Spec | Result | Usable? |
|---|---|---|---|
| MiniCPM-OCR, lines | `configs/pilot/minicpm_lines.yaml` | 1,687 dev lines: CER 59.1 % (base) -> **42.0 %** (pilot); exact lines 1.9 % -> 2.9 %; eval loss 2.89 -> 2.61, token accuracy 0.49 -> 0.52 | No: learning, far too short (under a fifth of one epoch) |
| MiniCPM-OCR, regions | `configs/smoke/minicpm_regions.yaml` | smoke only (eval loss 5.77 -> 5.69) | No |
| Jev (our English prompt) | `configs/pilot/jev_lines.yaml` | 4,000 seeded test pairs: AUROC 0.649 (base) -> **0.884** (pilot); accuracy at 0.5: 0.51 -> 0.81; mean score of a correct pair 0.02 -> 0.71, of a wrong one 0.003 -> 0.24 | Useful as a judge already; a full run should do better |
| DSpark draft | a pilot spec against the MiniCPM pilot (below) | training on 30,000 of the pilot's own answers, 3 epochs; train-time accept rate 0.02 -> ~0.4 in the first 3,000 steps; served acceptance not measured yet | Proves the pipeline; the real draft is trained against the final MiniCPM |

OpenBMB's released DSpark draft against base MiniCPM-OCR on our lines: 4.2 % of draft tokens accepted, 325 vs 341
tokens/s without it (no gain). A draft must be trained on the target's own answers.

On 4 Nov evening the envs were built on dmlpai01 (x86_64, RTX PRO 6000 Blackwell, 96 GB a GPU) and every route ran
there in small: MiniCPM-OCR and Jev on two GPUs (ZeRO-2, ZeRO-3, `--where local`), and DSpark end to end (vLLM with
hidden-state extraction, speculators' preparation and training, the trained draft served beside its target). The full
GT-Lejonet runs, matched to the TrOCR baseline, are `packages/htr_train/lion_data_prep/` (its README).

**Packing is off.** In `envs/minicpm`, Qwen3.5's linear attention carries state from one packed sample into the next:
a sample packed second differs by 0.82 relative from the same sample run alone (the first by 0.03), even with
`cu_seq_lens` passed. Packing would train on mixed lines, so MiniCPM-OCR and Jev train unpacked (sdpa).

## Run order on the new machine

1. **Code and environments.** Clone the repo at main; `uv sync` at the root, then each env the jobs use, one at a
   time: `uv sync --project packages/htr_train/envs/minicpm` (MiniCPM-OCR and Jev; add `--group multi-gpu` for
   DeepSpeed, `--group x86` for flash-attn and packing), `.../envs/vllm`, `.../envs/speculators`. Each env's
   [README table](../packages/htr_train/README.md) lists its pins.
2. **Models.** Download at their pinned revisions (the specs name them): `openbmb/MiniCPM-OCR-1.0`,
   `openbmb/MiniCPM-OCR-Jev`. On a node without internet, stage them in the Hugging Face cache first; Slurm runs set
   `HF_HUB_OFFLINE=1`.
3. **Data.** Rebuild the sets there rather than copying them (see "Data" below).
4. **Smoke runs** (minutes each), to prove the environments on this hardware: `configs/smoke/minicpm_lines.yaml`,
   `configs/smoke/jev.yaml`, then the DSpark smoke (`configs/smoke_vllm/dspark.yaml`) with the fixes below.
5. **MiniCPM-OCR lines, full**: `configs/full/minicpm_lines_scaled.yaml` (all GT training lines, OpenBMB's recipe,
   3 epochs, global batch 128). Score it (below) against base (59 % CER on the pilot dev lines).
6. **Jev, full**: `configs/full/jev_lines.yaml`, on pairs in the English prompt. Score AUROC on test pairs.
7. **DSpark draft** against the final MiniCPM-OCR: answers first (`htr-train serve-target`, `htr-train dspark-set`),
   then `configs/full/dspark.yaml` with the target's path filled in. Score acceptance served.

Every job is launched with `htr-train launch SPEC --where spark|slurm|local`, which writes `runs/htr/<name>/` (the
job, its `steps.sh`, the `command.sh` that runs it, the manifest of its sets). `local` runs it on this server's GPUs
that `HTR_TRAIN_LOCAL_GPUS` names (`1,2`), as many as the spec's `gpus_per_node`, ms-swift a process a GPU.

## Faults found in the first real runs (4 Nov), and the workaround until each is fixed in the code

The DSpark pipeline had only run in tests before 4 Nov; these all came up then. 2, 3 and 5 are fixed in `htr-train`
since (the draft's server gets `--enable-scale-out`, answers are written as text parts, the DSpark specs serve with
video off); for the others, edit the run's `steps.sh` (or the set) as described.

1. **Environment paths.** `steps.sh` points at `packages/htr_train/envs/<env>/.venv` under the checkout `htr-train`
   was run from; if the envs were built in another checkout (a worktree), the run stops at once ("No such file or
   directory"). Build the envs in the checkout you launch from.
2. **`--enable-scale-out`.** speculators' `prepare-data` renders each conversation through vLLM's render endpoints,
   which return 403 ("Scale-out endpoints are disabled") unless vLLM is started with `--enable-scale-out`.
   speculators' own launcher (`scripts/launch_vllm.py`) always adds it on vLLM releases that have it; our launch
   dropped it with the speed-only render flags. Add it to the `vllm serve` line, **after the model path** (`vllm
   serve` takes the model as its first argument).
3. **Answers that parse as numbers.** speculators loads the responses with `datasets`' JSON loader, whose message
   `content` column is a `Json` feature: an answer that is a bare number ("897", "18.") comes back as an int or float,
   and that conversation fails (`TypeError: 'int' object is not iterable`), about 1 % of OCR lines. Write each
   answer as a list of text parts, as the prompts are: `[{"type": "text", "text": "..."}]`. Checked on 300
   conversations: the same token ids and loss masks as the plain-string form, no failures.
4. **vLLM's memory share during the draft's training.** At `--gpu-memory-utilization 0.35` (the spec's
   `vllm_memory`), the box went from 68 to 104 GB when training started and the Spark's guard stopped it. 0.2 is
   plenty for 1,024-token lines (vLLM's share is mostly KV cache these short sequences barely use); the run since
   holds at ~80 GB, 97 GB peak. On a GPU with its own memory this matters less; keep vLLM's share small anyway.
5. **Video off for MiniCPM-OCR in vLLM.** vLLM probes MiniCPM-OCR's video path at start-up, and with transformers
   5.22.1 it fails (`unexpected keyword argument 'disable_small_image_upsize'`). Serve it with
   `--limit-mm-per-prompt '{"image": 4, "video": 0}'` (the spec's `serve_args`).
6. **Jev served by vLLM** (for scoring): ms-swift saves the fine-tuned Jev as `Qwen3_5ForSequenceClassification`,
   which vLLM builds text-only ("no module or parameter named 'visual'"). Its weights are laid out as the released
   Jev's (`model.visual`, `model.language_model`, `score`), so load it as the released one is:
   `hf_overrides={"num_labels": 2, "architectures": ["Qwen3_5ForConditionalGeneration"]}`. A script that runs
   vLLM's offline `LLM` needs the `if __name__ == "__main__":` guard (vLLM spawns its workers).

`dspark-set` sends its requests with little concurrency (~5 answers/s on the Spark against ~19/s from 8 parallel
requests); for ~300,000 answers, raise it first.

## Data

The sets are built by `diffu-reader-set` from the GT's PAGE/ALTO exports and scans, split by the GT's own test lists,
then `htr-train split-dev` and `htr-train jev-set` (HTR_TRAIN.md, "Data, and variants of it"). The JSONL rows hold
**absolute image paths** of the machine that built them, so a set copied elsewhere does not load; rebuild the sets on
the new machine from the GT (or rewrite the paths' prefix consistently in every set file, `*.options.json` included,
and check a few rows). Only the pilot and smoke variants exist so far; the full variants the `configs/full` specs name
are not built yet.

Before training on real lines, check the set was split by the GT's test lists: runs, dev splits and Jev pair sets
refuse one that was not.

## Scoring (not in `htr-train` yet)

The pilot numbers above were measured with throwaway scripts; until scoring is a command of `htr-train`, measure the
same way so numbers compare:

- **MiniCPM-OCR CER**: serve the checkpoint with vLLM (`--max-model-len 8192 --limit-mm-per-prompt '{"image": 4,
  "video": 0}' --mm-processor-kwargs '{"downsample_mode": "4x", "max_slice_nums": 9}' --trust-request-mm-kwargs
  --default-chat-template-kwargs '{"enable_thinking": false}'`); ask every dev line the training prompt ("Text
  Recognition:" after the image), temperature 0, at most 512 new tokens; CER = total edit distance / total GT
  characters, both stripped at the ends, case, accents and punctuation kept. Base on the 1,687 pilot dev lines: 59.1 %.
- **Jev AUROC**: vLLM's offline classifier (`runner="pooling", convert="classify"`, two images a prompt, the
  overrides of fault 6), the raw prompt from `diffu_htr_train.jev.prompt.model_prompt` (it ends with the empty
  assistant turn whose last token is pooled), `use_activation=False` and the two logits softmaxed: `prob_1`. AUROC
  over a seeded sample of the test pairs (4,000, `random.Random(0)`), ties averaged. Base: 0.649.
- **DSpark draft**: serve the target with the draft (`--speculative-config '{"method": "dspark", "model": "<draft>",
  "num_speculative_tokens": 15, "enable_adaptive_verification": false}'`), read held-out lines, and report accepted
  draft tokens / draft tokens (vLLM's `/metrics` `spec_decode_*` counters) and tokens/s against the target served
  alone, with the same lines and the same texts (decoding is greedy; the texts should match).

## Open questions for the full runs

- The DSpark recipe. Ours is speculators' example with OpenBMB's draft shape (block 15, lr 3e-4, 3 epochs). DeepSeek's
  DeepSpec (the DSpark authors' code; text targets only) publishes block 7, lr 6e-4, global batch 512, 10 epochs, a
  position decay of the loss (gamma 4) and 512 anchors. Which carries over to a line-OCR target is to be settled
  against the pilot draft's served acceptance.
- Packing for the full MiniCPM-OCR run (5-10x faster) needs flash-attn on the new machine; the specs use `sdpa` and no
  packing, as on the Spark.
- Regions: no pilot yet.
