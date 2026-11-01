# MOSS-Image: our implementation and its training

MOSS-Image (OpenMOSS; code and weights Apache-2.0) is the main model path: a 5B single-stream DiT with a frozen
Qwen3.5-9B text encoder and the FLUX.2 VAE, whose Layout prompt writes every line's exact text in its box. The
package `packages/moss` (diffu-moss) is our own implementation, written from the published design, around the released
weights; nothing of the official pipeline is copied or run. This page is how it draws, what it is trained on and how it
is trained. How a page becomes the crops it draws: [PIPELINE.md](PIPELINE.md).

## The model

- **The DiT** (`transformer.py`): `MossTransformer`, 42 single-stream blocks over the image's and the prompt's tokens;
  the released Base transformer (or a merged LoRA or a full fine-tune, exported in diffusers format) loads into it by
  name, nothing missing or unexpected.
- **The blocks** (`modular_pipelines/moss/`): `MossAutoBlocks` and `MossModularPipeline`. `MossLayoutPromptStep`
  compiles a crop plan's caption and lines into MOSS's Layout prompt (`layout_prompt.py`, as the MOSS prompt-enhancer
  skill compiles a draft): the caption's prose as the summary, a text item per line with its exact text and its box in
  pixels from the canvas centre (`yxyx`), the paper, fragments and ruled lines as objects. The prompt is encoded by the
  frozen text encoder (its last hidden states over the chat template, thinking enabled) and refused past 4,096 tokens,
  never cut (a 44-47-line page takes 1,500-2,500). The denoise step is laid out as FLUX.3's (`prepare_latents`,
  `set_timesteps`, a loop of `denoiser` and `after_denoiser`, the latents as tokens in FLUX.3's order), so the page
  pipeline's line repair runs in it unchanged; the noise is drawn in fp32, the sampling grid is the shifted steps moved
  to the nearest level of the 1,000-step training grid (shift 5, MOSS's demo's), and guidance (3) combines in fp32 with
  the unconditional pass skipping block 22 (`uncond_skip_layers` in the model index). The FLUX.2 VAE decodes
  (`latents.py`, which also encodes a drawn crop for a repair). On a tiny random model the blocks draw bit for bit what
  the verified scratchpad code drew, and that code redrew the official demo's published images at 20.9 and 26.2 dB.
- **A model directory** (`diffu-moss-init`): a `modular_model_index.json` naming `MossAutoBlocks`, the text encoder,
  tokenizer and VAE loading from the Base snapshot's folders, the DiT from the snapshot or a trained export. Nothing is
  copied. `takes_reference` is set only for a DiT of our reference training (read from its `export.json`).

      BASE=~/.cache/huggingface/hub/models--OpenMOSS-Team--MOSS-Image-Base/snapshots/85975793d8b3d432f1b6e27f352abaa4c2a4c0d3
      diffu-moss-init --base $BASE --out runs/moss/base
      diffu-moss-init --base $BASE --transformer runs/moss_hand_1715/merged_step350 --out runs/moss/hand_1715

- **Reference images** (sample pages): the DiT reads a reference's clean latents as tokens after the prompt's, on the
  reference axis (reference `k` at `k + 1`), its grid centred, modulated as at timestep 0 (FLUX.3 Klein's design), in
  every guidance pass; without one it is the released model bit for bit. The released models were not trained with
  references, so a model directory takes one only when its DiT was. `MossReferenceEncodeStep` encodes a reference on
  its own grid after the page pipeline's `ReferenceGridStep` (0.5 MP), and the Layout prompt names "the sample image"
  where the caption cites `<ref_image_0>` (without a reference the citation is dropped). Reference guidance
  (`--reference-guidance`) adds a pass without it.
- **Drawing**: `diffu-generate --model <MOSS dir>` (and `diffu-evaluate`, `diffu-studio`) draws through the page
  pipeline, at MOSS's own guidance and shift unless `--guidance` or `--sampling-shift` name others. A MOSS directory
  takes no `--lora`: merge a LoRA into the DiT folder it names (`diffu-moss-export-ref`, or KohakuMossTrainer's merge).
  `--reference` (a sample page) only for a DiT trained with references, its caption in the form it trained on
  (`--reference-caption`, below).

      jobs/spark_guard.sh --need-gb 50 --name pages -- uv run diffu-generate --model runs/moss/hand_1715 \
          --templates page_a.xml --text-from text.txt --title "the records of Överståthållarämbetet" \
          --paper yellowed --ink brown --whole-page --page-megapixels 2 --out runs/pages_moss

## Training data

Built on CPU by diffu-page's builders, cached once on the GPU by `diffu-moss-cache-pairs`.

- **Style pairs** (`python -m diffu_page.data.style_pairs`, also FLUX.3's style-reference LoRA's: [FLUX3.md](FLUX3.md)):
  a training crop (`diffu_page.data.build_crops`) and a crop of its own or its facing page sharing none of its lines,
  the target's caption describing the reference's measured hand, ink and paper (`data_flux3/style_pairs_v3d`). The
  cache leaves out the pairs whose reference repeats a target's line (85 of v3d's).
- **Page pairs** (`python -m diffu_page.data.page_pairs`, the full fine-tune's data): every single page of the
  handwritten GT (an opening split into its two; a scan wider than tall that its lines cannot split skipped and
  counted, never a page), cut at about 1 MP exactly as `diffu-generate --whole-page` plans it and prompted as drawing
  prompts it, with a sample page of its own volume sharing none of its lines (0.5 MP). Whole volumes are held out
  (`val.jsonl`, `val_eval/` for `diffu-evaluate`); prompts past 4,096 tokens are skipped and counted; scans run in
  parallel and a stopped build resumes from the scans it recorded (`scanned/`); `build.json` records the settings and
  what became of every scan, page and window. `--windows-per-page N` adds up to N windows of each page (whole lines of
  one column, planned on the single page as diffu-generate plans its crops, cut at the pages' pixels: larger letters),
  each a row of its own with its page's sample, after the page rows, which stay exactly as without windows.
  `python -m diffu_page.data.page_pair_forms` gives the rows of a set built before a form every form, in a new manifest
  beside it (each row planned again from its PAGE file and scan as the build planned it; `--replace <form>` plans a
  form whose caption changed again), its images as they are.

      python -m diffu_page.data.page_pairs --gt-root ~/Desktop/data/GT-Lejonet_HTR_202602 --model runs/moss/base \
          --out data_flux3/page_pairs --workers 8 --windows-per-page 2

- **Prompt forms** (`train/pairs.py`: `Variant`, `SHARES`): each pair is cached in one of four forms, fixed per pair by
  a seed: a fifth without the reference (`no_reference`: the citation dropped, the look words kept), a fifth with the
  sample and the look in words (`look`; drawn with `--reference-caption named`), a fifth with the sample and no look
  words (`plain`; a page pair still names its hand: `--reference-caption hand`, a style pair's `plain`), and two
  fifths with the sample alone (`sample`: no hand named, no look words, the paper "the paper of the sample image":
  `--reference-caption sample`), so the hand and the look must be read from the image. Where each archive has one
  look (the eight hands' set), its name gives both away: trained on the first three forms, the model ignored its
  sample. A style pair's caption names no hand, so its plain and sample forms differ in the paper alone
  (`sample_caption.py` drops the look words and names the sample image).
- **The cache** (`diffu-moss-cache-pairs`): each entry encoded exactly as drawing encodes it, by the model directory's
  own components loaded as the page pipeline loads them; one safetensors file per entry, written whole, so a stopped
  run resumes; `entries.jsonl` and `cache.json` last. `--limit`, `--every-form` (a validation set),
  `--latents-only` (no text encoder: for `--encode-text`), and `--restyle-share s`: that share of the sample-form
  entries takes one look change for the page and its sample alike (`train/restyle.py`: microfilm, colourised or
  tinted, each image's own ink and paper mapped onto the drawn tones, no pixel moved), so a page's look cannot come
  from its hand; each such entry records its change, and `diffu-draw-pairs` draws it back from its sample in that look.

      diffu-moss-cache-pairs --model runs/moss/base --pairs data_flux3/style_pairs_v3d/train.jsonl \
          --out data_flux3/cache_moss_ref_v3d --limit 4000
      diffu-moss-cache-pairs --model runs/moss/base --pairs data_flux3/page_pairs_eight/train_bal_sample2.jsonl \
          --out data_flux3/cache_moss_eight_bal_restyle --restyle-share 0.5

## Reference training (`diffu-moss-train-ref`, our own loop)

KohakuMossTrainer has no reference or editing training, so the sample page needs our own trainer.

- **Noise and loss**: MOSS-Image's own noise levels (KohakuMossTrainer's default sampler: the training grid j/1000, a
  truncated logit-normal of median 0.78, 1 % uniform), flow matching's loss over the drawn image's tokens, a validation
  loss per prompt form at the same noise. `--sigma-shift k` moves each drawn level to `k sigma / (1 + (k - 1) sigma)`
  (`train/sigmas.py`; under 1 towards the clean end, where letter strokes are decided); the validation stays at MOSS's
  levels, so runs compare.
- **A LoRA** (the default): rank 32, alpha 32, the 1715 hand LoRA's modules and optimizer (AdamW 1e-4, constant after
  50 warmup steps), the joint blocks checkpointed; `diffu-moss-export-ref` merges a checkpoint into Base for
  `diffu-moss-init`.

      diffu-moss-train-ref --transformer $BASE/transformer --cache data_flux3/cache_moss_ref_v3d \
          --val-cache data_flux3/cache_moss_ref_v3d_val --out runs/moss_ref_pilot/lora --steps 1500
      diffu-moss-export-ref --transformer $BASE/transformer \
          --checkpoint runs/moss_ref_pilot/lora/checkpoints/step_001500 --out runs/moss_ref_pilot/export_step_001500

- **The full fine-tune** (`--full`, `train/full.py`): every weight of the DiT, by MOSS-Image's own recipe (its
  technical report, Appendix A.2 and Table 7): Muon on the joint blocks' weight matrices (`train/muon.py`: Nesterov
  momentum 0.95, 5 Newton-Schulz steps, matched to AdamW's RMS; the fused gated feed-forward `ffn_up_gate`
  orthogonalised half by half, as the report steps fused matrices) and AdamW on the rest (betas 0.9, 0.95), both at
  5e-5; weight decay 0.01 but none on biases, norms and embedders; the last block's q/k norm weights decayed towards
  one; gradients clipped at 1. The weights stay in float32, the forward runs in bfloat16 autocast, the joint blocks
  checkpointed. On one GB10 that is about 73 GB at the peak (20.3 GB of weights, 20.3 of gradients, 19.0 of Muon
  momentum, 2.6 of AdamW moments, ~8 of activations at a 1 MP target and a 0.5 MP reference, ~3 for CUDA);
  `--muon-momentum-dtype bfloat16` keeps Muon's momentum in bfloat16 (stepped in float32): 9.5 GB less. A checkpoint is
  the DiT itself in bfloat16 (`checkpoints/step_NNNNNN/transformer/`, which `diffu-moss-init --transformer` draws with,
  no merge), written file by file through a memory map of its own (torch's CPU allocator kept 10 GB of copies after
  they were freed); `--ema 0.9999` keeps an average of the weights (20 GB more) and `--save-state` the state a run
  resumes from (`--resume <checkpoint>`, ~42 GB), both off by default: the Spark holds neither. `--transformer` also
  takes a checkpoint's `transformer/`: a new run from its weights (give it another `--seed`, or it draws that run's
  samples and noise again). `--limit-pairs N` trains on the cache's first N pairs drawn with their reference, and
  `diffu-draw-pairs` draws them back beside their real pages: an overfit test.

      diffu-moss-train-ref --full --transformer $BASE/transformer --cache data_flux3/cache_moss_ref_v3d \
          --val-cache data_flux3/cache_moss_ref_v3d_val --out runs/moss_ref_full/full --steps 500
      diffu-moss-init --base $BASE --transformer runs/moss_ref_full/full/checkpoints/step_000500/transformer \
          --out runs/moss/ref_full

- **Prompts encoded in the step** (`--encode-text`, either kind): the frozen text encoder, loaded as the cache loads it
  (`train/cache.py`: `PromptEncoder`), encodes each sample's prompt in its step, the cached embeddings bit for bit, so
  the cache holds the latents alone (`--latents-only`). It costs the encoder's 16.8 GB on each GPU.
- **Several GPUs** (`train/distributed.py`, the full fine-tune launched by `torchrun`, one node or several): each joint
  block an FSDP2 unit, every float32 weight, gradient and optimizer state sharded over the ranks, the weights gathered
  in bfloat16 and the gradients reduced in float32; a step's `--samples-per-step` (a multiple of the ranks) shared out,
  every rank drawing every sample's noise as one process does; Muon gathers each matrix's update whole and
  orthogonalises it on every rank; the main rank writes the log and the same checkpoints. At world size 2 the tiny
  models train, score, checkpoint and resume exactly as at world size 1 (`tests/test_distributed.py`, gloo on the
  CPU). Estimated per 80 GB A100: ~30 GB on 4 GPUs (47 with `--encode-text`, 52 with `--ema` too), ~22 GB on 8.

      torchrun --standalone --nproc-per-node 4 -m diffu_moss.train.loop --full --transformer $BASE/transformer \
          --cache data_flux3/cache_moss_pages --out runs/moss_pages_full/full --steps 500 --samples-per-step 8

## Jobs

Every GPU step goes through `jobs/spark_guard.sh` (one GPU job at a time on the DGX Spark, whose 128 GB are shared by
CPU and GPU).

- `jobs/moss/run_ref_pilot.sh`: the LoRA on the Spark: Base as a model directory, the caches, a 5-step smoke run, the
  training, the export and model directory, 8 held-out crops drawn with and without their reference.
- `jobs/moss/run_ref_full.sh`: the full fine-tune on the Spark: a smoke run, the overfit test (300 steps on 8 pairs,
  its loss and drawings read against stated pass criteria), the training, its model directory and the held-out crops.
- `jobs/moss/hpc_full.sbatch.example`: a Slurm template (no site names) for the full fine-tune over several GPUs; not
  yet run on GPUs.

## What keeps training and drawing consistent

The training reads a pair as drawing reads a crop: its prompt is compiled by `diffu_moss.layout_prompt` from the pair's
caption in the form drawing gives it (`diffu_moss.sample_caption`) and encoded by `MossTextEncoderStep` with the
components loaded as the page pipeline loads them, its latents by the reference encode step's functions.
`packages/page/tests/test_moss.py` checks a cached pair against the page pipeline's encoding of the same crop and
reference, bit for bit, and that a trained pair drawn back (`diffu-draw-pairs`) is the crop the page pipeline draws;
`packages/moss/tests/test_train.py` that `--encode-text` encodes what the cache holds.

## Hand LoRAs with KohakuMossTrainer

MOSS-Image hand LoRAs (one per volume) train with KohakuMossTrainer in its own environment
(`~/Desktop/repos/kohakumoss`), on manifests whose captions the scratchpad's `moss_prompt.layout_prompt` compiled:
`diffu_moss.layout_prompt` compiles a crop plan's caption to the same string (`packages/page/tests/test_moss.py`). The
trainer's text caches are written with transformers' forward, one prompt a call (`forward: hf`), as
`MossTextEncoderStep` encodes; its latents (the VAE's mode, 2x2, batch-norm statistics) matched ours at cosine 0.99998.
Its LoRA is merged into Base and drawn through `diffu-moss-init --transformer`.
