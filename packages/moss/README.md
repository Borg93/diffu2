# diffu-moss

MOSS-Image (OpenMOSS, Apache-2.0) as the page pipeline's third model family, beside FLUX.3 Klein and diffu: our own
implementation, written from the published design, around the released weights.

- **The DiT** (`transformer.py`): `MossTransformer`, 42 single-stream blocks over the image's and the prompt's tokens;
  the released Base transformer (or a merged LoRA or full fine-tune exported in diffusers format) loads into it by name.
- **The blocks** (`modular_pipelines/moss/`): `MossAutoBlocks` and `MossModularPipeline`. A crop plan's layout caption
  compiled as MOSS's Layout prompt (`layout_prompt.py`: a summary and one text item per line, its exact text in its
  pixel box), encoded by the frozen Qwen3.5-9B text encoder, denoised with classifier-free guidance whose
  unconditional pass skips a block, decoded by the FLUX.2 VAE (`latents.py`). The denoise loop has FLUX.3's step
  names, so the page pipeline's line repair runs in it.
- **A model repo** (`diffu-moss-init`): a `modular_model_index.json` naming `MossAutoBlocks`, the components loading
  from the Base snapshot's folders, the DiT from the snapshot or a trained export, and whether it takes a reference
  image (`takes_reference`: only for a DiT of a reference training, a merged LoRA or a full fine-tune's checkpoint,
  read from its `export.json`).

      diffu-moss-init --base <MOSS-Image-Base snapshot> --out runs/moss/base
      diffu-moss-init --base <MOSS-Image-Base snapshot> --transformer <merged export> --out runs/moss/hand_1715

- **Reference images** (sample pages): the DiT reads a reference's clean latents as tokens after the prompt's, on the
  reference axis (reference `k` at `k + 1`), its grid centred, modulated as at timestep 0 (FLUX.3 Klein's design);
  without one it is the released model bit for bit. The released models were not trained with references, so a model
  directory takes one only when its DiT was (`diffu_moss.train`, below); the page pipeline's `ReferenceGridStep` puts a
  reference on the grid it trained at (0.5 MP).
- **Reference training** (`train/`, our own loop): a LoRA (rank 32, alpha 32, the 1715 hand LoRA's modules and
  optimizer) on v3d's described style pairs (`data_flux3/style_pairs_v3d`: a target and a crop of its own or facing
  page, sharing none of its lines; the pairs whose reference repeats a line are left out). Each pair is cached in one
  prompt form: a fifth without the reference (its citation dropped, the look words kept), the rest with "the sample
  image", half of those without the look words (`sample_caption.py`), so the look must come from the image. MOSS's
  own noise levels (median 0.78) and flow-matching loss over the drawn image; the joint blocks checkpointed.

      diffu-moss-cache-pairs --model runs/moss/base --pairs data_flux3/style_pairs_v3d/train.jsonl \
          --out data_flux3/cache_moss_ref_v3d --limit 4000
      diffu-moss-train-ref --transformer <snapshot>/transformer --cache data_flux3/cache_moss_ref_v3d \
          --val-cache data_flux3/cache_moss_ref_v3d_val --out runs/moss_ref_pilot/lora --steps 1500
      diffu-moss-export-ref --transformer <snapshot>/transformer \
          --checkpoint runs/moss_ref_pilot/lora/checkpoints/step_001500 --out runs/moss_ref_pilot/export_step_001500

  `jobs/moss/run_ref_pilot.sh` runs the pilot through `jobs/spark_guard.sh`: the caches, a smoke run, the training,
  the export and model directory, and 8 held-out crops drawn with and without their reference.
- **Full fine-tune** (`diffu-moss-train-ref --full`, `train/full.py`): every weight of the DiT on the same pairs and
  caches, by MOSS-Image's own recipe (its technical report, Appendix A.2 and Table 7): Muon on the joint blocks'
  weight matrices (`train/muon.py`: Nesterov momentum 0.95, 5 Newton-Schulz steps, matched to AdamW's RMS; exactly
  torch's Muon, but the fused gated feed-forward `ffn_up_gate` is orthogonalised and RMS-scaled half by half, as the
  report steps fused matrices, its weights' names unchanged) and AdamW on the rest (betas 0.9, 0.95), both at 5e-5; weight decay 0.01 but none on biases, norms and the embedders; the last
  block's q/k norm weights decayed towards one; gradients clipped at 1. The weights stay in float32 and the forward
  runs in bfloat16 autocast, the joint blocks checkpointed: the layout FSDP2 wraps, with no master copy of our own.
  On one GB10 that is about 73 GB at the peak (20.3 GB of weights, 20.3 of gradients, 19.0 of Muon momentum, 2.6 of
  AdamW moments, ~8 of activations at a 1 MP target and a 0.5 MP reference, ~3 for CUDA and the process). A
  checkpoint is the DiT itself, in bfloat16 (`checkpoints/step_NNNNNN/transformer/`, which `diffu-moss-init
  --transformer` draws with, no merge); `--ema 0.9999` keeps an average of the weights beside it (`ema/`, 20 GB more
  in float32) and `--save-state` the float32 weights and optimizer state a run resumes from (`state/`, ~42 GB;
  `--resume <checkpoint>` goes on exactly as the run would have), both off by default: the Spark holds neither.
  `--limit-pairs N` trains on the cache's first N pairs drawn with their reference alone, and `diffu-draw-pairs`
  (diffu-page) draws them back beside their real pages: an overfit test.

      diffu-moss-train-ref --full --transformer <snapshot>/transformer --cache data_flux3/cache_moss_ref_v3d \
          --val-cache data_flux3/cache_moss_ref_v3d_val --out runs/moss_ref_full/full --steps 500
      diffu-moss-init --base <snapshot> --transformer runs/moss_ref_full/full/checkpoints/step_000500/transformer \
          --out runs/moss/ref_full

  `jobs/moss/run_ref_full.sh` tests it on the Spark through `jobs/spark_guard.sh`: a smoke run, the overfit test (300
  steps on 8 pairs, its loss and drawings read against stated pass criteria), the training, its model directory, and
  8 held-out crops drawn with and without their reference.
- **Page pairs** (`diffu_page.data.page_pairs`, diffu-page; the big full fine-tune's data): whole single pages of the
  handwritten GT (openings split into their pages), each cut at about 1 MP exactly as `diffu-generate --whole-page`
  plans a page and prompted as drawing prompts it, with a sample page of its own volume sharing none of its lines at
  0.5 MP. The sample carries paper and ink, not the hand, so every form names the hand (the archive and the volume's
  years): no reference (the page's own caption, its look in words), look (the sample, and the look in words:
  `--reference-caption named`) and plain (the sample, the hand alone in words: `--reference-caption hand`). Each row
  of `train.jsonl` and `val.jsonl` (`train/pairs.py`: `PagePairRow`) carries the page's Layout prompt in all three
  forms with its length in the text encoder's tokens; `diffu-moss-cache-pairs` reads either kind of manifest and fixes
  each pair's form by its seed as before. Pages past the encoder's 4,096 tokens are skipped and counted, never cut;
  whole volumes are held out (`val.jsonl`, and `val_eval/` for `diffu-evaluate`).

      python -m diffu_page.data.page_pairs --gt-root ~/Desktop/data/GT-Lejonet_HTR_202602 --model runs/moss/base \
          --out data_flux3/page_pairs --workers 8
      diffu-moss-cache-pairs --model runs/moss/base --pairs data_flux3/page_pairs/train.jsonl \
          --out data_flux3/cache_moss_pages
