# FLUX.3 Klein: the reference path

FLUX.3 Klein Base with our LoRAs is the research reference: what quality is reachable, and the target MOSS-Image is
judged against. It is never the model we ship (the FLUX Kommunity licence forbids training another model on its
outputs), so nothing of it flows into MOSS-Image. It draws through the same page pipeline ([PIPELINE.md](PIPELINE.md))
and trains its LoRAs with NeMo AutoModel.

## Drawing

A FLUX.3 Klein directory (diffusers' `Flux3AutoBlocks`, its prompt upsampler left out: it would rewrite the quoted
lines) is `--model`, or `FLUX3_MODEL` from the environment or a `.env` file. LoRAs are NeMo checkpoints' `model/`
dirs, loaded by the model's own LoRA loader as named adapters (`diffu_page.lora`: alpha/r folded into `lora_B`);
`--fuse-loras` fuses them into the transformer at load and `--compile-blocks` then compiles its repeated blocks.
Guidance is FLUX.3's 4 and the sampling shift 6.93 unless `--guidance` or `--sampling-shift` name others.

    cd ~/Desktop/repos/Automodel && ~/Desktop/repos/diffu2/jobs/spark_guard.sh --need-gb 45 --name pages -- \
        .venv-managed/bin/diffu-generate --templates page_a.xml --text-from text.txt \
        --styles ~/Desktop/repos/diffu2/styles.json --style pahlman --out runs/pages

`styles.json` holds a preset per hand LoRA (one LoRA per volume, trained on its transcribed pages; four pages already
carry the hand), drawn in the archive, years and look their evaluations drew them in. A style-reference LoRA draws
from a crop (`--reference <crop> --reference-caption described`, the crop's `metadata.jsonl` from `build_crops` beside
it), and `--reference-guidance` pushes toward the reference with a third pass. `diffu-evaluate` draws held-out crops
from their stored captions and scores them (read-back CER as information, the hand against the real crop's and its
reference's).

## Training a LoRA (NeMo AutoModel)

Training runs in NeMo AutoModel's environment (`~/Desktop/repos/Automodel/.venv-managed`), where
`jobs/setup_nemo_env.sh` installs diffu-core, diffu-moss, diffu-page and diffu-train; NeMo's own
`examples/diffusion/finetune/finetune.py` is the entry point, our dataloader configs its `_target_`s.

1. **Crops** (CPU): `python -m diffu_page.data.build_crops` cuts the GT pages into training windows with layout
   captions (whole lines, about 1 MP at a 64 px line pitch, NeMo bucket sizes; train and val splits by page);
   `python -m diffu_page.data.pick_eval` picks the held-out `val_eval` set from the val split, one crop per collection.
2. **Style pairs** (CPU, a style-reference LoRA only): `python -m diffu_page.data.style_pairs` pairs each training crop
   with a crop of its own or facing page sharing none of its lines, its caption describing the reference's measured
   hand, ink and paper (`--val-pairs 64` for NeMo's `val_loss`); MOSS-Image trains on the same pairs.
3. **Caches** (GPU): a text-to-image set by NeMo's own FLUX.3 preprocessing; style pairs by
   `python -m diffu_train.cache_pairs` (NeMo's FLUX.3 image-edit encoder on our manifest), which writes the empty prompt
   beside them (`empty_prompt.pt`) for caption dropout.
4. **Train** with a config of `packages/train/configs/flux3/`:
   - `flux3_layout_lora_v2.yaml`: the text-to-image LoRA (style by name: the caption names the archive, years and
     look), rank 32 on attention and feed-forward; the per-hand LoRAs of `styles.json` are this recipe on one volume's
     crops, 700 steps of 2 at a constant learning rate.
   - `flux3_style_ref_lora_v3d.yaml`: the style-reference LoRA (style by example: the reference's hand, ink and paper
     described in words beside `<ref_image_0>`), caption dropout 0.1, a constant learning rate, 3,000 steps, `val_loss`
     every 100. `jobs/flux3/run_v3d.sh` runs its whole chain through the guard: the pairs, the caches, a 5-step smoke
     run, the training and the held-out evaluation.

   Our dataloaders (`diffu_train.nemo`) interleave NeMo's aspect buckets, so any stretch of training sees every crop
   shape, and train a share of the items on the empty prompt (`caption_dropout`, `diffu_train.dropout`). A smoke run
   needs `--lr_scheduler.lr_warmup_steps 1` and an empty checkpoint dir (NeMo resumes from one it finds).

Every config logs to the local trackio project `diffu` (`~/.cache/huggingface/trackio`, no Space), the job scripts
naming the runs: `trackio list runs --project diffu --json`, `trackio get metric --project diffu --run <run> --metric
loss --json`.

## What keeps training and generation consistent

Checked 2026-10-21 (reference rows 2026-10-23); each would silently cost quality if it drifted.

| What | Training (NeMo) | Generation (Modular Diffusers) |
| --- | --- | --- |
| Caption | `diffu_page.captions` | the same code, from the `CropPlan`; from a reference, the same `reference_caption` call as its training pair (`test_plan`) |
| Prompt embeddings | NeMo's FLUX.3 encoder | FLUX.3's text encoder block: bit-identical on the same device |
| Latents | VAE `mode()`, 2×2 patchify, batch-norm stats | the same in FLUX.3's blocks |
| Line geometry | `page_xml.read_page_line` (builder) | the same reader (replay): 60/60 GT pages planned as cut |
| Reference image | shrunk to 0.5 MP on the 16 px grid (`crops.grid.as_reference`) | `ReferenceGridStep`, the same function |
| Reference look (described captions) | measured at the reference size (`captions.hand.measure_reference`) | the same function |
| Position ids | image (0, h·0.5, w·0.5, 0), reference l = −1, text (0, 0, 0, i) | the same |
| No-prompt input | v3d: a tenth of the pairs on the empty prompt as NeMo encodes a caption (`diffu_train.dropout`), bit-identical to diffusers' unconditional input; earlier runs none (`cfg_dropout_prob: 0`; NeMo's own would use zeros) | the empty prompt |
