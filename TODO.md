# TODO

What is known to be wrong or unfinished. Remove an item when it is done; the commit that does it says so. Everything
done before, with its measurements, is in git history (`git log -p -- TODO.md`).

## Direction

- **The product**: millions of synthetic whole pages with their ALTO/PAGE, to train a page-reading VLM end to end
  (**Unlimited-OCR 2**, fine-tuned with ms-swift; MiniCPM-OCR 1.0 as the region-crop control). Any template (a PAGE/ALTO
  file or a box layout) x any text x any look and hand (a sample page, or a hand's LoRA), each chosen independently and
  aimed at what the reader reads badly.
- **MOSS-Image is the model** (Apache-2.0; our own implementation and training). The prize: a MOSS-Image fine-tune
  that draws any template in the look and hand of a sample page, judged by eye beside the real scans (realism and the
  sample's hand first; CER is information only).
- **Where it stands** ([docs/MOSS_RUNS.md](docs/MOSS_RUNS.md), "What matters now"): the sample-above model
  (`Gabriel/diffu-moss-above`, 2 MP) writes the layout and text as given and takes the paper, the kind of scan and the
  pen's weight from the sample; the letter forms follow on some kinds of page only.

## The model: the hand and the ink

- [ ] **Experiment to try (owner, 21 Nov): a sample per text-region box, from the same page** (regions, not lines: a
      region is a target's lines grouped by their PAGE TextRegion). Code committed (1c679b6): the CPU builder
      (`python -m diffu_page.data.region_crop_pairs`), the crop cache (`diffu-moss-cache-region-crops`, sharded) and
      `--region-crops over|beside`. Decided (22 Nov): all crops on reference index 1, windows only. Crops built whole-region (the first
      build's 512 px corners set aside), looked at (9,472 windows, 12,011 crops; MOSS_RUNS), encoded; both arms trained and drawn
      (23 Nov): on four held-out single-region windows they write as the whole-sample model does (over the box the
      cleaner arm), so the DiT takes crops at no loss, but that test gives both models the same information. On the Space's
      templates with a sample of another document (23 Nov, Spark) a crop per box gives no gain: the letter the same
      page as with the whole sample, the register's names garbled, the faint sample not followed. Not shown to help
      after 480 steps; the owner asked for longer: the over arm for 2,000 steps (`region_long`) runs on Leonardo
      from 25 Nov 07:40 (the first queued job was cancelled by the cluster during the maintenance). Left: its pages on the four held-out windows and on the templates
      beside the pilot's; a held-out set of windows with several regions (only 4 of 18 validation windows have a
      sibling sharing no line); crops cut from the sample's own regions when drawing a template. Today one sample page goes in
      whole and nothing ties a line of the new page to anything in it; the pairs are pages of one volume, so the hand
      is shared only where the volume has one scribe. Instead give each line box (or a few) a crop of a real line,
      placed over its box (`reference_place: over`, one reference index each), and train on real pages whose other
      lines are the crops: the same page is almost always one hand, so every pair is certainly the same scribe.
      MS-Diffusion (ICLR 2025, arXiv 2406.07209) binds reference images to layout boxes in the same way. Steps: a
      pair builder from the existing caches, a pilot fine-tune from step 10,257 (as the 480-step pilots), judged by
      eye on held-out pages and the Space's templates beside the whole-page sample; then the Space and pipeline taking
      a sample per box (or a sample page cut into line crops). Line crops cost ~100-300 tokens each.
- [ ] The Space's path follows the sample less than drawing a real page does (heavy black headings from a thin grey
      sample, an invented dark border and page stack): check the full model against the student on the same
      template pages, then the template layouts, then the sample zoom.
- [ ] Letter forms follow the sample on short-entry pages (the 16th-century accounts) and not on running text, and not
      at all on the demo Space's new pages (the same letter from four samples in one hand). Not tried: writer-ID
      supervision in training, training weighted toward low noise (`--sigma-shift`) on a full run, more scribes.
- [ ] A faint sample's ink is beyond the model's range: from the 1687 sample (ink L 48.7) the model itself draws L 17-21
      at every guidance that writes, and at guidance 1 no line is readable. No student of this model can follow it:
      the model itself has to learn it.
- [ ] Caption dropout to the empty prompt, 2,000 steps (MOSS_RUNS 21-22 Nov): on three held-out pages with their own
      layouts the dropout arm's ink came nearest the sample's, but on the Space's templates (22 Nov, Spark) it follows
      the sample least: near-black ink in one neat hand whatever the sample, with cleaner lines. Not for the Space and
      not for the next full run as it stands. Open: why the two tests differ (the page's own layout and a sample from
      the same page, against a template and a sample of another document); held-out pages drawn with a sample of
      another document would tell.
- [x] Settings that do not fix the ink (21 Nov, MOSS_RUNS): the block-22 skip and the tag negative prompt, projected
      guidance, guidance on a student, the sample's colours as a prompt palette.
- [ ] Compiled joint blocks in training under FSDP2 (`compile_loop.py`, 30 steps at 2 MP from B's step 7,000, the 9 Nov
      probe's samples): 11 Nov 15.0 s a step against 18.4 eager, the same losses; 21 Nov with the leaner RMSNorm eager
      17.4 s (losses equal to the 9 Nov probe's to the third place), compiled failed in Inductor's C++ wrapper build on
      the compute node (`CppCompileError`, job 65020189): find why before a run uses it.
- [ ] Six of the 30 template jobs (dense registers and tables) are refused: their prompts pass the text encoder's 4,096
      tokens.
- [ ] On the Space's path lines flagged printed are written by hand and the short ruled lines under them left out.
- [ ] A blind realism rating of real and drawn pages mixed, with the sample-above model's pages, more raters and
      several seeds a page (the first, 27 Oct, picked the real scan in 12 of 12 sets); what gives the scans away
      decides what to train next.
- [ ] Drawn ink on the real page's paper (scratchpad `paper/paper_composite.py`), and writing no label asks for
      (scratchpad `hand/unlabelled_words.py`): each as a stage after drawing, checked in a blind test.
- [ ] **Idea from the HTR side (25 Nov; not started, for the MOSS session to judge): RL on the generator with our page
      reader as the reward.** Neither MOSS-Image (its technical report, arXiv 2610.23374: pretraining, then Base and
      SFT; no RL) nor, as far as published, FLUX.3 used RL. A drawn page is asked to say a text; the HTR reader
      (Nemotron Parse, `runs/htr/nemotron-parse-selftrain-finetune2/output/checkpoint-3000`, 6.63 % CER on the GT
      test) reads it back, and the reward is its character accuracy against the asked text, with the block boxes, and
      a realism term against reward hacking (DINO or TIPS distance to real pages, or a writer network: MOSS_AUDIT's
      point 5). Several seeds a prompt give the group RL needs. 2026 work on exactly this kind of reward: "Read It
      Back" (arXiv 2607.11886, the prompt's likelihood under a reader as reward), TDM-R1 (2603.07700, RL for few-step
      models with non-differentiable rewards, text rendering among them: fits the 8-step student), "RL post-training
      in DiTs is activation steering" (2611.26627: most of the gain is a constant shift, so a pilot can be small).
      None does handwriting. First step needs no training: draw a few hundred pages, score them with the reader and
      check the score orders them as the eye does. Risks: the generator learns writing the reader likes rather than
      the archive's; the reader and generator drift together (judge on real test pages only); each rollout is a whole
      drawing plus a reading. The HTR side's RL on the reader is CURATION E71, E72.

## Distillation

- [ ] **A better student (owner, 21 Nov).** Student e's live pages smear, strike through and misspell words; the Space draws
      guided by default until a student is good. Student i (e's recipe + `--matching 1.0`, DMD in velocities, 23 Nov)
      writes clean pages: no struck-through words or scribble, 98 of 106 lines read (e: 94); it follows the sample as
      little as e. In the Space as the fast student (23 Nov). A live page at
      another seed still strikes a line through: left, a stronger matching weight (4, 16), judged over several seeds a
      page.
      A published 4-step LoRA of MOSS-Image-SFT merged into our DiT (24 Nov, Spark) draws our pages at 4 steps: 103 of
      106 lines read, no struck minutes lines, lighter ink, but flat clean paper, soft strokes and garbled small words
      (MOSS_RUNS). Left: a student of ours warm-started from it, trained against our teacher with its recipe's
      settings (matching alone or beside the regression, learning rate 5e-6, beta1 0, one fake update a step), at 4
      steps; and several seeds a page for it as it is.
      Student i's own checkpoints (24 Nov) are level from step 200 to 400, the same lines struck: not longer, another recipe.
- [ ] **An Old Swedish LoRA as an option in the Space (owner, 25 Nov).** A LoRA trained on all of the owner's Old Swedish
      HTR set (charters and five books) on the above DiT, kept apart from it (MOSS_RUNS, "An Old Swedish LoRA"). Steps:
      (1) done 26 Nov: trained to step 750 on the Spark; (2) done 26 Nov: drawn beside the real pages; it learns the
      medieval hand, not yet the parchment look; step 750 picked; (3) the Space loads it as its own switch (as it loads the student), off by default, the page
      drawn as today without it; (4) tests, published, checked on ZeroGPU.
- [ ] **A 4-step fast mode in the Space (owner, 24 Nov: keep it in the backlog).** Not uploaded yet: the published 4-step
      LoRA on our DiT draws flat paper. Steps: (1) done 24 Nov: at three more seeds it stays cleaner than student i
      (215 against 205 of 219 lines, no struck line) but its seeds of a template are nearly one page; (2) a 4-step student of our own (owner, 24 Nov: our LoRA, not theirs): student i shortened to 4 steps against our teacher with the matching-only recipe
      (the whole rollout corrected, learning rate 5e-6, beta1 0, one fake update a step), checked for pages that vary;
      the code is in (`--matching-whole --matching-pull velocity --matching-level-beta 2.5 --beta1 0 --start`, 24 Nov), and a
      Spark trial ran 3 steps (about 3.5 min a page); student j runs on Leonardo from 25 Nov 08:39 with gradients
      through its last 2 steps (all 4 ran out of memory on a large page; about 50 s a step, 20 a job, about 30 jobs);
      left: if it works, whether gradients through all 4 (fewer tokens a page, or 80 GB cards) do better; done 26 Nov: student j
      (step 600) is the cleanest of the three (218 of 219 lines, no struck line), keeps our ink's strength and varies
      between seeds as student i does (MOSS_RUNS, "Student j"); (3) the better of
      the two in the Space as the fast mode, or as a third choice beside the 8-step student if the Space's 1 GB of
      storage holds both.
- [ ] Students g and h (`--swap 0.5`, teachers at guidance 2.0 and 1.5; queued 20 Nov 22:55): draw them on the held-out
      pages and on the Space's six template pages beside d and e at step 400 and the first student, which the Space
      keeps until a student's ink is closer to the sample's there.
- [ ] Student f (teacher at guidance 1.25) waits for room in the debug queue.
- [ ] The first student's text was never checked against the labels on the held-out pages.

## MOSS-Image drawing

- [ ] A converter of `train_log.jsonl` to TensorBoard for model cards lives in the scratch queue
      (`hf_publish/to_tensorboard.py`); into the repo, now that a second model is published.
- [ ] A content-mix model drawn without a sample should be prompted in its generic form (nothing named); diffu-generate
      captions a page drawn without a sample in its named style, and has no option for the generic form yet.
- [ ] A legible model draws a window only from a sample given with its line boxes (a job's `reference.lines`,
      `diffu-evaluate`'s rows); `diffu-generate --reference` gives none (whole pages need none), and the SGLang plugin
      takes a sample sized already.
- [ ] The four hand adapters (`Gabriel/diffu-moss-hand-*`) drawn through `--lora` on a GPU, fused and not, looked at
      beside the pages their merged DiTs drew (`runs/moss_hand_*/windows_step*`) and the scans; then a presets file of
      the MOSS hands (`--styles`), and an adapter in the studio.
- [ ] Not looked at yet: the 27-29 Oct queue's pages (microfilm LoRAs, the print mix, the blank-page canvas test, the
      1715 LoRA at guidance 2 and 2.5).
- [ ] KohakuMossTrainer's glue (manifests, caches, configs, checks) still lives in the scratchpad, and the
      sample-above set's builders (`sibling_pairs.py`, `sibling_cache.py`) with the runs on the cluster.
- [ ] A sample-page input in the studio for a model trained with references.
- [ ] Drawing checks `--reference-caption` against the form a model trained on, once training records it.

## Inference speed

- [ ] The demo Space's blocks compiled ahead of time (1.5x a pass on the Spark, the same pages; MOSS_RUNS 21 Nov): the
      Space builds the package at start-up on its own GPU and runs both DiTs from it (student merged into a copy).
      Live 21 Nov: a fast page drawn in 14-15 s (23 s eager). To do: a guided page live, and re-set the GPU seconds
      the Space asks for (`SECONDS_PER_PASS_KILOTOKEN`) to the compiled speed.
- [ ] Choose the attention kernel (`sdpa_kernel`: cuDNN, flash) and measure it with a sample on the page: attention is
      about 35-45 % of a compiled pass at our 10-16k tokens (the diffusers reading, 21 Nov). Exact, no fidelity cost.
- [ ] The step cache with warm-up steps and a cap on skips in a row (as Cache-DiT's 4 and 3), thresholds 0.05-0.24,
      on top of the compiled package; judged on the pages beside the samples.
- [ ] `torch.compile`/AOTInductor with `emulate_precision_casts=True` (TensorRT-LLM VisualGen's FLUX.3 setting): may
      bring a compiled page closer to the eager one.
- [ ] MXFP8's scale rounds down (`mxfp8.quantize`: amax/scale in [256, 512) while E4M3 stops at 448, so a block's
      largest values are clipped by up to 12.5 %): try NVIDIA's rounding up (RCEIL), then judge MXFP8 again against
      bf16 compiled with a sample on the page (KohakuMossTrainer's MXFP8 keeps velocity cosine 0.998 on a 5090).
- [ ] The speed numbers of compile and MXFP8 were taken without a sample on the page: re-measure with one.
- [ ] Guided cond and uncond passes as one batch-2 call (TensorRT-LLM packs them): fewer weight reads a step.
- [ ] UniPC in 16 steps and the first-block step cache each drew a guided page in 44-47 % less time on two pages and
      one seed: more pages and seeds before either becomes a default. TaylorSeer and MagCache are not measured.
- [ ] MXFP8 (2.41x compiled, the page lighter in ink, some letter forms changed) is not judged against sample pages.
- [ ] The MXFP8 AOTInductor package fails at run time (PyTorch's `isIntList()` assert); compiled blocks work.
- [ ] `diffu-generate` has no flags for `precision`, `sampler`, `cache` or `aoti_package` (draw settings only).
- [ ] The leaner RMSNorm is not measured on an A100 or in a training step.
- [ ] Qwen3.5's linear-attention layers run in transformers' reference code (`causal-conv1d` and
      `flash-linear-attention` are not installed), which slows caching and drawing.

## Scaling (waiting: no scaling work until testing on one machine is done)

- [ ] Profile on the cluster's Blackwell GPUs (Morgan, `~/diffu_queue/morgan_test.tar.gz`): bf16 against MXFP8 +
      compiled blocks, actors per card; then NVFP4 and fused QKV projections.
- [ ] A crop is drawn one per call and one actor does everything in turn: batch same-size crops, and split the work into
      stages with their own resources (encode, denoise, read-back, labels and sink); the line repair as such a stage.
- [ ] `diffu-run` writes ALTO and PAGE only: the page readers' formats come from `diffu-reader-set --xml` on its export.
- [ ] The runner's draw test should call `run()`; its direct imports (numpy, pillow) declared.

## Line reader against the Swedish Lion TrOCR (dmlpai01)

What runs, what is queued, the numbers and the reasons: [docs/HTR_RUNS.md](docs/HTR_RUNS.md).

- [ ] `minicpm-all` to its end (around 01:00 Tue 10 Nov). Then the queue runs by itself: scoring, the reader published,
      Jev trained and published, the DSpark draft trained and published (`after_minicpm.sh`).
- [ ] Every test line (88,317) read by both readers, CER by collection and by page, each bad page's reason.
- [ ] Collections drawn at n^0.7, repeats capped at about 4, in `lion_data_prep`, and more lines in the scarce hands
      (diffu-moss LoRAs, curated batch lines). Then a continued fine-tune, measured by collection and out of domain
      (docs/HTR_RUNS.md, "The data itself": label hygiene, thin crops and overrunning labels measured as not worth it).
- [ ] Curation of the batch output (about 1M documents): a page-level detection check, then the agreement filter and
      Jev, per-collection coverage, a human-review sample, a machine-label mark in the set builder.
- [ ] Pretraining on the curated batch (plus diffu-moss pages in the weak hands), then fine-tuning on the GT.
- [ ] Jev choosing between our reading and TrOCR's (best of the two: 3.24 % against TrOCR's 4.54 %).
- [ ] The draft's served speed-up on lines and regions.

## Page readers

- [ ] Unlimited-OCR 2 (remote code allowed, pinned revision, its `save_results` path never called; transformers 4.52.1 in
      its own environment; ms-swift from main) and MiniCPM-OCR 1.0 (its own environment, transformers >= 5.21): base
      readings of the GT test pages (`--split-from-gt-test-lists`; pages the generator trained on excluded), scored by
      page CER in reading order and order-free, per line, layout (blocks, labels, order); errors sliced by collection,
      region kind and image quality. Then LoRA fine-tunes on real GT, then synthetic pages aimed at the weak slices (at
      most 1:1, compared against real-only on held-out real pages).
- [ ] Check the Unlimited-OCR 2 and MiniCPM-OCR sets against a few base-model outputs on our pages, for format only:
      the box grid (0-1000 as its post-processing clamps; v1's documentation says 0-999), whether `<PAGE>` follows the
      last block with no line break, and whether a paragraph's line breaks are kept.
- [ ] A reader set of the whole GT (~30,000 scans, about 1.7 hours on one core).
- [ ] htr-train ([docs/HTR_TRAIN.md](docs/HTR_TRAIN.md)) GPU smoke runs on the Spark, one guarded job at a time, after
      the smoke sets (10-20 training scans of at least 3 volumes: `diffu-reader-set --task lines|elements|parse`,
      `htr-train split-dev`, `htr-train jev-set`): `htr-train smoke` queues MiniCPM-OCR lines and regions, Jev and the
      Unlimited-OCR 2 LoRA in `~/diffu_queue/htr_smoke.sh`; each passes when a checkpoint is written and the eval loss
      is finite and falls. Check Jev's load log for a newly initialised `score` (its head must load).
- [ ] htr-train's envs on HPC (login node), the models staged (`HF_HUB_OFFLINE=1`), a one-node smoke of each full
      spec, then the full runs: MiniCPM-OCR lines and regions -> Jev -> Unlimited-OCR 2 -> the DSpark draft.
- [ ] Scaled or native line crops for MiniCPM-OCR: the same 2,000 test lines read by the base model both ways.
- [ ] The vllm env (x86 first; vLLM at or after #64105, its full commit) and the DSpark smoke
      (`configs/smoke_vllm/dspark.yaml`), only after a measurement shows decoding dominates line serving time.
- [ ] htr-train evaluation: predictions (`swift infer`; our generate-only loop for Unlimited-OCR 2), CER/WER by slice,
      Jev's AUROC, a Jev judge stage beside the read-back (its threshold a setting, default 0.5), publishing
      checkpoints private under `Gabriel/`.
- [ ] Does the TrOCR read-back reject correct lines? Measure the share of real held-out lines it tags unclear (CPU, ~10
      minutes); above 20 %, a line-verification judge (a VLM asked whether a line shows exactly its text) is worth
      building.

## Pages, templates and labels

- [ ] A run-file builder: page jobs written from templates (a catalogue of GT pages by kind: tables, columns, margin
      notes, headings) or generated box layouts, a text corpus and sample pages, in shards of tens of thousands of lines.
- [ ] Page jobs given as an inline layout or keeping their own text are checked on CPU only: draw each combination on a
      GPU and look at the pages beside a template's.
- [ ] New text goes into the running-text lines only: a template's table cells, margin notes and headings keep their
      own (a layout gives each of its lines its text), and text left over when the lines run out is dropped silently;
      return it, so a document continues on its next page.
- [ ] Text: new period Swedish text (the GT holds ~52 M characters; a million pages would reuse each ~17 times): an
      open model prompted with a few real excerpts of the collection first, judged by character 3-gram divergence
      against held-out real text (the floor: two real volumes of one collection).
- [ ] Tables and forms in training data: `build_crops` leaves table cells out; it needs ruled lines found on the scans
      (the GT has no SeparatorRegions) and a filter for the 135 of 249 tables whose cells overlap.
- [ ] Printed and mixed pages: printed GT pages and templates; printed lines read back by a printed-text OCR (the
      handwriting TrOCR reads a well-drawn letterhead at CER 0.71-0.80), and no unclear tag on lines too short to read.
- [ ] An opening's two pages both take the job's text from its start, so the right page repeats the left's first words.
- [ ] Openings whose lines or tables cross the fold (~1 %) are cut inside a page, and the 1,176 wide scans whose lines
      leave no gutter are refused: a fold check on the scan would place the cut.
- [ ] Marking from the scan: an opening's two page jobs mark it twice; wrong rules on plain pages; fraktur and
      typewriting under 0.9; HTR ground truth that labels printed headings as handwriting.
- [ ] Narrow columns get no crops (under the 640 px minimum when a margin note narrows a column).
- [ ] Lines cut by a window's top edge are captioned as whole lines with their full text (2,438 of 143,941 rows):
      count a line whole only if its x-height band is inside.
- [ ] Caption grammar: "carries 1 lines", "carries 0 lines of cursive handwriting"; change it only with the next caption
      rebuild (the trained models and `diffu_moss.sample_caption` read the grammar as it is).
- [ ] Templates from scans (YOLO26): a `DetectLayoutStep` beside `LoadLayoutStep` for a page image, writing the same
      `Layout`; checked on GT pages; `ultralytics` is AGPL-3.0 (a public studio needs compliance).
- [ ] Templates outside the GT: DocLayout-YOLO's Mesh-candidate BestFit as a template generator (reimplemented from
      arXiv 2410.12628; its code is AGPL-3.0), and real GT regions pasted into BestFit layouts as a baseline.

## Quality control of drawn pages

- [ ] Embeddings probed first (TIPSv2-L/SO400m, Nemotron-3.5-Embed-1B): does kNN find the right writer, does a linear
      probe tell real from drawn, as well as the writer-ID CNN? Each later detector adopted only if it beats a label-free
      kNN baseline.
- [ ] Training crops cut from the wrong image have their boxes on blank paper (fixed for Byggningskollegium's JPEG/TIFF
      pairs): check a whole set by reading its real crops back.

## Repo

- [ ] Publish our blocks to the Hub (`diffusers-cli custom_blocks`) if others should use them.
