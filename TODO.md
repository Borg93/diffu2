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
- **FLUX.3 Klein with our LoRAs is the research reference**, kept as it is.

## To the HPC run (Wed/Thu 4-5 Nov)

- [ ] **The content pilot is running** (`~/diffu_queue/queue_pilot2.sh` -> `jobs/moss/run_content_pilot.sh`, from 2
      Nov 17:20): from Base on the 55-collection set without `GT-RA_landskapshandlingar_20260227` (the scans of
      `..._20260220` again), the content mix (no prompt names an archive, years or hand), legible samples, look and hand
      changes for half the sample-form entries each; 2,500 steps; then the hand swap test. It passes on the rule fixed in
      `~/diffu_queue/VERIFY_SAMPLE.md`: the sample's hand +0.30 over the page's own sample on at least half the swaps,
      the look kept (L1 4/5, L2 + L3 at least r6b's), and the pages beside the scans. Before it, r6b: L1 4/5, L2 3/5,
      L2 + L3 78 %, H1 1/12; r6c (r6b trained on) regressed (L2 0/5, H1 0/12).
- [ ] The recipe from the pilot's result: if it passes, the HPC run trains it on all collections; if not, look and text
      only, hands from per-hand adapters, and a separate path for the sample (a frozen base with a reference encoder,
      DiffSynth's template design) as the next research step.
- [ ] The full training set with that recipe: the page pairs of the whole GT (~59,000 single pages; the duplicate
      collection left out), their content forms and legible samples (`page_pair_forms`, `page_pair_samples`).
- [ ] An HPC runbook: the environment, the Base snapshot and the data on a filesystem every node reads (Leonardo's
      compute nodes have no internet), the submit script, a 1-node smoke test; `--ema` and `--save-state` there only.
      `jobs/moss/hpc_full.sbatch.example` is tested on CPU only (gloo, world size 2, tiny models). Estimates for one
      pass over 200,000 pages: ~23 h on 8 A100s or ~9 h on 3 B200s at 1 MP, ~51 h / ~19 h at 2 MP; ~30 GB a GPU on 4
      A100s, ~22 GB on 8. The legible 1 MP sample adds ~25 % to a step.
- [ ] Optional: the old recipe on the 55 collections (full_m1, killed by the memory guard at step ~370 on 2 Nov) as a
      control, ~18 GPU hours.

## MOSS-Image drawing

- [ ] A content-mix model drawn without a sample should be prompted in its generic form (nothing named); diffu-generate
      captions a page drawn without a sample in its named style, and has no option for the generic form yet.
- [ ] A legible model draws a window only from a sample given with its line boxes (a job's `reference.lines`,
      `diffu-evaluate`'s rows); `diffu-generate --reference` gives none (whole pages need none), and the SGLang plugin
      takes a sample sized already.
- [ ] The four hand adapters (`Gabriel/diffu-moss-hand-*`) drawn through `--lora` on a GPU, fused and not, looked at
      beside the pages their merged DiTs drew (`runs/moss_hand_*/windows_step*`) and the scans; then a presets file of
      the MOSS hands beside `styles.json`, and an adapter in the studio.
- [ ] Reference guidance (a third pass toward the sample, off by default) never tried for the hand: scales 2 and 3 on
      the swap set, ~3.3 GPU hours; passes if the hand follows in at least 4 of 12 swaps with L1 and L2 + L3 kept.
- [ ] A blind realism rating of real and drawn pages mixed, more raters (the first, 27 Oct, picked the real scan in 12
      of 12 sets) and several seeds a page; what gives the scans away decides what to train next.
- [ ] Not looked at yet: the 27-29 Oct queue's pages (microfilm LoRAs, the print mix, the blank-page canvas test, the
      1715 LoRA at guidance 2 and 2.5).
- [ ] Drawn ink on the real page's paper (scratchpad `paper/paper_composite.py`), and writing no label asks for
      (scratchpad `hand/unlabelled_words.py`): each as a stage after drawing, checked in a blind test.
- [ ] KohakuMossTrainer's glue (manifests, caches, configs, checks) still lives in the scratchpad.
- [ ] A sample-page input in the studio for a model trained with references.
- [ ] Drawing checks `--reference-caption` against the form a model trained on, once training records it.
- [ ] Training never sees the negative prompt guidance draws against (MOSS's demo default): caption dropout in training
      would give guidance a trained unconditional branch.

## Inference speed

- [ ] The speed sweep (queued after the pilot): UniPC and Euler at 20 steps, guidance stopped at 80 %, FirstBlockCache,
      TaylorSeer, each against the 50-step pages (writer ID, colour, pixel difference, by eye).
- [ ] **Few-step distillation** of the final model after the HPC run: on-policy distillation as DiffSynth-Studio's
      `examples/moss_image/model_training/special/opd` does it (teacher 50 steps at guidance 3, student 8 steps at
      guidance 1, a rank-64 LoRA, lr 1e-4), in our trainer and with the sample page in both; ~12x fewer DiT passes a
      page. Passes only if the 8-step pages match the 50-step ones on read-back, writer ID, look, the swap test and by
      eye.
- [ ] Profile on the cluster's Blackwell GPUs (Morgan, `~/diffu_queue/morgan_test.tar.gz`): bf16 against MXFP8 +
      compiled blocks, actors per card; then NVFP4 and fused QKV projections. The MXFP8 AOTInductor package fails at run
      time (PyTorch's `isIntList()` assert); compiled blocks work.
- [ ] Qwen3.5's linear-attention layers run in transformers' reference code (`causal-conv1d` and
      `flash-linear-attention` are not installed), which slows caching and drawing.
- [ ] A crop is drawn one per call and one actor does everything in turn: batch same-size crops, and split the work into
      stages with their own resources (encode, denoise, read-back, labels and sink); the line repair as such a stage.
- [ ] `diffu-run` writes ALTO and PAGE only: the page readers' formats come from `diffu-reader-set --xml` on its export.

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

## FLUX.3 Klein (the reference, paused)

- [ ] Open from the LoRA studies: the style-reference LoRA (`packages/flux3/configs/flux3_style_ref_lora_v3d.yaml`)
      never trained (rebuild `data_flux3/style_pairs_v3d` with the overlap fix first); caption probes on base FLUX.3;
      `norm_q`/`norm_k` trained in bf16 but drawn in float32; the caption-dropout wiring untested in NeMo's environment;
      Klein's tokenizer regex warning; guidance 4 with shift 6.93 (the hand LoRAs' ink is truest at 2).

## Repo

- [ ] Publish our blocks to the Hub (`diffusers-cli custom_blocks`) if others should use them.
- [ ] The runner's draw test should call `run()`; its direct imports (numpy, pillow) declared.
