# TODO

What is known to be wrong or unfinished. Remove an item when it is done; the commit that does it says so. Everything
done before 1 Nov 2026, with its measurements (the FLUX.3 LoRA studies, MOSS-Image's first validation, the audits, our
own FLUX.3-architecture model), is in git history (`git log -p -- TODO.md`).

## Direction

- **The product**: millions of synthetic whole pages with their ALTO/PAGE, to train page-reading VLMs end to end
  (DeepSeek-OCR 2, PaddleOCR-VL 1.6). Any template x any text x any style (a sample page's hand, ink and paper), each
  chosen independently and aimed at what the reader reads badly: handwritten running text, tables and forms, printed
  and mixed pages.
- **MOSS-Image is the model** (Apache-2.0; our own implementation and training). The prize: a MOSS-Image fine-tune that
  draws any template in the look and hand of a sample page, judged by eye beside the real scans (realism and the
  sample's hand first; CER is information only).
- **FLUX.3 Klein with our LoRAs is the research reference**: never trained on, never shipped.
- Our own model on FLUX.3's architecture (diffu) was set aside on 1 Nov: after its last run it drew word-like strokes
  but no letters.

## MOSS-Image: drawing from a sample page

- [ ] **The eight-hands full fine-tune ignored its sample pages** (31 Oct, full_r3 at step 1500): layout, text and the
      look in words follow the page, the hand by name carries for some volumes only, and the swap test
      (`~/diffu_queue/weekend/draw_swap.py`: one page and noise, three samples) draws the same page whatever the sample;
      validation scored every form alike. Every form named the hand and each volume has one look, so the name gave both
      away. The fixes are in: the sample form (c0d5eff, the sample alone, two fifths of the steps) and look changes for
      it (27c9fc9, `--restyle-share`). Running: full_r6b (`~/diffu_queue/queue_restyle2.sh`: from full_r6's step 250,
      750 steps, then the verify swap set, the held-out pages in the sample form and writer ID). It passes when the
      sample form's validation drops below the by-name form's and the swap test's pages follow their samples.
- [ ] The results of the 27-29 Oct MOSS queue (the microfilm LoRAs, the print mix, the eight-hands LoRA, the blank-page
      canvas test, the 1715 LoRA's step-350 pages and its guidance 2 and 2.5) are in `runs/` and the queue logs, not yet
      looked at beside the scans and written up.
- [ ] A blind realism rating of real, FLUX.3 and MOSS pages mixed, more raters (the first, 27 Oct, picked the real scan
      in 12 of 12 sets: neither model passes for a scan yet), and several seeds a page; what gives the scans away
      decides what to train next.
- [ ] The page-pair set of the whole GT (~59,000 handwritten single pages, ~60 GB; `diffu_page.data.page_pairs`) is not
      built; the mix of pages and windows (`--windows-per-page`) is the trainer's to draw.
- [ ] Drawn ink on the real page's paper (scratchpad `paper/paper_composite.py`: a template's scan kept, its transcribed
      strokes erased whole and inpainted, the drawing's ink matched to the scan's and multiplied on): on windows, in a
      blind test where a rebuilt page never sits beside its own scan, then as a stage after drawing.
- [ ] Writing no label asks for (scratchpad `hand/unlabelled_words.py`: ink outside every labelled box; a LoRA writes a
      caption's dates or an invented heading where real pages carry untranscribed marks): a judge stage, a learned line
      detector in place of the thresholds.
- [ ] The four hands drawn through their adapters on a GPU (`--lora`, fused and not), looked at beside the pages their
      merged DiTs drew (`runs/moss_hand_*/windows_step*`) and the scans: on CPU the adapters are exact (each layer's
      change, and fused weights rounded once, as the merges were), not yet looked at drawn. Then a presets file of the
      MOSS hands beside `styles.json` (FLUX.3's), and an adapter in the studio.
- [ ] KohakuMossTrainer's glue in the repo: its manifests, cache and train configs and checks still live in the
      scratchpad (`write_manifest.py`, `moss_encoders.py`, `moss_hand_*.py`, `check_cache.py`, `check_merged.py`).
- [ ] `--compile-blocks` for MOSS (the DiT names no repeated blocks; its per-sample loop and text mask would break a
      full graph); a sample-page input in the studio for a model trained with references.
- [ ] Qwen3.5's linear-attention layers run in transformers' reference code (`causal-conv1d` and
      `flash-linear-attention` are not installed), which makes caching slow; installing them needs the user's approval.
- [ ] Drawing checks `--reference-caption` against the form a model trained on, once training records it beside its
      checkpoints.

## The HPC run (MOSS-Image's full fine-tune on many GPUs)

- [ ] `jobs/moss/hpc_full.sbatch.example` is tested on CPU only (gloo, world size 2, tiny models). Estimates for one
      pass over 200,000 pages: ~23 h on 8 A100s or ~9 h on 3 B200s at 1 MP (+0.5 MP sample), ~51 h / ~19 h at 2 MP;
      per 80 GB A100 ~30 GB on 4 GPUs, ~22 GB on 8.
- [ ] Leonardo's compute nodes have no internet: the Base snapshot, the caches and the environment on a filesystem every
      node reads; `--ema` and `--save-state` there only.
- [ ] Which text-encoder forward is nearer an fp32 encode at our prompt lengths (KohakuMossTrainer's packed kernels gave
      cosine 0.991-0.994 against transformers' on ~1,000-token captions): the fast kernels matter for big caches.

## Page readers

- [ ] The loop: (1) where the reader is weak: it reads whole held-out real pages of every handwritten collection
      (scratchpad `readers/reader_eval.py`; PaddleOCR-VL with `use_cache=True` and at full size); (2) learn those hands;
      (3) draw pages aimed at what it misreads, on the volumes' own layouts; (4) oversample them; (5) fine-tune the
      reader on real and on real + synthetic pages, compared on held-out real pages (scratchpad
      `readers/finetune_paddle.py` overfits one page on CPU).

## Pages, templates and labels

- [ ] Tables and forms in training data: `build_crops` leaves table cells out; it needs ruled lines found on the scans
      (the GT has no SeparatorRegions) and a filter for the 135 of 249 tables whose cells overlap.
- [ ] Printed and mixed pages: printed GT pages and templates; printed lines read back by a printed-text OCR (the
      handwriting TrOCR reads a well-drawn letterhead at CER 0.71-0.80), and no unclear tag on lines too short to read.
- [ ] Text: other historical Swedish text; an LLM text generator in period style.
- [ ] An opening's two pages both take the job's text from its start, so the right page repeats the left's first words.
- [ ] Openings whose lines or tables cross the fold (~1 %) are cut inside a page, and the 1,176 wide scans whose lines
      leave no gutter are refused: a fold check on the scan (the binding's shadow, the page edges) would place the cut.
- [ ] Marking from the scan: an opening's two page jobs mark it twice; wrong rules on plain pages; fraktur and
      typewriting under 0.9; HTR ground truth that labels printed headings as handwriting.
- [ ] Narrow columns get no crops (under the 640 px minimum when a margin note narrows a column).
- [ ] Lines cut by a window's top edge are captioned as whole lines with their full text (2,438 of 143,941 rows, 21 % of
      crops): count a line whole only if its x-height band is inside.
- [ ] Caption grammar: "carries 1 lines", "carries 0 lines of cursive handwriting"; change it only with the next caption
      rebuild (the trained models and `diffu_moss.sample_caption` read the grammar as it is).
- [ ] Rebuild `data_flux3/layout_crops_v2b` before a new set is derived from it: records built before 24 Oct keep no
      caption template.
- [ ] Templates from scans (YOLO26): a `DetectLayoutStep` beside `LoadLayoutStep` for a page image (Riksarkivet's
      yolo26s region and line models, HTRflow's two stages) writing the same `Layout`; region roles through a class
      table (margin notes by geometry until then); checked on GT pages (line recall, box IoU, baseline offset);
      `ultralytics` is AGPL-3.0 (a public studio needs compliance); an image input in the studio.
- [ ] Templates outside the GT: DocLayout-YOLO's Mesh-candidate BestFit as a template generator (reimplemented from
      arXiv 2410.12628; its code is AGPL-3.0), and real GT regions pasted into BestFit layouts as a baseline.

## Quality control of drawn pages

- [ ] Embeddings probed first (TIPSv2-L/SO400m, Nemotron-3.5-Embed-1B): does kNN find the right writer, does a linear
      probe tell real from drawn, as well as the writer-ID CNN? Then a label-free baseline (a window's kNN distance to
      the real windows of its hand and layout), KumoAnomaly on the drawing stream and KumoTabular (real vs drawn), each
      adopted only if it beats the baseline.
- [ ] A VLM judge (Nemotron 3.5 Super VL) once remote code is decided (its serving recipes need `trust_remote_code`):
      numbered overlays, pairwise questions, JSON verdicts, a failed call no verdict; calibrated on the blind test's
      labels.
- [ ] Training crops cut from the wrong image have their boxes on blank paper (fixed for Byggningskollegium's JPEG/TIFF
      pairs): check a whole set by reading its real crops back.
- [ ] Production QC as Ray Data stages after drawing.

## FLUX.3 Klein (the reference)

- [ ] The style-reference LoRA (`packages/flux3/configs/flux3_style_ref_lora_v3d.yaml`, `jobs/flux3/run_v3d.sh`) has
      never trained; whether to, now that four pages of a volume give a hand LoRA, is the user's call. Before it trains:
      rebuild `data_flux3/style_pairs_v3d` with the overlap fix (3d4b0c6: 85 references change; the set predates
      `build.json`, so move it aside first), and hold out whole volumes for its eval (23 of 24 `val_eval` volumes are
      also in training).
- [ ] Caption probes, each a control probe on base FLUX.3 (same crops and seeds) before it changes training data: a
      new-content paper row; ink described physically; the script and pen named instead of "cursive"; colour names and
      hexes from one measurement; the archive and years as a colour grade; "aged" as a trigger word.
- [ ] NeMo moves the transformer to bf16 with `.to()`, so FLUX.3's `norm_q`/`norm_k` train in bf16 but are drawn in
      float32: measure whether it matters.
- [ ] A NeMo-environment test of the caption-dropout wiring (a 2-crop cache, a collated batch's `prompt_embeds`).
- [ ] Klein's Qwen3-VL tokenizer warns of "an incorrect regex pattern" (`fix_mistral_regex`): training and drawing load
      it alike; check BFL's pipeline before changing it.
- [ ] Guidance: we draw at 4 with shift 6.93 (diffusers' 3.5/6.93, BFL's code 4/16); the hand LoRAs' ink is truest at 2.

## Inference at scale (Ray)

The runner and the KubeRay hand-off: [docs/RAY.md](docs/RAY.md), [docs/INFERENCE.md](docs/INFERENCE.md).

- [ ] Profile MOSS-Image on the cluster's Blackwell GPUs (96 GB, sm_120) before optimizing: FP8 / NVFP4, fewer steps,
      compile, several actors per card, text encoding as its own stage, a memory snapshot of one crop to size the
      actors.
- [ ] `diffu-run --guidance` is a literal 4 (MOSS-Image draws at 3).
- [ ] A crop is drawn one per call and one actor does everything in turn: batch same-size crops, and split the work into
      stages with their own resources (read and encode, denoise, read-back, labels and sink).
- [ ] The driver reads every job into a list and seeds crops by position: read jobs with `ray.data.read_json`, a
      content-based page id naming and seeding its crops.
- [ ] The runner plans each job as it stands: give each page of an opening its own job
      (`diffu_page.draw.jobs.single_pages`), as diffu-generate does.
- [ ] `DrawActor` catches every exception as a failed crop, CUDA faults included: error rows for content errors only,
      the actor's process ended on a device fault so Ray restarts it.
- [ ] The sink is Lance (`lance_ray.write_lance`, the image as a Blob v2 column), one row per page keyed by its id; a
      run resumes from the committed table (Ray's checkpointing doesn't fit a Lance sink).
- [ ] The line repair as a stage (a row with rejected lines in, the repaired row out).
- [ ] The pair builders' process pools (`style_pairs`, `page_pairs`) as Ray Data maps.
- [ ] From NeMo Curator: a small stage contract (setup, `process_batch`, declared resources), deterministic task ids,
      failures as sentinels, atomic writes, composite stages, tuning by diagnosis; a spike on Curator itself later.
- [ ] The runner's `int(str(...))` round-trip and direct imports (numpy, pillow); its draw test should call `run()`.
- [ ] Page jobs given as an inline layout or keeping their own text are checked on CPU only: draw each combination on a
      GPU (template or layout, new text or its own) and look at the pages beside a template's.
- [ ] A run-file builder: page jobs written from templates (a catalogue of GT pages by kind: tables, columns, margin
      notes, headings) or generated box layouts, a text corpus and sample pages, in shards of tens of thousands of lines.
- [ ] New text goes into the running-text lines only: a template's table cells, margin notes and headings keep their
      own (a layout gives each of its lines its text), and text left over when the lines run out is dropped silently;
      return it, so a document continues on its next page.
- [ ] `diffu-run` writes ALTO and PAGE only: the page readers' own formats come from `diffu-generate --reader-labels` and
      `diffu-reader-set`.

## Repo

- [ ] Publish our blocks to the Hub (`diffusers-cli custom_blocks`) if others should use them.
