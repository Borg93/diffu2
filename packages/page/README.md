# diffu-page

Archive pages as data, and the page pipeline built on them ([docs/PIPELINE.md](../../docs/PIPELINE.md)). Its
subpackages are layers, listed here in import order: each imports only those before it, and all of them use
`output.py` (the consoles, image files), `settings.py` (where the models are) and `local_run.py` (the local commands'
loop over many items).

- **`page/`**, a page as data (CPU): a real PAGE or ALTO page read as its lines (`layout/`), new text refilled into
  it, the training-scale crops it is cut into (`crops/`), its labels (`labels.py`: PAGE/ALTO, and for page readers
  Markdown, DeepSeek-OCR 2's grounded Markdown, PaddleOCR-VL's spotting and layout elements), and a ground-truth
  template marked from its scan (`marking/`): its lines read as printed or handwritten by a small CNN (`print_hand.py`,
  trained by `python -m diffu_page.page.marking.train` on the lines `python -m diffu_page.page.marking.print_hand_set`
  cuts) and its ruled lines found with OpenCV (`rules.py`), where its PAGE file says neither.
- **`captions/`**: what a crop's layout prompt says: the prompt itself, the scan's look, the archive and years, a
  reference's measured hand and the captions drawn from it.
- **`quality/`**: drawn lines read back by an HTR model (`readback.py`), the judge whose verdicts the labels carry
  (`judge.py`), and how closely a drawn hand follows its real crop's and its reference's (`fidelity.py`).
- **`draw/`**: a page planned as crops (`plan.py`) and our Modular Diffusers steps (`blocks/`). Before any model:
  `LoadLayoutStep`, `MarkFromScanStep`, `OpeningPageStep`, `RefillStep`, `PlanCropsStep`. Around a model's own blocks:
  `CropPromptStep`, `ReferenceGridStep`, the line repair (`repair.py`) in its denoise loop; after it, the labels
  (`ground_truth.py`). The pipeline (`pipeline.py`) puts them around whichever model's blocks a model directory names
  (FLUX.3 Klein or MOSS-Image), loaded with its LoRAs (`lora.py`), guided toward a reference too (`guidance.py`);
  `styles.py` holds the style presets and `jobs.py` a run's page and crop jobs.
- **`data/`**: the builders of the training and evaluation sets (each `python -m diffu_page.data.<name>`):
  `build_crops` (FLUX.3's training windows), `pick_eval` (their held-out set), `style_pairs` (a crop and a reference
  crop of the same hand, for FLUX.3's style-reference LoRA and MOSS-Image), `page_pairs` (whole pages with a sample
  page, for MOSS-Image) and `page_pair_forms` (a page-pair manifest's rows given every prompt form); `evalset.py` is
  the eval sets' rows, which the builders write and diffu-evaluate reads.
- **`cli/`**: `diffu-generate` (pages), `diffu-evaluate` (held-out crops), `diffu-draw-pairs` (a MOSS reference
  training's own pairs drawn back beside their real pages: an overfit test) and `diffu-reader-set` (a run's pages as a
  page reader's fine-tuning set).
