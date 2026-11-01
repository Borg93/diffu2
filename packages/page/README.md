# diffu-page

Archive pages as data, and the page pipeline built on them ([docs/PIPELINE.md](../../docs/PIPELINE.md)).

- **Data** (CPU): a real PAGE or ALTO page read as its lines (`layout/`), new text refilled into it, the
  training-scale crops it is cut into (`crops/`), each crop's layout prompt (`captions/`), its labels (`labels.py`:
  PAGE/ALTO, and for page readers Markdown, DeepSeek-OCR 2's grounded Markdown, PaddleOCR-VL's spotting and layout
  elements), and the builders of the training and evaluation sets (`data/`, each `python -m diffu_page.data.<name>`):
  `build_crops` (FLUX.3's training windows), `pick_eval` (their held-out set), `style_pairs` (a crop and a reference
  crop of the same hand, for FLUX.3's style-reference LoRA and MOSS-Image), `page_pairs` (whole pages with a sample
  page, for MOSS-Image), `page_pair_forms` (a page-pair manifest's rows given every prompt form) and `print_hand_set`
  (the print/hand classifier's lines).
- **Marking** (`marking/`, CPU): a ground-truth template marked from its scan: its lines read as printed or
  handwritten by a small CNN (`print_hand.py`, trained by `python -m diffu_page.marking.train`) and its ruled lines
  found with OpenCV (`rules.py`), where its PAGE file says neither.
- **Blocks** (`blocks/`): our Modular Diffusers steps. Before any model: `LoadLayoutStep`, `MarkFromScanStep`,
  `OpeningPageStep`, `RefillStep`, `PlanCropsStep`. Around a model's own blocks: `CropPromptStep`,
  `ReferenceGridStep`, the line repair (`repair.py`) in its denoise loop; after it, the judge (`judge.py`) and the
  labels (`ground_truth.py`).
- **Pipeline** (`pipeline.py`): those blocks around whichever model's blocks a model directory names (FLUX.3 Klein or
  MOSS-Image), loaded with its LoRAs. CLIs: `diffu-generate` (pages), `diffu-evaluate` (held-out crops),
  `diffu-draw-pairs` (a MOSS reference training's own pairs drawn back beside their real pages: an overfit test) and
  `diffu-reader-set` (a run's pages as a page reader's fine-tuning set).
