# diffu-page

Archive pages as data, and the page pipeline built on them.

- **Data** (CPU): a real PAGE or ALTO page read as its lines (`layout/`), new text refilled into it, the
  training-scale crops it is cut into (`crops/`), each crop's layout prompt (`captions/`), its labels (`labels.py`:
  PAGE/ALTO, and for page readers Markdown, DeepSeek-OCR 2's grounded Markdown, PaddleOCR-VL's spotting and layout
  elements),
  and the builders of the training and evaluation sets (`python -m diffu_page.data.build_crops`, `pick_eval`,
  `steer_set`, `style_pairs`, `print_hand_set`).
- **Marking** (`marking/`, CPU): a ground-truth template marked from its scan: its lines read as printed or
  handwritten by a small CNN (`print_hand.py`, trained by `python -m diffu_page.marking.train`) and its ruled lines
  found with OpenCV (`rules.py`), where its PAGE file says neither.
- **Blocks** (`blocks/`): our Modular Diffusers steps. Before any model: `LoadLayoutStep`, `MarkFromScanStep`,
  `OpeningPageStep`, `RefillStep`, `PlanCropsStep`. Around a model's own blocks: `CropPromptStep`,
  `ReferenceGridStep`, then `LineReadbackStep` (HTR) and `LabelExportStep`. `CropLoopStep` draws every crop of a page.
- **Pipeline** (`pipeline.py`): those blocks around whichever model's blocks a model repo names (FLUX.3 Klein or
  diffu), loaded with its LoRAs. CLIs: `diffu-generate` (pages), `diffu-evaluate` (held-out crops), `diffu-sheets`,
  `diffu-reader-set` (a run's pages as a page reader's fine-tuning set).
