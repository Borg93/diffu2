# diffu-page

Archive pages as data, and the page pipeline built on them.

- **Data** (CPU): a real PAGE or ALTO page read as its lines (`layout/`), new text refilled into it, the
  training-scale crops it is cut into (`crops/`), each crop's layout prompt (`captions/`), PAGE/ALTO labels, and the
  builders of the training and evaluation sets (`python -m diffu_page.data.build_crops`, `pick_eval`, `steer_set`,
  `style_pairs`).
- **Blocks** (`blocks/`): our Modular Diffusers steps. Before any model: `LoadLayoutStep`, `RefillStep`,
  `PlanCropsStep`. Around a model's own blocks: `CropPromptStep`, `ReferenceGridStep`, then `LineReadbackStep` (HTR)
  and `LabelExportStep`. `CropLoopStep` draws every crop of a page.
- **Pipeline** (`pipeline.py`): those blocks around whichever model's blocks a model repo names (FLUX.3 Klein or
  diffu), loaded with its LoRAs. CLIs: `diffu-generate` (pages), `diffu-evaluate` (held-out crops), `diffu-sheets`.
