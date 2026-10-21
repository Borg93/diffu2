# diffu-runner

Generation at scale on a Ray cluster. A run is a JSON-lines file of `diffu_page.jobs.PageJob`s (a template page, the
text to write into it, the style). Ray Data plans every page into crop jobs on CPU workers (`PagePlanBlocks`), GPU
actors each holding one crop pipeline (`diffu_page.pipeline.crop_blocks`) draw them one crop per block (image,
per-line read-back, labels), and the rows are written as Parquet with a `run_meta.json` beside them. A crop that fails
is kept with its error, never dropped silently. Crops are named and seeded as `diffu-generate` names and seeds them.

    diffu-run --jobs run.jsonl --out runs/pages --lora v2=<ckpt>/model --drawers 4 --gpus-per-drawer 1
