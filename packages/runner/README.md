# diffu-runner

Generation at scale on a Ray cluster. A run is a JSON-lines file of `diffu_page.draw.jobs.PageJob`s (a template page or
a layout given inline, the text to write into it or none, the style, a sample page to draw in the hand of, a seed and an
id). Ray Data plans every page whole on CPU workers (an opening becomes its two pages), GPU actors each holding one
pipeline (MOSS-Image's or FLUX.3 Klein's, with the run's LoRAs: `--lora hand=<adapter>`) draw, judge and label them,
and the driver commits them to a Lance table, `<out>/pages.lance`: one row per page, the image as a Blob v2 column, the
labels as ALTO and PAGE XML. A page that fails is a row with its error. A run into an `--out` that holds a table
resumes it: drawn pages are skipped, failed ones drawn again.

    diffu-run --jobs run.jsonl --out /outputs/run --model /models/moss --gpus 8 --actors-per-gpu 1
    diffu-run-export --out /outputs/run --to /outputs/run/files --limit 20

The hand-off guide (models, image, KubeRay, outputs, resuming, throughput): docs/INFERENCE.md. How Ray Data runs,
retries and fails the stages: docs/RAY.md.
