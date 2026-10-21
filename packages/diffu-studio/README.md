# diffu-studio

The page pipeline as a `gr.Workflow` canvas. A **plan** node (a PAGE/ALTO template, its new text or its own, a style
as JSON) feeds one **draw** node per drawer, a model directory with its LoRAs. So FLUX.3 Klein, FLUX.3 + a LoRA and
diffu draw the same page from the same noise, side by side. Each draw node returns a sheet of the crops (line boxes
outlined) and a zip of the crops with their ALTO labels. Drawers on one model share it and switch its LoRAs.

The canvas also has an app view (a form; `?ui=app`) and a REST endpoint, `/<first drawer>_crops`.

    diffu-studio --out runs/studio --drawer base=<FLUX.3 dir> --drawer v2=<FLUX.3 dir> --lora v2.v2=<ckpt>/model \
        --drawer diffu=<diffu repo>

The graph is rebuilt from the drawers at every start (`<out>/workflow.json`). Edit it on the canvas through the
write-access URL `launch` prints, and keep that URL private. On the DGX Spark the studio is a GPU job like any other:
start it through `jobs/spark_guard.sh`.
