# diffu-studio

The page pipeline as a `gr.Workflow` canvas, for MOSS-Image models. A **plan** node (a PAGE/ALTO template, its new
text or its own, a style as JSON) feeds one **draw** node per drawer, a MOSS-Image model directory
(`diffu-moss-init`'s): Base, a hand LoRA merged into it, a full fine-tune. So the models draw the same page from the
same noise, side by side. Each draw node returns a sheet of the crops (line boxes outlined) and a zip of the crops with
their ALTO labels.

The canvas also has an app view (a form; `?ui=app`) and a REST endpoint, `/<first drawer>_crops`.

    diffu-studio --out runs/studio --drawer base=runs/moss/base --drawer hand=runs/moss/hand_1715

The graph is rebuilt from the drawers at every start (`<out>/workflow.json`). Edit it on the canvas through the
write-access URL `launch` prints, and keep that URL private. On the DGX Spark the studio is a GPU job like any other:
start it through `jobs/spark_guard.sh`.
