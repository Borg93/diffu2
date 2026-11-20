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

## The demo Space

`space/` is a Hugging Face Space (Gradio, ZeroGPU) on the same pipeline, for the fine-tuned model
`Gabriel/diffu-moss-above`: a page editor of our own (boxes drawn on the page or loaded from a template, each a block
of line slots with its text), a sample page, and the drawn page back with its ALTO and PAGE labels, every line read
back.

- `space/pagedemo/document.py`: the editor's page as the pipeline's `Layout` (each box's slots exactly the boxes the
  model is given, its text written into them as `refill` writes it).
- `space/pagedemo/sample.py`: an uploaded sample's line pitch, measured from its rows, so the sample is cut at the
  page's writing scale as the model's training samples were.
- `space/pagedemo/planning.py`, `drawing.py`: the page planned whole in the `sample` caption form, and drawn by the
  crop pipeline, by the model's own guided steps or by a distilled student's (`diffu-moss-distill`'s checkpoint,
  loaded unmerged).
- `space/pagedemo/editor.js`, `editor.html`, `editor.css`: the editor, a Gradio custom HTML component.
- `space/tests`: the Space's own code on CPU.

`publish_space.sh` stages and uploads it: the Space's files, the sample pages and the student's checkpoint (both kept
outside the repo), and `requirements.txt` with the diffu packages pinned to the checkout's commit on `origin/main`.
`PAGEDEMO_STUB=1 python space/app.py` serves the page without the model, for work on the editor.
