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
`Gabriel/diffu-moss-above`, published as
[Gabriel/diffu-archive-pages](https://huggingface.co/spaces/Gabriel/diffu-archive-pages) (private while it is tested):
a sample page and the settings in a band on top; under it a page editor of our own (boxes drawn on the page or loaded
from a template, each a block of line slots with its text) and, beside it at the same size, the drawn page with its
ALTO and PAGE labels, every line read back, and its record (the prompt and the settings it was drawn with).

- `space/pagedemo/document.py`: the editor's page as the pipeline's `Layout` (each box's slots exactly the boxes the
  model is given, its text written into them as `refill` writes it).
- `space/pagedemo/sample.py`: an uploaded sample's line pitch, measured from its rows, so the sample is cut at the
  page's writing scale as the model's training samples were.
- `space/pagedemo/planning.py`, `drawing.py`: the page planned whole in the `sample` caption form, and drawn by the
  crop pipeline, by the model's own guided steps or by a distilled student's (`diffu-moss-distill`'s checkpoint,
  loaded unmerged).
- `space/pagedemo/editor.js`, `editor.html`, `editor.css`: the editor, a Gradio custom HTML component (mouse and
  touch; undo; a page saved to and opened from a JSON file).
- `space/pagedemo/record.py`: a drawn page's record, which the editor opens as that page again.
- `space/tests`: the Space's own code on CPU.

`publish_space.sh` stages and uploads it: the Space's files, the sample pages and the student's checkpoint (both kept
outside the repo), and `requirements.txt` with the diffu packages pinned to the checkout's commit on `origin/main`.
`PAGEDEMO_STUB=1 python space/app.py` serves the page without the model, for work on the editor.

On ZeroGPU (20 Nov 2026, half an RTX PRO 6000): the models load and pack in about three minutes (28.6 GB), a fast
2 MP page is drawn in 24 to 26 s and its lines read back in the same call. A free account hosts two ZeroGPU Spaces
and gives a visitor five GPU minutes a day: a fast page asks for 56 s of them, a guided one for about 200.
