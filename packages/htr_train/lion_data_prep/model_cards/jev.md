---
license: apache-2.0
base_model: openbmb/MiniCPM-OCR-Jev
base_model_relation: finetune
pipeline_tag: image-text-to-text
library_name: transformers
language:
  - sv
tags:
  - minicpm-ocr
  - jev
  - ocr-verifier
  - handwriting
  - historical-documents
  - archives
  - htr
---

# @NAME@

[MiniCPM-OCR-Jev](https://huggingface.co/openbmb/MiniCPM-OCR-Jev) fine-tuned as a proof-reader for Swedish handwritten
archive lines. Given a line's crop and a candidate transcription (with the candidate rendered as an image), it returns
the probability that the transcription matches the line. It can pick the best of several readings, rank readers, and
flag doubtful readings or labels for human review. It is part of [diffu](https://github.com/Borg93/diffu2), beside the
reader [@READER_NAME@](https://huggingface.co/@READER_REPO@).

## Training

- **Pairs:** from every GT-Lejonet training line (February 2026, the Swedish Lion's training side):
  - the line's crop with its GT text (a match);
  - about as many wrong candidates (rule perturbations of the GT text, and the line of the same page closest to it in
    length);
  - [@READER_NAME@](https://huggingface.co/@READER_REPO@)'s readings of 100,000 training lines, a match where the
    reading equals the GT after normalisation and a mismatch where it does not.
- **Rendering:** each candidate is rendered by diffu's renderer (`@RENDERER@`). Scores depend on the renderer, so render
  with the same one.
- **Recipe:** OpenBMB's (full fine-tune of the language model, the vision encoder and aligner frozen, learning rate
  1e-5, ZeRO-3), 2 epochs, 128 pairs a step, on two RTX PRO 6000 Blackwell GPUs. Not packed, unlike theirs: in
  Qwen3.5 a packed sample's linear-attention state leaks into the next one.
- **Checkpoint:** @STEP@, the last.

## Evaluation

On its held-out dev pairs at the last evaluation: @EVAL@.

This has not yet been measured on JevOCRBench, nor as a label filter on machine-transcribed lines.

## Use

Jev reads three things: the line's crop, a candidate text, and that text rendered as an image. Two of them must be
made exactly as in training, or the scores shift:

- **The prompt is diffu's own, in English**, not OpenBMB's: `diffu_htr_train.jev.prompt.model_prompt(text)`.
- **The rendering is diffu's renderer** (`@RENDERER@`): `diffu_htr_train.jev.render.render_text(text, font, size)`,
  one line of black text on white at the crop's size, in DejaVu Sans. The version names the font file's digest:
  `diffu_htr_train.jev.render.renderer_version(font)` must give the same string for yours.
- **The crop** is the line's bounding box, white outside its polygon, scaled to 48 pixels high (aspect kept).

Both functions are in [diffu](https://github.com/Borg93/diffu2):
`pip install "git+https://github.com/Borg93/diffu2#subdirectory=packages/htr_train"`.

So OpenBMB's Jev services (the cloud API, the Docker image, the SDK's render+score service) do not fit this model:
they render with their own renderer and send their own prompt. Serve it with vLLM as OpenBMB serves Jev (their
`docs/vllm_deploy_jev_en.md`), and assemble the prompt yourself:

```python
import math
from pathlib import Path

from PIL import Image
from vllm import LLM, PoolingParams

from diffu_htr_train.jev.prompt import model_prompt
from diffu_htr_train.jev.render import render_text

if __name__ == "__main__":  # vLLM starts its workers as new processes
    jev = LLM(
        model="@REPO@", runner="pooling", convert="classify",
        hf_overrides={"num_labels": 2}, limit_mm_per_prompt={"image": 2},
    )
    crop = Image.open("line.jpg").convert("RGB")  # masked, 48 pixels high
    text = "Anno 1649 den 6 Aprilis"
    rendering = render_text(text, Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), crop.size)
    [out] = jev.classify(
        [{"prompt": model_prompt(text), "multi_modal_data": {"image": [crop, rendering]}}],
        pooling_params=PoolingParams(use_activation=False),
    )
    no, yes = out.outputs.probs  # the two logits
    print(1 / (1 + math.exp(no - yes)))  # softmax(logits)[1]: the probability that the text matches the line
```

This is how diffu scored Jev in its pilot (AUROC on held-out test pairs). It has not yet been run against this
checkpoint as published.

- **The score is not calibrated.** There is no threshold to take as given: choose one on labelled lines of your own,
  by how many readings pass and how many of those are right, and check it again when the reader or the material
  changes, as OpenBMB advises for their Jev.
- **Compare scores within one setup.** A different renderer, font or prompt is a different input; scores from it do
  not compare with these.

## Licence

Apache-2.0, as its base model.
