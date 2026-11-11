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
  1e-5, ZeRO-3), 2 epochs, 128 pairs a step, unpacked, on two RTX PRO 6000 Blackwell GPUs.
- **Checkpoint:** @STEP@, the last.
- **Curves:** the training and evaluation metrics are in this repo's `runs/` folder (TensorBoard, the Training metrics
  tab).

## Evaluation

On its held-out dev pairs (from training lines) at the last evaluation: @EVAL@.

@TEST@

This has not yet been measured on JevOCRBench, nor as a label filter on machine-transcribed lines.

## Use

Score a line with the inputs made exactly as in training, or the scores shift:
- the crop: the line's bounding box, white outside its polygon, 48 pixels high;
- the prompt: diffu's own, in English (`diffu_htr_train.jev.prompt.model_prompt`), not OpenBMB's;
- the rendering: diffu's renderer `@RENDERER@` (`diffu_htr_train.jev.render.render_text`, DejaVu Sans, drawn with the
  Pillow that string names).

So OpenBMB's Jev services, which render and prompt their own way, do not fit this model. Set up vLLM (a nightly from
29 October 2026 or later, with transformers 5.22.1) and the two functions:

```bash
uv venv --python 3.12 jev && source jev/bin/activate
echo "transformers==5.22.1" > overrides.txt
uv pip install --override overrides.txt "pillow==12.2.0" \
  "vllm @ https://wheels.vllm.ai/fa3b9d1377219247060d3c8b49eae199c9e7e154/vllm-0.33.0rc1.dev234%2Bgfa3b9d137-cp38-abi3-manylinux_2_28_x86_64.whl" \
  "git+https://github.com/Borg93/diffu2#subdirectory=packages/htr_train"
```

Then score a line:

```python
import math
from pathlib import Path

from PIL import Image
from vllm import LLM, PoolingParams

from diffu_htr_train.jev.prompt import model_prompt
from diffu_htr_train.jev.render import render_text

if __name__ == "__main__":
    jev = LLM(
        model="@REPO@",
        runner="pooling",
        convert="classify",
        hf_overrides={"num_labels": 2},
        limit_mm_per_prompt={"image": 2},
    )
    crop = Image.open("line.jpg").convert("RGB")
    text = "Anno 1649 den 6 Aprilis"
    rendering = render_text(text, Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), crop.size)
    [out] = jev.classify(
        [{"prompt": model_prompt(text), "multi_modal_data": {"image": [crop, rendering]}}],
        pooling_params=PoolingParams(use_activation=False),
    )
    no, yes = out.outputs.probs
    print(1 / (1 + math.exp(no - yes)))  # the probability that the text matches the line
```

Without vLLM, transformers gives the same scores (within 0.01 on 60 test pairs):

```python
import torch
from transformers import AutoProcessor, Qwen3_5ForSequenceClassification

processor = AutoProcessor.from_pretrained("@REPO@")
jev = Qwen3_5ForSequenceClassification.from_pretrained("@REPO@", num_labels=2, dtype=torch.bfloat16).to("cuda")
inputs = processor(text=[model_prompt(text)], images=[crop, rendering], return_tensors="pt").to("cuda")
with torch.inference_mode():
    print(float(torch.softmax(jev(**inputs).logits[0].float(), -1)[1]))
```

The score is not calibrated: choose a threshold on labelled lines of your own.

## Licence

Apache-2.0, as its base model.
