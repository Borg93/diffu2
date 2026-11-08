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
  1e-5, ZeRO-3), 2 epochs, 128 pairs a step, on two RTX PRO 6000 Blackwell GPUs.
- **Checkpoint:** @STEP@, the last.

## Evaluation

On its held-out dev pairs at the last evaluation: @EVAL@.

This has not yet been measured on JevOCRBench, nor as a label filter on machine-transcribed lines.

## Use

As OpenBMB's Jev (see its [vLLM deployment](https://github.com/OpenBMB/MiniCPM-OCR)): a 2-class sequence classifier
served with `--runner pooling --convert classify`. The input is the crop, the candidate and the candidate's rendering,
in OpenBMB's prompt.

## Licence

Apache-2.0, as its base model.
