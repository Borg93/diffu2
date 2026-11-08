---
license: apache-2.0
base_model: openbmb/MiniCPM-OCR-1.0
base_model_relation: finetune
pipeline_tag: image-text-to-text
library_name: transformers
language:
  - sv
tags:
  - minicpm-ocr
  - handwriting
  - historical-documents
  - archives
  - htr
  - ocr
---

# @NAME@

A full fine-tune of [MiniCPM-OCR-1.0](https://huggingface.co/openbmb/MiniCPM-OCR-1.0) that reads Swedish handwritten
archive records: one text line, or a whole text region (its lines in reading order, one to a line). It is the reader
of [diffu](https://github.com/Borg93/diffu2), trained to be compared with Riksarkivet's
[Swedish Lion TrOCR](https://huggingface.co/Riksarkivet/swedish-lion-trocr-large-handwritten-char) on the same data and
the same test lines.

Its proof-reader is [@NAME@-jev](https://huggingface.co/@REPO@-jev) and its DSpark draft for faster decoding
[@NAME@-dspark](https://huggingface.co/@REPO@-dspark), where they have been published.

## Training

- **Data:** Riksarkivet's GT-Lejonet HTR ground truth (February 2026), the Swedish Lion's training side (its test split
  `test_0050` held out): @LINES@ text lines, each crop white outside its line's polygon, and @REGIONS@ text regions,
  each crop masked the same way. Lines whose text marks an uncertain reading (`??`, `(?)`, `[`, `]`) and lines under
  10 pixels are left out, as the baseline left them out. The text is trained as transcribed, not normalised.
- **Recipe:** OpenBMB's full fine-tune (vision encoder and language model trained, the aligner frozen), learning rate
  5e-6 on a cosine schedule, 3 epochs, 128 samples a step, bfloat16, on two RTX PRO 6000 Blackwell GPUs.
- **Checkpoint:** @STEP@, the one of least loss on the held-out dev lines.

## Use

With vLLM (OpenBMB's flags for MiniCPM-OCR):

```bash
vllm serve @REPO@ --served-model-name htr --max-model-len 8192 \
  --limit-mm-per-prompt '{"image": 4, "video": 0}' \
  --mm-processor-kwargs '{"downsample_mode": "4x", "max_slice_nums": 9}' --trust-request-mm-kwargs \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

Each line or region crop is sent as the image, followed by the prompt `Text Recognition:`, with OpenBMB's near-greedy
sampling:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="-")
reply = client.chat.completions.create(
    model="htr",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        {"type": "text", "text": "Text Recognition:"},
    ]}],
    max_tokens=512,  # a line; a region takes up to 4096
    temperature=0.01, top_p=0.001, extra_body={"top_k": 1, "repetition_penalty": 1.0},
)
print(reply.choices[0].message.content)
```

Crop and mask lines as in training (white outside the line's polygon) for the accuracy below.

## Evaluation

This checkpoint read through vLLM as above, beside the Swedish Lion TrOCR's greedy readings of the same lines: 5,000
lines of the baseline's own test split (`test_0050`, `test_*` rows) and all 6,358 lines of
[`Riksarkivet/eval_htr_out_of_domain_lines`](https://huggingface.co/datasets/Riksarkivet/eval_htr_out_of_domain_lines)
(`ood_*` rows). Each table is one reader on one set (`trocr` the baseline, `minicpm` this model).

- **mean CER:** the per-line mean, whitespace normalised (the baseline's own scoring).
- **CER, WER (OpenBMB):** corpus rates, all edits over all reference characters or words (NFC, ends trimmed).
- **mean CER, norm.:** the baseline's training normalisation applied to both sides.
- **cut off:** readings that reached the token limit.
- **loops cut:** each reading scored again with a repeating tail cut to one copy.

```
@RESULTS@
```

This model read the 5,000 test lines at @SPEED@ lines a second on one RTX PRO 6000 Blackwell (vLLM, 64 requests at a
time); the baseline read them at @TROCR_SPEED@ (transformers, float32, batches of 32, greedy).

## Limitations

- Trained and tested on GT-Lejonet's collections. Hands, periods and kinds of record it has not seen read worse (the
  out-of-domain rows).
- Like other vision-language readers, it can write a familiar modern spelling for an old one, or drop a letter, where
  a dedicated line recogniser copies the strokes.
- Near-greedy decoding can repeat one unit to the token limit; the `loops cut` columns show what cutting such a tail
  changes.

## Licence

Apache-2.0, as its base model.
