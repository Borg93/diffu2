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
  5e-6 on a cosine schedule planned over 3 epochs, 128 samples a step, bfloat16, on two RTX PRO 6000 Blackwell GPUs.
  Stopped after 28,000 of its 36,681 steps (2.3 epochs), where the held-out lines' error had all but stopped falling.
- **Checkpoint:** @STEP@, the one of lowest corpus CER on 1,000 held-out dev lines.

## Use

**vLLM version.** MiniCPM-OCR needs the fix of [vllm-project/vllm#64105](https://github.com/vllm-project/vllm/pull/64105)
(merged 2026-10-29, not in v0.33.0rc3): vLLM at commit `fa3b9d13`, the commit OpenBMB's installer builds
(`bash scripts/install.sh vllm` in [OpenBMB/MiniCPM-OCR](https://github.com/OpenBMB/MiniCPM-OCR), see its
`docs/minicpm_ocr_tutorial_en.md`). This model was scored with vLLM's own wheel of that commit
(`vllm-0.33.0rc1.dev234+gfa3b9d137`) and transformers 5.22.1.

**The crop, as in training.** For a line: its bounding box cut from the scan, white outside the line's polygon, scaled
down (aspect kept) to at most 200,704 pixels, one 448 x 448 view. For a region: the same, at most 1,806,336 pixels (9
views); the reading comes back one line of text a line. The scores below are for crops made this way.

Serve it with OpenBMB's flags for MiniCPM-OCR (`--trust-request-mm-kwargs` and `enable_thinking: false` are needed:
without them requests fail or come back empty), the settings these scores were measured with:

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

**Loops.** On a few lines of dotted leaders (`1: Ducats . . . . .`) the reading repeats the dots up to the token limit
(3 of the 5,000 test lines below). `finish_reason == "length"` marks such a reading. Cut its repeated tail as these
scores' `loops cut` columns do: `diffu_htr_train.evaluate.cer.cut_loop` in [diffu](https://github.com/Borg93/diffu2)
(`packages/htr_train`).

**Only text was trained.** The fine-tune taught `Text Recognition:` on handwritten lines and regions. OpenBMB's other
prompts (tables, formulas, seals, charts, field extraction) are untested here and may have lost ground.

**Whole pages.** OpenBMB's SDK (`pip install minicpm-ocr`, the command `minicpmvocr parse --server-url
http://localhost:8000 --model htr`) finds the text regions of a page with PP-DocLayoutV3 (blocks such as a paragraph or
a marginal note, several lines each, never single lines) and sends each region's crop to this server. The model was
trained on regions too, but the SDK's crops are rectangles, not masked to their outlines as in training, and the SDK
route has not been scored here. During training, 200 held-out regions, masked as in training, read at 4.29 % corpus
CER (the checkpoint's own evaluation); the scores below are for lines.

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
