---
license: apache-2.0
base_model: @READER_REPO@
base_model_relation: finetune
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - minicpm-ocr
  - dspark
  - speculative-decoding
  - draft-model
  - handwriting
  - htr
---

# @NAME@

A DSpark draft (speculative decoding) head for [@READER_NAME@](https://huggingface.co/@READER_REPO@), the Swedish
handwriting reader of [diffu](https://github.com/Borg93/diffu2). It proposes up to 15 tokens at a time for the reader to
verify, so the reader decodes faster and, decoding greedily, writes the same text. It is trained the way OpenBMB trained
[MiniCPM-OCR-1.0-DSpark](https://huggingface.co/openbmb/MiniCPM-OCR-1.0-DSpark), but on this reader's own answers.

## Training

- **Data:** the reader's own answers to 250,000 training lines and 50,000 training regions of GT-Lejonet (February
  2026), each a seeded draw.
- **Recipe:** 3 online epochs, with vLLM serving the reader on one RTX PRO 6000 Blackwell and the trainer on another.
- **Checkpoint:** epoch @STEP@, the best on held-out answers.

The files are in the [speculators](https://github.com/vllm-project/speculators) format (`config.json`, its `config.py`
and the weights), and the config names [@READER_REPO@](https://huggingface.co/@READER_REPO@) as the verifier.

## Use

Serve the reader with this draft beside it. It fits this reader only: a draft learns its target's own answers.

```bash
vllm serve @READER_REPO@ --served-model-name htr --max-model-len 8192 \
  --limit-mm-per-prompt '{"image": 4, "video": 0}' \
  --mm-processor-kwargs '{"downsample_mode": "4x", "max_slice_nums": 9}' --trust-request-mm-kwargs \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{"method": "dspark", "model": "@REPO@", "num_speculative_tokens": 15, "enable_adaptive_verification": false}'
```

Requests are the reader's, and greedy decoding gives the same text. vLLM's `spec_decode_num_*` metrics rise when the
draft is used.

## Evaluation

On held-out answers: @VAL@.

How much faster the reader decodes with it, on lines and on regions, has not yet been measured.

## Licence

Apache-2.0, as its base model.
