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
- **Checkpoint:** speculators' epoch folder `@STEP@` (counted from 0), the best on held-out answers.

The files are in the [speculators](https://github.com/vllm-project/speculators) format (`config.json`, its `config.py`
and the weights), and the config names [@READER_REPO@](https://huggingface.co/@READER_REPO@) as the verifier.

## Setup

The draft runs inside vLLM, beside the reader. It fits this reader only: a draft learns its target's own answers.

You need Linux, an NVIDIA GPU with about 20 GB free, and [uv](https://docs.astral.sh/uv/). MiniCPM-OCR needs a vLLM
nightly from 29 October 2026 or later (vllm-project/vllm#64105) with transformers 5.22.1; this is the build the draft
was trained and tested with:

```bash
uv venv --python 3.12 htr && source htr/bin/activate
echo "transformers==5.22.1" > overrides.txt
uv pip install --override overrides.txt openai \
  "vllm @ https://wheels.vllm.ai/fa3b9d1377219247060d3c8b49eae199c9e7e154/vllm-0.33.0rc1.dev234%2Bgfa3b9d137-cp38-abi3-manylinux_2_28_x86_64.whl"
```

On an ARM machine (such as a DGX Spark), use the `..._aarch64.whl` of the same commit. While the repos are private,
log in first with `hf auth login`.

## Use

Serve the reader with the draft:

```bash
vllm serve @READER_REPO@ --served-model-name htr --max-model-len 8192 \
  --limit-mm-per-prompt '{"image": 4, "video": 0}' \
  --mm-processor-kwargs '{"downsample_mode": "4x", "max_slice_nums": 9}' --trust-request-mm-kwargs \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{"method": "dspark", "model": "@REPO@", "num_speculative_tokens": 15, "enable_adaptive_verification": false}'
```

Then ask it exactly as you ask the reader alone: one line or text-region image and the prompt `Text Recognition:`,
decoded greedily.

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="-")
image = base64.b64encode(open("line.jpg", "rb").read()).decode()
reply = client.chat.completions.create(
    model="htr",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
                {"type": "text", "text": "Text Recognition:"},
            ],
        }
    ],
    max_tokens=512,  # 4096 for a text region
    temperature=0.01,
    top_p=0.001,
    extra_body={"top_k": 1, "repetition_penalty": 1.0},
)
print(reply.choices[0].message.content)
```

Leave out `--speculative-config` to serve the reader alone. To check that the draft is used, read vLLM's
`spec_decode_num_*` counters at `http://localhost:8000/metrics`. They rise with every request, and the acceptance
length is 1 + accepted tokens / drafts.

The reader is a hybrid model (linear-attention layers), so with a draft vLLM keeps one copy of each request's recurrent
state per draft token ([speculators: throughput vs. interactivity](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/throughput_interactivity/)):
each request takes more memory, and fewer fit in a batch. Measure with and without the draft at the concurrency you
read at.

## Evaluation

On held-out answers: @VAL@.

Speed against the reader alone, and accuracy with and without the draft: `runs/htr/verify/bench_three.sh`, its tables added to the card after publishing.

## Licence

Apache-2.0, as its base model.
