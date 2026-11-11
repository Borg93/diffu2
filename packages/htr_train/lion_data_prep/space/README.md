---
title: Swedish HTR speed and Jev
emoji: 🪶
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
python_version: "3.12"
short_description: TrOCR vs MiniCPM-OCR with a DSpark draft, and Jev
license: apache-2.0
startup_duration_timeout: 1h
models:
  - Gabriel/diffu-minicpm-ocr-htr
  - Gabriel/diffu-minicpm-ocr-htr-dspark
  - Gabriel/diffu-minicpm-ocr-htr-jev
  - Gabriel/swedish-lion-trocr-large-handwritten-char
---

# Swedish HTR speed and Jev

Reads the same handwritten text region with one stream each on the same GPU, times each and replays them from the same
start: the Swedish Lion TrOCR on the region's text lines, one after the other (and once more as one batch), the
[MiniCPM-OCR HTR reader](https://huggingface.co/Gabriel/diffu-minicpm-ocr-htr) on the region as one image, and the same
reader with its [DSpark draft](https://huggingface.co/Gabriel/diffu-minicpm-ocr-htr-dspark). Under the race: the same
region's timings on our server, where vLLM serves the reader, and what the readers do there when everything is batched.
Then [Jev](https://huggingface.co/Gabriel/diffu-minicpm-ocr-htr-jev), the reader's proof-reader, scores both readers'
reading of each line.

- `app.py`: the Gradio app; the readers run one after the other and the page replays them at their measured pace.
- `dspark_hf.py`: DSpark speculative decoding in PyTorch on CUDA graphs (ZeroGPU cannot host vLLM).
- `race_view.py`, `race_view.js`, `race_view.css`: the custom component that draws the race, our server's timings and
  Jev's rows.
- `reference.json`, `reference.py`, `measure_reference.py`: each example region's timings on our server (TrOCR in
  transformers, the reader in vLLM), and the script that measured them.
- `check_decode.py`: compares the reader alone and with its draft on a GPU, token for token, and times them.

See the app's "How it works" tab. Part of [diffu](https://github.com/Borg93/diffu2).
