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

Reads the same handwritten text region three ways on the same GPU, times each and replays the three side by side: the
Swedish Lion TrOCR on the region's text lines, the
[MiniCPM-OCR HTR reader](https://huggingface.co/Gabriel/diffu-minicpm-ocr-htr) on the region as one image, and the same
reader with its [DSpark draft](https://huggingface.co/Gabriel/diffu-minicpm-ocr-htr-dspark). Then
[Jev](https://huggingface.co/Gabriel/diffu-minicpm-ocr-htr-jev), the reader's proof-reader, scores both readers' reading
of each line.

- `app.py`: the Gradio app; the readers run one after the other and the page replays them at their measured pace.
- `dspark_hf.py`: DSpark speculative decoding in PyTorch on CUDA graphs (ZeroGPU cannot host vLLM).
- `race_view.py`, `race_view.js`, `race_view.css`: the custom component that draws the race and Jev's rows.
- `check_decode.py`: compares the reader alone and with its draft on a GPU, token for token, and times them.

See the app's "How it works" tab. Part of [diffu](https://github.com/Borg93/diffu2).
