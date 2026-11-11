"""Nemotron 3.5 Parse fine-tuned on whole pages (``diffu-reader-set --model nemotron-parse``) by htr-train's own trainer
(:mod:`diffu_htr_train.nemotron_parse.train`, run in ``envs/nemotron_parse``): transformers' Trainer around the model's
remote code, whose forward takes no labels and returns no loss, with the loss of NVIDIA's own fine-tuning recipe
(:mod:`diffu_htr_train.nemotron_parse.loss`). Neither ms-swift nor NeMo AutoModel has this model: AutoModel's
``nemotron_parse`` is the encoder-decoder of Nemotron-Parse 1.1 to 2.0 (RADIO-H, cross-attention, mBART positions), and
3.5 (``RadioMBartPrefixForCausalLM``) is a decoder reading the page as a causal prefix, with grouped-query attention, no
positions and packed NaViT patches.

This module holds what both sides of the run agree on, torch-free: the model and the one revision whose code runs, and
the task prompt every sample is asked with (diffu-page writes it after ``<image>``; the trainer checks it).
"""

REPO = "nvidia/Nemotron-3.5-Parse"
REVISION = "72ad4d71dd827135ee581502363e14748008900a"  # 2026-11-06; its code, processor and tokenizer read at it
TASK_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"
USER_TURN = f"<image>{TASK_PROMPT}"  # a sample's user turn: the page, then the task (the page is the prefix it reads)
