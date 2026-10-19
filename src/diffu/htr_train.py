"""Fine-tune an HTR recognizer (TrOCR) on a line manifest — the training half of the augmentation A/B.

The Diffu verdict is: train one recognizer on REAL lines, another on REAL + SYNTHETIC, then
``diffu-eval`` both on held-out real ``test.jsonl`` and compare CER. A drop means the synthetic
data helped. This module trains one such recognizer from a manifest; run it twice (real, real+synth).

    diffu-htr-train --train data_out/train.jsonl --val data_out/val.jsonl --out runs/htr_real --epochs 3
    # then:  diffu-eval --manifest data_out/test.jsonl --htr runs/htr_real

Uses HF ``Seq2SeqTrainer``, which runs on **accelerate** — so multi-GPU is a launch flag, no code change:

    accelerate launch --multi_gpu --num_processes 2 -m diffu.htr_train --train … --out … --track

``--track`` logs loss/CER to **trackio** (project ``diffu-htr``, run = out-dir basename), so the
real-vs-real+synth recognizers show up side by side next to the diffu runs: ``uv run trackio show``.
"""

from __future__ import annotations

import argparse
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from .data.dataset import load_manifest
from .eval import _materialize_trocr_positions
from .model.metrics import cer

# 2026 Swedish Lion (char-based): the SOTA historical-Swedish TrOCR — the single default for both the
# API and the CLI, so a bare `diffu-htr-train` run fine-tunes the right recognizer.
_DEFAULT_BASE = "Riksarkivet/trocr-large-handwritten-hist-swe-3-char"


class LineOCRDataset(Dataset):
    """Manifest rows -> TrOCR ``pixel_values`` + tokenized ``labels`` (pad positions masked to -100)."""

    def __init__(self, rows: list[dict[str, str]], processor: Any, max_target_length: int = 96) -> None:
        self.rows = rows
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with Image.open(row["image"]) as im:
            img = im.convert("RGB")
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values[0]  # [3, 384, 384]
        labels = self.processor.tokenizer(
            row["text"], padding="max_length", truncation=True, max_length=self.max_target_length
        ).input_ids
        pad_id = self.processor.tokenizer.pad_token_id
        labels = [t if t != pad_id else -100 for t in labels]  # mask pad for loss
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


def train_htr(
    train_manifest: str,
    out_dir: str,
    *,
    base: str = _DEFAULT_BASE,
    val_manifest: str | None = None,
    epochs: float = 3.0,
    batch_size: int = 8,
    lr: float = 5e-5,
    max_samples: int | None = None,
    device: str | None = None,
    lora: bool = False,
    lora_rank: int = 16,
    seed: int = 42,
    max_steps: int | None = None,
    track: bool = False,
    track_project: str = "diffu-htr",
) -> str:
    """Fine-tune ``base`` TrOCR on ``train_manifest`` and save to ``out_dir`` (returns ``out_dir``).

    ``lora=True`` wraps the recognizer in a PEFT LoRA adapter (decoder attention) instead of full FT — the
    parameter-efficient "pretrain-then-adapt" arm (e.g. synth-pretrain base -> LoRA on the real domain). The
    adapter is merged back into the base before saving, so ``diffu-eval`` loads a plain model.
    """
    import os

    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
        default_data_collator,
    )

    class _TrackioCallback(TrainerCallback):
        """Log Trainer metrics (loss, eval CER, lr) to trackio on the main process only."""

        def __init__(self, project: str, run_name: str, config: dict[str, Any]) -> None:
            self._project, self._run, self._config = project, run_name, config

        def on_train_begin(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
            if state.is_world_process_zero:
                import trackio

                trackio.init(project=self._project, name=self._run, config=self._config)

        def on_log(
            self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kw: Any
        ) -> None:
            if state.is_world_process_zero and logs:
                import trackio

                metrics = {k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}
                if metrics:
                    trackio.log(metrics, step=int(state.global_step))

        def on_train_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
            if state.is_world_process_zero:
                import trackio

                trackio.finish()

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = TrOCRProcessor.from_pretrained(base)
    model = VisionEncoderDecoderModel.from_pretrained(base).to(device)
    _materialize_trocr_positions(model, device)  # transformers-5.x TrOCR meta fix
    tok = processor.tokenizer
    model.config.decoder_start_token_id = (
        tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
    )
    model.config.pad_token_id = tok.pad_token_id
    if lora:  # parameter-efficient adapt (decoder attention) instead of full FT
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=lora_rank,
                lora_alpha=2 * lora_rank,
                lora_dropout=0.05,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "out_proj",
                ],  # TrOCR decoder attention projections
            ),
        )
        model.print_trainable_parameters()

    rows = load_manifest(train_manifest)
    if max_samples is not None:
        rows = rows[:max_samples]
    train_ds = LineOCRDataset(rows, processor)
    val_ds = LineOCRDataset(load_manifest(val_manifest), processor) if val_manifest else None

    def compute_metrics(pred: Any) -> dict[str, float]:
        import numpy as np

        # Seq2SeqTrainer pads generated predictions with -100; the fast tokenizer's Rust decode casts ids
        # to u32 and OverflowErrors on a negative id -> replace -100 (any negative) with the pad id first.
        pred_ids = np.asarray(pred.predictions)
        pred_ids = np.where(pred_ids >= 0, pred_ids, tok.pad_token_id)
        preds = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids = [[t for t in seq if t != -100] for seq in pred.label_ids]
        gts = tok.batch_decode(label_ids, skip_special_tokens=True)
        return {"cer": cer(preds, gts)}

    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps is not None else -1,  # >0 overrides epochs (step-matched arms)
        learning_rate=lr,
        seed=seed,  # HF seeds model init shuffling + dropout; vary to get CER error bars
        bf16=(device == "cuda"),
        predict_with_generate=True,
        eval_strategy="epoch" if val_ds else "no",
        save_strategy="no",
        logging_steps=50,
        remove_unused_columns=False,
        report_to=[],
    )
    callbacks = None
    if track:
        run_name = os.path.basename(out_dir.rstrip("/"))
        cfg = {"base": base, "epochs": epochs, "batch_size": batch_size, "lr": lr, "n_train": len(rows)}
        callbacks = [_TrackioCallback(track_project, run_name, cfg)]
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if val_ds else None,
        callbacks=callbacks,
    )
    trainer.train()
    if lora:  # bake the adapter into the base so diffu-eval loads a plain VisionEncoderDecoderModel
        trainer.model.merge_and_unload().save_pretrained(out_dir)
    else:
        trainer.save_model(out_dir)
    processor.save_pretrained(out_dir)
    print(f"saved HTR recognizer -> {out_dir}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune a TrOCR HTR recognizer on a line manifest.")
    ap.add_argument("--train", required=True, help="train.jsonl (real, or real+synthetic concatenated)")
    ap.add_argument("--val", default=None, help="val.jsonl for CER during training (optional)")
    ap.add_argument("--out", required=True, help="output checkpoint dir (load it with diffu-eval --htr)")
    ap.add_argument("--base", default=_DEFAULT_BASE)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument(
        "--lora", action="store_true", help="LoRA-adapt the recognizer (decoder attention) instead of full FT"
    )
    ap.add_argument("--lora-rank", type=int, default=16, help="LoRA rank when --lora")
    ap.add_argument(
        "--seed", type=int, default=42, help="training seed (vary across runs for CER error bars)"
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="cap optimizer steps (>0 overrides --epochs; step-matches arms)",
    )
    ap.add_argument("--track", action="store_true", help="log loss/CER to trackio (project 'diffu-htr')")
    args = ap.parse_args()

    train_htr(
        args.train,
        args.out,
        base=args.base,
        val_manifest=args.val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
        lora=args.lora,
        lora_rank=args.lora_rank,
        seed=args.seed,
        max_steps=args.max_steps,
        track=args.track,
    )


if __name__ == "__main__":
    main()
