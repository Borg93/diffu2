"""``htr-nemotron-parse``: Nemotron 3.5 Parse fine-tuned on whole pages, by transformers' Trainer around the model's
remote code (run in ``envs/nemotron_parse``, one process a GPU under ``torchrun``, as ``htr-train`` renders it).

Its own options come as one JSON (``--options``, :class:`diffu_htr_train.nemotron_parse.options.TrainerOptions`), every
other flag is transformers' ``TrainingArguments``' own, parsed by its ``HfArgumentParser``. Three are always set, as
this model needs them: ``--remove_unused_columns false`` (the batches are the collator's), ``--label_names labels``
(its forward names no labels, so the Trainer would compute no evaluation loss) and ``--prediction_loss_only true``
(an evaluation keeps no logits: 72,512 a token). ``--gradient_checkpointing`` is refused (the model does not support
it; ``activation_checkpointing`` does the same), and so is more than one GPU in one process (DataParallel would split
the packed patches): run it under ``torchrun``. A run resumes from ``--resume_from_checkpoint`` (the Trainer's own
state: optimizer, scheduler, RNG, and the batches already trained skipped).

    torchrun --standalone --nproc-per-node 2 htr-nemotron-parse --options '{"model": ...}' --output_dir out ...
"""

from __future__ import annotations

from typing import Annotated

import typer
from pydantic import ValidationError
from transformers import AutoProcessor, HfArgumentParser, Trainer, TrainingArguments

from diffu_htr_train.nemotron_parse import TASK_PROMPT
from diffu_htr_train.nemotron_parse.data import PageCollator, page_dataset
from diffu_htr_train.nemotron_parse.loss import LayoutWeightedLoss, layout_mask
from diffu_htr_train.nemotron_parse.model import (
    CopyModelFiles,
    checkpoint_activations,
    export,
    freeze,
    load_model,
    model_files,
)
from diffu_htr_train.nemotron_parse.options import TrainerOptions
from diffu_htr_train.output import validation_problems

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

# TrainingArguments this model always needs (the module's note), given after the run's own so they win
FORCED = ["--remove_unused_columns", "false", "--label_names", "labels", "--prediction_loss_only", "true"]


def training_arguments(flags: list[str]) -> TrainingArguments:
    """The Trainer's arguments from ``flags`` and :data:`FORCED`.

    Raises:
        typer.BadParameter: If a flag is not one of TrainingArguments', or it asks for gradient checkpointing.
    """
    try:
        (args,) = HfArgumentParser(TrainingArguments).parse_args_into_dataclasses(args=[*flags, *FORCED])
    except SystemExit as error:
        raise typer.BadParameter(f"not TrainingArguments' flags: {' '.join(flags)}") from error
    if args.gradient_checkpointing:
        raise typer.BadParameter(
            "Nemotron 3.5 Parse does not support transformers' gradient checkpointing: set activation_checkpointing "
            "in --options instead",
            param_hint="--gradient_checkpointing",
        )
    return args


def _say(args: TrainingArguments, text: str) -> None:
    """``text`` printed by the first process only."""
    if args.process_index == 0:
        typer.echo(f"htr-nemotron-parse: {text}")


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def main(
    ctx: typer.Context,
    options: Annotated[str, typer.Option(help="the trainer's own options, as JSON (TrainerOptions)")],
) -> None:
    """Fine-tune Nemotron 3.5 Parse on page sets; every flag but --options is TrainingArguments'."""
    try:
        settings = TrainerOptions.model_validate_json(options)
    except ValidationError as error:
        raise typer.BadParameter(validation_problems(error, "options"), param_hint="--options") from error
    args = training_arguments(ctx.args)
    if args.n_gpu > 1:
        raise typer.BadParameter(f"{args.n_gpu} GPUs in one process: run under torchrun, a process a GPU")
    model = load_model(settings)
    architectures = list(model.config.architectures)  # the repo's, before saving renames them
    trainable = freeze(model, vision_encoder=settings.freeze_vision_encoder, adapter=settings.freeze_adapter)
    _say(args, f"trainable parameters by part: {trainable}")
    if settings.activation_checkpointing:
        _say(args, f"{checkpoint_activations(model)} blocks and layers recompute their activations")
    files = model_files(settings)
    processor = AutoProcessor.from_pretrained(files, trust_remote_code=True)
    tokenizer = processor.tokenizer
    train_set, read = page_dataset(settings.train, tokenizer, max_tokens=settings.max_target_tokens)
    _say(args, f"training pages: {read.kept} ({read.too_long} left out: answers past {settings.max_target_tokens})")
    dev_set = None
    if settings.dev is not None:
        dev_set, read = page_dataset([settings.dev], tokenizer, max_tokens=settings.max_target_tokens)
        _say(args, f"dev pages: {read.kept} ({read.too_long} left out)")
    prompt = tokenizer(TASK_PROMPT, add_special_tokens=False)["input_ids"]
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=PageCollator(processor.image_processor, prompt, tokenizer.pad_token_id),
        train_dataset=train_set,
        eval_dataset=dev_set,
        compute_loss_func=LayoutWeightedLoss(
            layout_mask(tokenizer, model.config.vocab_size), settings.coordinate_weight
        ),
        callbacks=[CopyModelFiles(files, architectures)],
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    _say(args, f"the model trained, in bfloat16: {export(trainer, files, architectures)}")


if __name__ == "__main__":
    app()
