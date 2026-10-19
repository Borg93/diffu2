"""Training loop + entrypoint for Diffu (diffusers-first, Stage-1 flow matching).

Modern optimizations baked in: bf16 mixed precision (accelerate), EMA weights (diffusers EMAModel),
optional gradient checkpointing (diffusers), optional regional torch.compile (bucket line widths to bound
recompiles), attention through diffusers' kernel backends. Single- and multi-GPU via `accelerate launch`
(see the Makefile `train-*` targets).

Every checkpoint dir holds ``config.json`` — the model config every loader reads back — and the weights:
``step_N/`` the full resumable accelerate state plus ``ema_model.safetensors`` (the EMA weights, what
inference uses); ``final/`` the EMA-applied ``model.safetensors`` export.

    make train-1gpu DATA_DIR=data_out
    accelerate launch --mixed_precision bf16 -m diffu.train --data-dir data_out --steps 100000 --batch-size 8
"""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Generator
from contextlib import contextmanager
from functools import partial
from time import perf_counter
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from accelerate import Accelerator, PartialState
from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs, set_seed
from diffusers.training_utils import EMAModel
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from .config import Config, add_arch_args, apply_arch_args
from .data.collate import collate_lines
from .data.dataset import HandwritingLineDataset, load_manifest
from .data.sampler import WidthBucketBatchSampler, line_widths
from .generate import load_checkpoint, to_pil
from .model import Diffu
from .model.metrics import cer
from .recognizer import load_recognizer, read_lines

if TYPE_CHECKING:
    from PIL import ImageFont
    from PIL.Image import Image as PILImage

_VAL_BATCHES = 8  # held-out batches averaged per val_loss (damps the per-batch timestep noise)
_HF_VAL_ROWS = 64  # rows of an --hf-dataset val split kept for the val loss / sample sheet
# classic-CFG scale of the guided gen_CER (cfg=5 measured best vs 3; docs/PERF_AUDIT.md §0c)
_GEN_CER_CFG_SCALE = 5.0
# The sample-sheet geometry: the REAL/OURS tag gutter and one caption line's height
# (scripts/make_training_animation.py crops the sheets by these).
SAMPLE_LABEL_W, SAMPLE_CAPTION_H = 88, 22
# Frozen pretrained submodules: rebuilt from their Hub weights by Diffu(cfg), so the exported weights skip them.
_PRETRAINED_PREFIXES = ("vae.", "style.enc.")


def trainable_parameters(model: Diffu) -> list[torch.nn.Parameter]:
    """Parameters trained in Stage 1 (flow matching).

    Trained: SD3 backbone, DINOv3 style resampler + pool (``style_mode="dino"`` only), the glyph-line content
    encoder, the fill-ratio MLP, the REPA head. Frozen: the whole VAE (the Stage-1 forward never decodes) and
    the DINOv3 encoder.
    """
    params = list(model.backbone.parameters())
    if model.style is not None and model.style.resampler is not None and model.style.pool is not None:
        params += list(model.style.resampler.parameters())
        params += list(model.style.pool.parameters())  # learned-query style attention-pool
    params += list(model.glyph_content.parameters())
    if model.fill_ratio_mlp is not None:  # fill-ratio AdaLN conditioning MLP (EMA shadows it too)
        params += list(model.fill_ratio_mlp.parameters())
    if model.repa_proj is not None:  # REPA alignment head (when cfg.aux.repa)
        params += list(model.repa_proj.parameters())
    return [p for p in params if p.requires_grad]


_OPTIONAL_BATCH_KEYS = ("style_pixel_values", "ref_images", "ref_texts", "ref_widths")


def model_inputs(batch: dict[str, Any], device: torch.device | None = None) -> dict[str, Any]:
    """The optional model kwargs a batch carries: ``text_widths`` (glyph_pad), the DINO style crop (dino
    mode) or the canvas reference (canvas mode).

    Moves tensors to ``device`` when given (the val/sample batches are not accelerate-prepared).
    """
    out: dict[str, Any] = {"text_widths": batch.get("widths")}
    for key in _OPTIONAL_BATCH_KEYS:
        value = batch.get(key)
        if value is not None:
            out[key] = value.to(device) if device is not None and isinstance(value, torch.Tensor) else value
    return out


@torch.no_grad()
def evaluate_val_loss(model: Diffu, val_loader: DataLoader, device: torch.device) -> float:
    """Mean flow loss on held-out (writer-disjoint) lines.

    The overfitting signal: this rising while train loss keeps falling = the model is memorizing the
    training writers instead of learning generalizable handwriting. Averaged over several batches to damp
    the per-batch timestep noise so the trend is readable.
    """
    model.eval()
    total, seen = 0.0, 0
    try:
        for i, batch in enumerate(val_loader):
            if i >= _VAL_BATCHES:
                break
            loss = model(batch["images"].to(device), batch["texts"], **model_inputs(batch, device))
            total += loss.item()
            seen += 1
    finally:
        model.train()  # always restore train mode, even if a batch raises (shared module, rank-0 toggle)
    return total / max(seen, 1)


def _label_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """A label font that can render Swedish glyphs (å ä ö): DejaVu if present, else PIL's scalable default
    (a missing font must never crash a training run)."""
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _compose_sample_grid(
    reals: list[PILImage],
    gens: list[PILImage],
    texts: list[str],
    width: int,
    line_h: int,
    reads: list[str] | None = None,
    style_texts: list[str] | None = None,
) -> PILImage:
    """Stack each (real, generated) pair into one labeled strip: a ``want: <text>`` caption, the REAL
    target row (green tag) and OURS directly below (red tag). The burned-in tags make the trackio media
    tab self-explanatory — which row is real vs ours is otherwise pure guesswork.

    When ``reads`` (the recognizer's transcription of each OURS row) is given, a red
    ``HTR reads: <pred>`` caption is drawn directly UNDER the OURS row, so the PNG alone shows what
    the recognizer thinks our line says (the legibility gap) at a glance."""
    from PIL import Image, ImageDraw

    lbl_w, cap_h = SAMPLE_LABEL_W, SAMPLE_CAPTION_H
    read_h = cap_h if reads is not None else 0
    style_h = cap_h if style_texts is not None else 0
    pair_h = style_h + cap_h + 2 * line_h + read_h
    canvas = Image.new("RGB", (lbl_w + width, len(reals) * pair_h), "white")
    draw = ImageDraw.Draw(canvas)
    cap_font, tag_font = _label_font(15), _label_font(16)
    for i, (real_img, gen_img, text) in enumerate(zip(reals, gens, texts, strict=False)):
        y0 = i * pair_h
        draw.line([(0, y0), (lbl_w + width, y0)], fill="#cccccc")
        if style_texts is not None and i < len(style_texts):  # the text the STYLE ref shows (purple) —
            draw.text(  # we ask for a DIFFERENT text, so a copy-the-style model renders THIS, not `want`
                (lbl_w + 4, y0 + 4), f"style ref wrote: {style_texts[i]!r}", fill="#8250df", font=cap_font
            )
        top = y0 + style_h
        draw.text((lbl_w + 4, top + 4), f"want: {text!r}", fill="black", font=cap_font)
        canvas.paste(real_img.resize((width, line_h)), (lbl_w, top + cap_h))
        canvas.paste(gen_img.resize((width, line_h)), (lbl_w, top + cap_h + line_h))
        draw.text((6, top + cap_h + line_h // 2 - 8), "REAL", fill="#1a7f37", font=tag_font)
        draw.text((6, top + cap_h + line_h + line_h // 2 - 8), "OURS", fill="#cf222e", font=tag_font)
        if reads is not None and i < len(reads):  # what the HTR recognizer reads our OURS line as
            draw.text(
                (lbl_w + 4, top + cap_h + 2 * line_h + 3),
                f"HTR reads: {reads[i]!r}",
                fill="#cf222e",
                font=cap_font,
            )
    return canvas


class Samples(NamedTuple):
    """Generated held-out lines: the requested texts, what each style reference wrote, the real targets, the
    unguided generations, the CFG-guided ones (when measured) and the canvas width."""

    texts: list[str]
    style_texts: list[str]
    reals: list[PILImage]
    gens: list[PILImage]
    gens_cfg: list[PILImage] | None
    width: int


@torch.no_grad()
def generate_samples(
    model: Diffu, batch: dict[str, Any], cfg: Config, device: torch.device, n: int = 4, cfg_scale: float = 0.0
) -> Samples:
    """Generate ``n`` held-out lines of a val batch.

    ``gens`` are the raw (unguided) generations — what the sample sheet shows: high guidance over-saturates
    an undertrained model to pure white, hiding the real (on-manifold) generation. With ``cfg_scale > 0`` a
    second, CFG-guided generation of the same texts/styles is returned as ``gens_cfg`` (else ``None``) so the
    caller can report gen_CER with AND without guidance (does CFG actually help legibility here?).
    """
    n = min(n, len(batch["texts"]))
    texts = list(batch["texts"][:n])
    style_texts = list(batch.get("style_texts", [""] * len(batch["texts"]))[:n])
    real = batch["images"][:n]
    w = real.shape[-1]
    latent_hw = (cfg.data.line_height // cfg.vae.downscale_factor, w // cfg.vae.downscale_factor)
    # glyph_pad widths + the style input, sliced to the n samples
    gen_kwargs = {k: v[:n] for k, v in model_inputs(batch, device).items() if v is not None}
    style = gen_kwargs.pop("style_pixel_values", None)  # dino mode; canvas models read the ref_* keys
    if "ref_texts" in gen_kwargs:  # caption the sheet with what the in-canvas reference actually says
        style_texts = list(gen_kwargs["ref_texts"])

    # Sampling runs in eval mode; the try/finally guarantees train mode is restored even if a sampling
    # step raises. Otherwise the shared module is left in eval() and the next training steps silently
    # skip the CFG cond/text-dropout (it gates on `self.training`) — quietly breaking CFG training.
    model.eval()
    try:
        gen = model.generate(texts, style, latent_hw=latent_hw, **gen_kwargs)
        gens_cfg = None
        if cfg_scale > 0.0:
            gen_cfg = model.generate(texts, style, latent_hw=latent_hw, cfg_scale=cfg_scale, **gen_kwargs)
            gens_cfg = [to_pil(img) for img in gen_cfg]
    finally:
        model.train()
    return Samples(
        texts, style_texts, [to_pil(img) for img in real], [to_pil(img) for img in gen], gens_cfg, w
    )


def _setup_model(cfg: Config, *, grad_checkpoint: bool, attention_backend: str | None = None) -> Diffu:
    """Build the model and freeze everything Stage-1 flow matching never touches.

    The VAE (encode is no_grad, decode unused) is frozen so DDP's reducer doesn't wait on gradients
    that never arrive, which crashes on step 1 under ``--multi_gpu``.
    """
    model = Diffu(cfg)
    if grad_checkpoint:
        model.backbone.enable_gradient_checkpointing()  # off = ~20% faster on big VRAM
    if attention_backend:
        model.backbone.set_attention_backend(attention_backend)  # before compile: it is traced into the graph
    model.vae.requires_grad_(False)
    return model


def _init_tracker(
    track: bool, is_main: bool, config: dict[str, Any], track_space: str | None, run_name: str | None = None
) -> Any:
    """Initialise trackio on the main process (live dashboard + alerts); return None otherwise.

    ``run_name`` (the out-dir basename) is passed so parallel runs get distinct dashboard names —
    trackio auto-names collide when several runs launch at once with the same seed. A ``track_space`` the
    run creates is PRIVATE: the sample sheets show training lines, and an auto-created Space is public.
    """
    if not (track and is_main):
        return None
    import trackio

    trackio.init(project="diffu", name=run_name, config=config, space_id=track_space, private=True)
    return trackio


def _run_profile(
    accel: Accelerator,
    model: torch.nn.Module,
    dataloader: DataLoader,
    params: list[torch.nn.Parameter],
    out_dir: str,
    profile_steps: int,
    dropouts: dict[str, float],
    with_stack: bool,
) -> Diffu:
    """Profile ``profile_steps`` real forward+backward steps and write the kernel report.

    Mirrors the Stage-1 training step (model forward under accelerate's bf16 autocast with the run's
    dropouts, backward, grad zeroing) but skips the optimizer/scheduler/EMA so the profile reflects compute
    only. The report goes to ``out_dir/profile_step_table.txt`` and the trace to ``trace.json`` (main
    process). Called from :func:`train` only when ``profile_steps > 0``; returns the unwrapped model after.
    """
    from .profiling import profile_callable, read_dcgm

    batches = iter(dataloader)

    def step() -> None:
        nonlocal batches
        try:
            batch = next(batches)
        except StopIteration:  # small datasets: re-iterate so we always have profile_steps batches
            batches = iter(dataloader)
            batch = next(batches)
        loss = model(batch["images"], batch["texts"], **dropouts, **model_inputs(batch))
        accel.backward(loss)
        for p in params:
            p.grad = None

    model.train()
    trace = os.path.join(
        out_dir, "trace.json" if accel.is_main_process else f"trace_rank{accel.process_index}.json"
    )
    report = profile_callable(step, warmup=2, active=profile_steps, trace_path=trace, with_stack=with_stack)
    if accel.is_main_process:
        path = os.path.join(out_dir, "profile_step_table.txt")
        with open(path, "w") as fh:
            fh.write(report.render())
            dcgm = read_dcgm()
            if dcgm is not None:
                fh.write(f"\n\n==== DCGM (GPU 0): {dcgm} ====\n")
        print(f"profile written to {path}", flush=True)
        print(report.render(), flush=True)
    accel.wait_for_everyone()
    return accel.unwrap_model(model)


def _warmup_cosine_min_lr(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay to ``min_lr`` (a floor, not 0).

    diffusers' ``get_cosine_schedule_with_warmup`` decays to exactly 0; this decays to a small ``min_lr``
    instead, so late training keeps making tiny but non-zero updates.
    """
    floor = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return floor + (1.0 - floor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _resume_step(ckpt_dir: str) -> int:
    """Recover the optimizer-step count from an ``accelerate`` checkpoint dir named ``step_<N>`` (else 0).

    The step counter isn't stored inside ``save_state``; it lives in the directory name (``step_160000``),
    so the resumed run continues logging/save/val/sample cadence and the REPA early-stop from the right step.
    """
    base = os.path.basename(ckpt_dir.rstrip("/"))
    tail = base.removeprefix("step_")
    return int(tail) if tail.isdigit() else 0


def _latest_checkpoint(out_dir: str) -> str | None:
    """The newest complete ``step_N`` checkpoint under ``out_dir``, or None (a fresh run). Complete = it has
    the ``config.json`` written after the state, so a job killed mid-save resumes from the one before."""
    if not os.path.isdir(out_dir):
        return None
    done = [
        (int(name.removeprefix("step_")), name)
        for name in os.listdir(out_dir)
        if name.removeprefix("step_").isdigit() and os.path.isfile(os.path.join(out_dir, name, "config.json"))
    ]
    return os.path.join(out_dir, max(done)[1]) if done else None


@contextmanager
def _ema_weights(ema: EMAModel | None, params: list[torch.nn.Parameter]) -> Generator[None]:
    """Swap the EMA weights into ``params`` for the block, then restore the raw training weights."""
    if ema is None:
        yield
        return
    ema.store(params)
    ema.copy_to(params)
    try:
        yield
    finally:
        ema.restore(params)


def _export_state(model: Diffu) -> dict[str, torch.Tensor]:
    """The model's weights for inference: everything but the frozen pretrained encoders (CPU copies)."""
    return {
        k: v.detach().to("cpu", copy=True).contiguous()
        for k, v in model.state_dict().items()
        if not k.startswith(_PRETRAINED_PREFIXES)
    }


def _save_checkpoint(
    accel: Accelerator,
    unwrapped: Diffu,
    out_dir: str,
    *,
    cfg: Config,
    ema: EMAModel | None,
    params: list[torch.nn.Parameter],
    lora: bool,
    final: bool,
) -> None:
    """Write a checkpoint dir: ``config.json`` + the weights (see the module docstring).

    ``final=False`` (``step_N``): the full resumable accelerate state (collective, raw weights) + the
    ``ema_model.safetensors`` export. ``final=True``: just the EMA-applied ``model.safetensors`` export. A
    LoRA run writes the few-MB adapter instead (EMA-applied in ``final``).
    """
    if not lora and not final:
        accel.save_state(out_dir)
    if accel.is_main_process:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
            fh.write(cfg.model_dump_json(indent=2))
        if lora:
            from .lora import save_lora

            with _ema_weights(ema if final else None, params):
                save_lora(unwrapped, out_dir)
        elif final or ema is not None:
            with _ema_weights(ema, params):
                name = "model.safetensors" if final else "ema_model.safetensors"
                save_file(_export_state(unwrapped), os.path.join(out_dir, name))
    accel.wait_for_everyone()


def train(
    cfg: Config,
    dataloader: DataLoader,
    *,
    steps: int = 100_000,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    log_every: int = 50,
    save_every: int = 10_000,
    out_dir: str = "checkpoints",
    use_ema: bool = True,
    compile_model: bool = False,
    grad_accum: int = 1,
    save_final: bool = True,
    track: bool = False,
    track_space: str | None = None,
    grad_checkpoint: bool = True,
    max_grad_norm: float = 1.0,
    warmup_steps: int = 2000,
    min_lr: float = 1e-6,
    cond_dropout: float = 0.0,
    text_dropout: float = 0.0,
    ref_dropout: float = 0.0,
    init_from: str | None = None,
    resume_from: str | None = None,
    val_loader: DataLoader | None = None,
    val_every: int = 500,
    sample_every: int = 2000,
    log_cer: bool = True,
    profile_steps: int = 0,
    profile_stack: bool = False,
    lora: bool = False,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    run_config: dict[str, Any] | None = None,
    attention_backend: str | None = None,
    compile_mode: str | None = None,
    compile_vae: bool = False,
) -> Diffu:
    """Run the Stage-1 flow-matching training loop.

    Args:
        cfg: Full model/data config (saved as ``config.json`` in every checkpoint).
        dataloader: Yields ``collate_lines`` batches (``images`` / ``texts`` + the style input).
        steps: Total optimizer steps.
        lr / weight_decay / max_grad_norm / warmup_steps / min_lr: AdamW + grad clipping + the warmup ->
            cosine-to-``min_lr`` schedule.
        log_every: Print (and track) the losses every N steps; also when a non-finite loss is caught.
        save_every: Write a ``step_N`` checkpoint every N steps.
        out_dir: Directory for checkpoints and sample sheets.
        use_ema: Maintain EMA weights of the trained parameters (sampling, exports).
        compile_model: Regionally ``torch.compile`` the MMDiT blocks (fastest with ``grad_checkpoint=False``).
        grad_accum: Gradient accumulation steps (``step`` counts optimizer steps).
        save_final: Write the EMA-applied ``final`` export when done.
        track / track_space: Log to trackio (+ alerts), optionally synced to a private HF Space.
        grad_checkpoint: diffusers gradient checkpointing (less memory, ~20% slower).
        cond_dropout / text_dropout / ref_dropout: the guidance dropout modes (see ``Diffu.forward``).
        init_from: Warm-start weights from a checkpoint (optimizer/LR/step restart).
        resume_from: FULL resume from an accelerate ``step_N`` dir (model+optimizer+LR+EMA+RNG+step).
        val_loader / val_every / sample_every / log_cer: held-out val loss, sample sheets and the live
            gen_CER legibility gauge (rank 0).
        profile_steps / profile_stack: If > 0, profile that many real steps (optionally with Python stack
            attribution), write the report to ``out_dir``, and return BEFORE the training loop.
        lora / lora_rank / lora_alpha / lora_dropout: Stage-3 adapter training on a frozen base.
        run_config: CLI-only extras logged with the run (batch size, dataset, …).
        attention_backend / compile_mode / compile_vae: kernel backend, compile mode, VAE-encode compile.

    Returns:
        The (unwrapped) trained model.
    """
    # find_unused_parameters: under multi-GPU DDP, a parameter that receives NO gradient on a step is a
    # fatal "didn't finish reduction" error. The REPA head (repa_proj) goes unused whenever REPA is
    # early-stopped (repa_stop_frac < 1) — tolerate that so the run never dies at the stop step.
    accel = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=grad_accum,
        # Async H2D: with pin_memory=True (build_dataloader) the prepared loader overlaps the next batch's
        # host->device copy with the current step's compute. No-op without pinned memory.
        dataloader_config=DataLoaderConfiguration(non_blocking=True),
        # gradient_as_bucket_view: alias grads onto DDP's reduction buckets (one fewer grad copy + lower
        # peak memory). broadcast_buffers=False: skip the per-step buffer broadcast — safe here (the only
        # live buffers are the content ResNet's BN stats, which in train() normalize on the local minibatch,
        # plus the VAE's constant lat_mean/std and per-call RoPE freqs; init is identical across ranks via
        # set_seed). Both are no-ops on a single GPU.
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
                broadcast_buffers=False,
            )
        ],
    )
    torch.set_float32_matmul_precision("high")  # TF32 on the fp32 GEMMs outside bf16 autocast
    set_seed(cfg.seed)  # reproducible init + data order across ranks
    model = _setup_model(cfg, grad_checkpoint=grad_checkpoint, attention_backend=attention_backend)
    if init_from is not None:  # warm-start from prior weights (strict=False) instead of random init
        load_checkpoint(model, init_from)
        if accel.is_main_process:
            print(f"warm-started weights from {init_from}", flush=True)
    if lora:  # Stage-3: freeze the base, train only a low-rank adapter on the backbone (diffu.lora)
        from .lora import add_lora

        n_lora = add_lora(model, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
        if accel.is_main_process:
            base = "random init — pass --init-from for a real adapter" if init_from is None else init_from
            print(
                f"LoRA rank={lora_rank}: {n_lora / 1e6:.3f}M trainable adapter params (base frozen: {base})",
                flush=True,
            )
    if compile_model:  # after the LoRA injection: the adapter must be in the compiled graph
        model.backbone.compile_blocks(compile_mode)  # regional torch.compile (see Backbone.compile_blocks)
    if compile_vae:  # the live x0 encode runs every step
        model.vae.compile_codec(compile_mode)

    # LoRA trains only the (requires_grad) adapter tensors; a full run trains the Stage-1 parameter set.
    params = [p for p in model.parameters() if p.requires_grad] if lora else trainable_parameters(model)
    # fused=True: single-kernel optimizer step (vs the foreach default) — free on CUDA. bf16 autocast uses
    # no GradScaler, so the fused+scaler caveat doesn't apply; fused is lazily device-checked at the first
    # step() (after accel.prepare has moved params to CUDA), so constructing on CPU params here is fine.
    opt = torch.optim.AdamW(
        params, lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay, fused=torch.cuda.is_available()
    )
    # Warmup -> cosine-decay LR: warm up then decay when training the generator from scratch;
    # constant LR is harder to converge from random init.
    # accelerate's prepared scheduler advances once PER PROCESS per optimizer step, so scale the
    # horizon by num_processes — otherwise an N-GPU run consumes the whole schedule N x too fast
    # (LR reaches its floor at ~steps/N and the rest of the run trains at ~0 LR).
    sched_scale = accel.num_processes
    lr_scheduler = _warmup_cosine_min_lr(
        opt,
        warmup_steps=warmup_steps * sched_scale,
        total_steps=steps * sched_scale,
        base_lr=lr,
        min_lr=min_lr,
    )
    model, opt, dataloader, lr_scheduler = accel.prepare(model, opt, dataloader, lr_scheduler)
    # Build EMA AFTER prepare so its shadow copies are created on the training device. accel.prepare
    # moves params to CUDA in place, so `params` still references the (now CUDA-backed) objects;
    # constructing EMA on the pre-prepare CPU model would crash on the first ema.step (CPU vs CUDA).
    # foreach=True: vectorized (_foreach_) EMA update over the whole param list in one shot vs a Python
    # per-tensor loop. use_ema_warmup: low decay early, ramp to 0.9999. (foreach isn't serialized in the EMA
    # state_dict, so toggling it is resume-compatible with existing checkpoints.)
    ema = EMAModel(params, use_ema_warmup=True, foreach=True) if use_ema else None
    if ema is not None:
        accel.register_for_checkpointing(ema)
    # Full resume: load model + optimizer + LR scheduler + EMA + RNG from an `accelerate` step_N checkpoint
    # (must run AFTER prepare + EMA registration so every registered object is restored). The step counter
    # comes from the dir name. Unlike --init-from (weights only -> restarts optimizer/LR/step), this
    # continues the run exactly. Resume on the SAME process count the checkpoint was saved with.
    start_step = 0
    if resume_from is not None:
        accel.load_state(resume_from)
        if ema is not None:
            # load_state restores the EMA shadow params on CPU; re-home them to the train device or the
            # next ema.step() mixes CPU shadow with CUDA params (device-mismatch crash).
            ema.to(accel.device)
        start_step = _resume_step(resume_from)
        if accel.is_main_process:
            print(f"resumed full training state from {resume_from} (step {start_step})", flush=True)
    if accel.is_main_process:
        os.makedirs(out_dir, exist_ok=True)

    unwrapped = accel.unwrap_model(model)
    dropouts = {"cond_dropout": cond_dropout, "text_dropout": text_dropout, "ref_dropout": ref_dropout}
    if (
        profile_steps > 0
    ):  # off by default; returns BEFORE the tracker and main loop, so training is untouched
        return _run_profile(accel, model, dataloader, params, out_dir, profile_steps, dropouts, profile_stack)

    # The run's full config (the model config + every train() knob + the CLI extras) to trackio, so runs are
    # comparable by every knob we sweep.
    run_cfg: dict[str, Any] = {
        **cfg.model_dump(),
        "steps": steps,
        "lr": lr,
        "weight_decay": weight_decay,
        "grad_accum": grad_accum,
        "max_grad_norm": max_grad_norm,
        "warmup_steps": warmup_steps,
        "min_lr": min_lr,
        **dropouts,
        "use_ema": use_ema,
        "grad_checkpoint": grad_checkpoint,
        "num_processes": accel.num_processes,
        "attention_backend": attention_backend,
        "init_from": init_from,
        "resume_from": resume_from,
        "compile": compile_model,
        "compile_mode": compile_mode,
        "compile_vae": compile_vae,
        "lora": lora,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "log_cer": log_cer,
        "val_every": val_every,
        "sample_every": sample_every,
        "save_every": save_every,
        **(run_config or {}),
    }
    if "batch_size" in run_cfg:
        run_cfg["effective_batch"] = run_cfg["batch_size"] * accel.num_processes * grad_accum
    tracker = _init_tracker(
        track, accel.is_main_process, run_cfg, track_space, run_name=os.path.basename(out_dir.rstrip("/"))
    )
    save = partial(_save_checkpoint, accel, unwrapped, cfg=cfg, ema=ema, params=params, lora=lora)

    # One fixed held-out batch on rank 0, reused for every sample strip so progress is comparable
    # step-to-step (same texts + same style references each time).
    sample_batch = next(iter(val_loader)) if val_loader is not None and accel.is_main_process else None
    recognizer = None
    if log_cer and sample_batch is not None:  # rank-0 legibility gauge: read our lines back -> gen_CER
        recognizer = load_recognizer(cfg.aux.htr_recognizer_eval, accel.device)
        print(f"loaded HTR recognizer for live gen_CER ({cfg.aux.htr_recognizer_eval})", flush=True)
    val_history: list[float] = []
    model.train()
    step = start_step  # 0 for a fresh run; the resumed optimizer-step count otherwise
    last_log_t, last_log_step = perf_counter(), start_step
    while step < steps:
        seen = 0
        for batch in dataloader:
            seen += 1
            with accel.accumulate(model):
                # REPA early-stop (HASTE): it helps early, then its gradient fights the denoiser
                repa_w = cfg.aux.repa_weight if step < cfg.aux.repa_stop_frac * steps else 0.0
                out = model(
                    batch["images"],
                    batch["texts"],
                    **dropouts,
                    return_losses=True,
                    repa_weight=repa_w,
                    **model_inputs(batch),
                )
                loss = out["loss"]
                accel.backward(loss)
                if accel.sync_gradients:  # gradient clipping
                    accel.clip_grad_norm_(params, max_grad_norm)
                opt.step()
                lr_scheduler.step()
                opt.zero_grad()
            # Gradient accumulation: run the per-OPTIMIZER-step bookkeeping (EMA, logging, save, val,
            # sample, the step counter) ONLY when grads actually synced — so `step` counts optimizer steps
            # (matching the LR horizon and cadences), not micro-batches. Diffusers' own training loops gate
            # on `accelerator.sync_gradients` the same way. No-op at grad_accum=1 (sync every iteration).
            if not accel.sync_gradients:
                continue
            if ema is not None:
                ema.step(params)
            if step % log_every == 0:
                # The losses stay on the GPU between log steps: a per-step .item() makes the CPU wait for
                # the GPU every step, which stalls kernel launching in this host-bound loop. Each rank checks
                # its own loss, so a NaN is caught at most log_every steps late. flow is the term comparable
                # to val_loss (REPA is training-only); log both so the overfitting check is apples-to-apples.
                loss_val = loss.item()
                flow_val = out["flow"].item()
                repa_val = out["repa"].item() if "repa" in out else None
                if not math.isfinite(loss_val):  # NaN/inf -> training is broken
                    if tracker is not None:
                        tracker.alert(
                            title="Non-finite loss",
                            text=f"loss={loss_val} at step {step}",
                            level=tracker.AlertLevel.ERROR,
                        )
                    raise RuntimeError(f"non-finite loss {loss_val} at step {step}")
                lr_now = lr_scheduler.get_last_lr()[0]
                msg = f"loss {loss_val:.4f}  flow {flow_val:.4f}"
                if repa_val is not None:
                    msg += f"  repa {repa_val:.4f}"
                msg += f"  lr {lr_now:.2e}"
                if accel.is_main_process:
                    now = perf_counter()
                    metrics: dict[str, float] = {"loss": loss_val, "flow_loss": flow_val, "lr": lr_now}
                    if repa_val is not None:
                        metrics["repa_loss"] = repa_val
                    if step > last_log_step:  # throughput since the last log (none on the first)
                        lines = (
                            (step - last_log_step)
                            * batch["images"].shape[0]
                            * grad_accum
                            * accel.num_processes
                        )
                        metrics["images_per_s"] = lines / max(now - last_log_t, 1e-6)
                        msg += f"  {metrics['images_per_s']:.1f} img/s"
                    last_log_t, last_log_step = now, step
                    print(f"step {step}/{steps}  {msg}", flush=True)
                    if tracker is not None:
                        tracker.log(metrics, step=step)
            rank0_work = False  # whether rank 0 did extra work the other ranks must wait for
            if step > 0 and step % save_every == 0:
                save(os.path.join(out_dir, f"step_{step}"), final=False)
                if tracker is not None:
                    tracker.alert(
                        title="Checkpoint saved", text=f"step_{step}", level=tracker.AlertLevel.INFO
                    )
            if val_loader is not None and accel.is_main_process and step > 0 and step % val_every == 0:
                rank0_work = True
                val_loss = evaluate_val_loss(unwrapped, val_loader, accel.device)
                val_history.append(val_loss)
                print(f"step {step}/{steps}  val_loss {val_loss:.4f}", flush=True)
                if tracker is not None:
                    tracker.log({"val_loss": val_loss}, step=step)
                    if len(val_history) >= 3 and val_history[-1] > val_history[-2] > val_history[-3]:
                        tracker.alert(
                            title="Val loss rising",
                            text=f"val_loss rose 3 evals in a row to {val_loss:.4f} at step {step}: overfitting?",
                            level=tracker.AlertLevel.WARN,
                        )
            if sample_batch is not None and step > 0 and step % sample_every == 0:
                rank0_work = True
                # Sample from the EMA weights (what inference/export uses), not the noisier raw weights —
                # EMA shadows only the trainable params, which is exactly what is swapped.
                with _ema_weights(ema, params):
                    samples = generate_samples(
                        unwrapped,
                        sample_batch,
                        cfg,
                        accel.device,
                        cfg_scale=_GEN_CER_CFG_SCALE if recognizer is not None else 0.0,
                    )
                line = f"step {step}/{steps}  samples"
                payload: dict[str, object] = {}
                preds: list[str] | None = None
                if recognizer is not None:  # read our lines back -> gen_CER (the legibility gauge)
                    # micro-average CER (total edits / total chars), matching eval.py
                    reads = read_lines(recognizer[0], recognizer[1], samples.gens)
                    preds = reads
                    gen_cer = cer(reads, samples.texts)
                    payload["gen_cer"] = gen_cer
                    line += f"  gen_CER {gen_cer:.2f}"
                    # Controllability: is the read closer to the REQUESTED text than to a RANDOM other
                    # text? Pair each read with a different sample's text (roll by 1) and take the CER
                    # gap. >0 = the output follows the instruction; ~0 = marginal collapse / copying.
                    if len(samples.texts) > 1:
                        ctrl_gap = cer(reads, samples.texts[1:] + samples.texts[:1]) - gen_cer
                        payload["ctrl_gap"] = ctrl_gap
                        line += f"  ctrl_gap {ctrl_gap:+.2f}"
                    if samples.gens_cfg is not None:
                        gen_cer_cfg = cer(
                            read_lines(recognizer[0], recognizer[1], samples.gens_cfg), samples.texts
                        )
                        payload["gen_cer_cfg"] = gen_cer_cfg
                        line += f"  gen_CER_cfg {gen_cer_cfg:.2f}"
                sheet = _compose_sample_grid(
                    samples.reals,
                    samples.gens,
                    samples.texts,
                    samples.width,
                    cfg.data.line_height,
                    reads=preds,
                    style_texts=samples.style_texts,
                )
                path = os.path.join(out_dir, "samples", f"step_{step:06d}.png")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                sheet.save(path)
                print(f"{line}  -> {path}", flush=True)
                if tracker is not None:  # noise->ink + prompt->reading + gen_CER, by step, in trackio
                    caption = f"step {step} — top: real, bottom: generated"
                    if preds is not None:
                        pairs = "  |  ".join(
                            f"style {s!r} → want {t!r} → read {p!r}"
                            for s, t, p in zip(samples.style_texts, samples.texts, preds, strict=False)
                        )
                        caption = (
                            f"step {step} | gen_CER (0=perfect, unguided) | top=real, bottom=ours | {pairs}"
                        )
                    payload["samples"] = tracker.Image(path, caption=caption)
                    tracker.log(payload, step=step)
            if rank0_work:
                # The other ranks idled through rank 0's val/sample work; barrier so a slow sample step
                # (CFG + TrOCR generate) can't trip the NCCL watchdog at the next all-reduce.
                accel.wait_for_everyone()
            step += 1
            if step >= steps:
                break
        if seen == 0:
            raise RuntimeError(
                "DataLoader yielded 0 batches — the dataset is smaller than batch_size × num_processes "
                "with drop_last=True, which would hang forever. Lower --batch-size or add more data."
            )
    accel.wait_for_everyone()
    if save_final:
        save(os.path.join(out_dir, "final"), final=True)
    if tracker is not None:
        tracker.finish()
    return unwrapped


def build_dataloader(
    cfg: Config,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    *,
    split: str = "train",
    shuffle: bool = True,
    strict: bool = True,
    bucket_width: bool = False,
) -> DataLoader:
    """Build a DataLoader over ``{data_dir}/{split}.jsonl`` (from ``make data``).

    Args:
        split: Manifest split to load (``train`` / ``val`` / ``test``).
        shuffle: Shuffle and drop the last partial batch (training), with ``cfg.data.white_pad_prob``
            augmentation. ``False`` keeps every row in order and text-tight — the held-out val loader, so
            the overfitting metric is stable and honest.
        strict: Raise if the split has fewer than ``batch_size × num_processes`` rows. ``True`` for
            train (a too-small set hangs DDP under ``drop_last``); ``False`` for val.
        bucket_width: Batch similar-width lines together (``WidthBucketBatchSampler``; training only).

    With ``cfg.data.drop_wide_lines`` the lines wider than ``max_line_width`` are skipped instead of squashed.

    Raises:
        ValueError: if ``strict`` and the split has fewer than ``batch_size × num_processes`` rows.
    """
    rows = load_manifest(os.path.join(data_dir, f"{split}.jsonl"))
    canvas = cfg.cond.style_mode == "canvas"
    widths = (  # raw px widths (cached sidecar), when something needs them
        line_widths(rows, cache_path=os.path.join(data_dir, f"{split}.widths.json"))
        if cfg.data.drop_wide_lines or canvas or (bucket_width and shuffle)
        else None
    )
    if cfg.data.drop_wide_lines and widths is not None:
        keep = [i for i, w in enumerate(widths) if w <= cfg.data.max_line_width]
        if len(keep) < len(rows) and PartialState().is_main_process:
            print(
                f"{split}: skipping {len(rows) - len(keep)} of {len(rows)} lines wider than "
                f"{cfg.data.max_line_width} px",
                flush=True,
            )
        rows, widths = [rows[i] for i in keep], [widths[i] for i in keep]
    ds = HandwritingLineDataset(
        rows,
        height=cfg.data.line_height,
        max_width=cfg.data.max_line_width,
        ref_lines=canvas,
        widths=widths,
        max_ref_width=cfg.data.max_ref_width,
    )
    world = PartialState().num_processes
    if strict and len(ds) < batch_size * world:
        raise ValueError(
            f"{split}.jsonl has {len(ds)} rows < batch_size × num_processes "
            f"({batch_size} × {world} = {batch_size * world}); with drop_last=True every rank would "
            f"get 0 batches and training would hang. Lower --batch-size or add more data."
        )
    collate = partial(
        collate_lines,
        max_width=cfg.data.max_line_width,
        white_pad_prob=cfg.data.white_pad_prob if shuffle else 0.0,
        white_pad_max_frac=cfg.data.white_pad_max_frac,
        max_ref_width=cfg.data.max_ref_width,
    )
    if bucket_width and shuffle and widths is not None:
        # Group similar-width lines per batch -> ~54% padding-FLOPs reclaimed + stable shapes (lets
        # torch.compile stop recompiling). accelerate's prepared loader shards whole batches across ranks
        # (BatchSamplerShard); DDP all-reduces param-shaped grads, so per-rank differing widths are fine.
        capped = [min(w, cfg.data.max_line_width) for w in widths]
        batch_sampler = WidthBucketBatchSampler(capped, batch_size, seed=cfg.seed)
        return DataLoader(
            ds,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=shuffle,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def _hf_loaders(cfg: Config, args: argparse.Namespace) -> tuple[DataLoader, DataLoader | None]:
    """Train (+ val) loaders straight off a Hub image+text dataset (e.g. Teklia/IAM-line)."""
    from .data.hf_dataset import HFLineDataset

    dataset = partial(
        HFLineDataset,
        args.hf_dataset,
        height=cfg.data.line_height,
        max_width=cfg.data.max_line_width,
        ref_lines=cfg.cond.style_mode == "canvas",
    )
    collate = partial(
        collate_lines,
        max_width=cfg.data.max_line_width,
        white_pad_max_frac=cfg.data.white_pad_max_frac,
        max_ref_width=cfg.data.max_ref_width,
    )
    loader = partial(
        DataLoader, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0
    )
    train_ds = dataset(split=args.hf_split, max_rows=args.hf_max_rows)
    train_loader = loader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=partial(collate, white_pad_prob=cfg.data.white_pad_prob),
    )
    if not args.hf_val_split:
        return train_loader, None
    try:  # a small held-out val loader for the overfitting / sample gauge
        val_ds = dataset(split=args.hf_val_split, max_rows=_HF_VAL_ROWS)
    except Exception as exc:  # noqa: BLE001 - a missing/renamed val split must not kill training
        print(f"(no HF val split {args.hf_val_split!r}: {exc})", flush=True)
        return train_loader, None
    return train_loader, loader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)


def main() -> None:
    ap = argparse.ArgumentParser(description="Diffu Stage-1 flow-matching training.")
    ap.add_argument(
        "--data-dir", default=None, help="dir with train.jsonl (from `make data`); or use --hf-dataset"
    )
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument(
        "--save-every", type=int, default=10_000, help="checkpoint every N steps (a resumable state: ~17 GB)"
    )
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument(
        "--no-grad-checkpoint",
        action="store_true",
        help="disable grad checkpointing (~20%% faster, more VRAM)",
    )
    ap.add_argument(
        "--compile",
        action="store_true",
        help="regionally torch.compile the MMDiT blocks (~1.4-1.9x; fastest with --no-grad-checkpoint)",
    )
    ap.add_argument(
        "--compile-mode",
        choices=("max-autotune-no-cudagraphs",),
        default=None,
        help="with --compile: autotune the GEMMs (+ diffusers' SD3 Inductor flags); slower first compile. "
        "CUDA-graph modes are inference-only (diffu-generate)",
    )
    ap.add_argument(
        "--compile-vae", action="store_true", help="torch.compile the VAE encode (the live x0 encode)"
    )
    ap.add_argument(
        "--no-repa",
        dest="repa",
        action="store_false",
        help="disable the REPA representation-alignment aux loss",
    )
    ap.add_argument(
        "--no-ink-focal",
        dest="ink_focal",
        action="store_false",
        help="don't up-weight the loss on ink cells vs the paper (on in every production run so far)",
    )
    ap.add_argument(
        "--track", action="store_true", help="log to trackio (live dashboard + alerts); needs --extra track"
    )
    ap.add_argument(
        "--track-space",
        default=None,
        help="HF Space id to sync trackio to (created private), e.g. user/diffu",
    )
    ap.add_argument(
        "--val-every", type=int, default=500, help="held-out val loss every N steps (overfitting check)"
    )
    ap.add_argument(
        "--sample-every", type=int, default=2000, help="save [real|generated] sample images every N steps"
    )
    ap.add_argument(
        "--no-cer", action="store_true", help="disable the live gen_CER legibility gauge at sample steps"
    )
    ap.add_argument("--max-grad-norm", type=float, default=1.0, help="gradient clipping norm")
    ap.add_argument("--warmup-steps", type=int, default=2000, help="LR warmup steps before cosine decay")
    ap.add_argument("--min-lr", type=float, default=1e-6, help="cosine LR floor (decays toward this, not 0)")
    ap.add_argument(
        "--weight-decay", type=float, default=1e-4, help="AdamW weight decay (lighter than the 1e-2 default)"
    )
    ap.add_argument(
        "--cond-dropout", type=float, default=0.0, help="drop ALL conditioning prob (full CFG null); 0=off"
    )
    ap.add_argument(
        "--text-dropout",
        type=float,
        default=0.0,
        help="drop ONLY the text/content prob, KEEP style (text-only CFG, DiffInk drop_text); 0=off",
    )
    ap.add_argument(
        "--ref-dropout",
        type=float,
        default=0.0,
        help="--style-canvas: drop ONLY the reference (pixels + its text), KEEP the target text — trains the "
        "mode style guidance contrasts against (DiffInk uses 0.2); 0=off",
    )
    ap.add_argument(
        "--drop-wide-lines",
        action="store_true",
        help="skip lines wider than --max-line-width instead of squashing them (keeps every hand's aspect)",
    )
    ap.add_argument(
        "--attention-backend",
        default=None,
        help="diffusers attention kernel, e.g. _native_cudnn / flash / flash_hub (default: torch SDPA). "
        "Needs a backward kernel for training (not sage*); benchmark on your GPU",
    )
    ap.add_argument(
        "--init-from", default=None, help="warm-start model weights from a checkpoint (strict=False)"
    )
    ap.add_argument(
        "--resume",
        default=None,
        help="FULL resume from an accelerate step_N checkpoint dir (model+optimizer+LR+EMA+step), or 'auto' "
        "for the newest complete one in --out-dir (a fresh start if none: chained Slurm jobs). Unlike "
        "--init-from (weights only -> restarts optimizer/LR/step), this continues the run; use the SAME "
        "number of GPUs the checkpoint was saved with.",
    )
    ap.add_argument(
        "--profile-steps",
        type=int,
        default=0,
        help="profile N real training steps to out-dir/profile_step_table.txt + trace.json, then exit (0=off)",
    )
    ap.add_argument(
        "--profile-stack",
        action="store_true",
        help="--profile-steps: attribute kernels to Python lines (slower)",
    )
    ap.add_argument(
        "--white-pad-prob",
        type=float,
        default=0.0,
        help="train-only right-white-pad augmentation prob (short-text-on-wide-canvas); 0=off",
    )
    ap.add_argument(
        "--bucket-width",
        action="store_true",
        help="group similar-width lines into each batch (reclaims ~54%% padding-FLOPs + stabilizes shapes "
        "so --compile stops recompiling). Batches become width-correlated (small training-dynamics change) "
        "-> validate gen_CER before treating as default.",
    )
    ap.add_argument(
        "--logit-normal-mean",
        type=float,
        default=None,
        help="override flow timestep logit-normal mean (default 0.0); <0 emphasizes low-noise detail",
    )
    # --- LoRA (Stage-3): freeze the base, train a low-rank adapter on the backbone; saves a few-MB adapter.
    ap.add_argument(
        "--lora",
        action="store_true",
        help="LoRA fine-tune: freeze the base, train only a low-rank backbone adapter (pair with --init-from "
        "a base checkpoint). Saves a few-MB adapter, not a full 17 GB state.",
    )
    ap.add_argument("--lora-rank", type=int, default=16, help="LoRA rank r (adapter capacity)")
    ap.add_argument(
        "--lora-alpha", type=int, default=32, help="LoRA alpha (scaling; effective scale = alpha/r)"
    )
    ap.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout")
    # --- Train straight off a Hugging Face dataset (image + text) instead of a local manifest.
    ap.add_argument(
        "--hf-dataset",
        default=None,
        help="train off a Hub datasets image+text set (e.g. Teklia/IAM-line) instead of --data-dir; "
        "needs `uv sync --extra hf`",
    )
    ap.add_argument("--hf-split", default="train", help="HF dataset split to train on")
    ap.add_argument(
        "--hf-val-split", default="validation", help="HF split for the held-out val loader ('' to skip)"
    )
    ap.add_argument(
        "--hf-max-rows", type=int, default=None, help="cap the HF train split to N rows (smoke/dev)"
    )
    ap.add_argument(
        "--htr-recognizer-eval",
        default=None,
        help="HF TrOCR id for the live gen_CER legibility gauge (default: the Swedish char-level model). "
        "Pass microsoft/trocr-base-handwritten for an English corpus like IAM, else the gauge reads gibberish.",
    )
    add_arch_args(ap)
    args = ap.parse_args()
    # Fail before any data scan or model build.
    if args.lora and args.resume:
        raise SystemExit("--resume can't restore a --lora run: its checkpoints hold only the adapter")
    if args.hf_dataset and (args.bucket_width or args.drop_wide_lines):
        raise SystemExit("--bucket-width / --drop-wide-lines need a --data-dir manifest, not --hf-dataset")
    if not (args.hf_dataset or args.data_dir):
        raise SystemExit("provide --data-dir (a `make data` manifest) or --hf-dataset (a Hub dataset)")
    if args.resume == "auto":
        args.resume = _latest_checkpoint(args.out_dir)

    cfg = apply_arch_args(Config(), args)
    cfg.aux.repa = args.repa
    cfg.aux.ink_focal = args.ink_focal
    if args.htr_recognizer_eval:  # override the default (Swedish) gauge recognizer — e.g. English for IAM
        cfg.aux.htr_recognizer_eval = args.htr_recognizer_eval
    cfg.data.drop_wide_lines = args.drop_wide_lines
    if args.logit_normal_mean is not None:
        cfg.flow.logit_normal_mean = args.logit_normal_mean
    cfg.data.white_pad_prob = args.white_pad_prob  # train-only: the val loaders stay text-tight

    if args.hf_dataset:
        dataloader, val_loader = _hf_loaders(cfg, args)
    else:
        dataloader = build_dataloader(
            cfg, args.data_dir, args.batch_size, args.num_workers, bucket_width=args.bucket_width
        )
        val_loader = build_dataloader(
            cfg, args.data_dir, args.batch_size, args.num_workers, split="val", shuffle=False, strict=False
        )
    train(
        cfg,
        dataloader,
        steps=args.steps,
        lr=args.lr,
        out_dir=args.out_dir,
        use_ema=not args.no_ema,
        compile_model=args.compile,
        grad_accum=args.grad_accum,
        save_every=args.save_every,
        track=args.track,
        track_space=args.track_space,
        grad_checkpoint=not args.no_grad_checkpoint,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        cond_dropout=args.cond_dropout,
        text_dropout=args.text_dropout,
        ref_dropout=args.ref_dropout,
        init_from=args.init_from,
        resume_from=args.resume,
        val_loader=val_loader,
        val_every=args.val_every,
        sample_every=args.sample_every,
        log_cer=not args.no_cer,
        profile_steps=args.profile_steps,
        profile_stack=args.profile_stack,
        lora=args.lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        attention_backend=args.attention_backend,
        compile_mode=args.compile_mode,
        compile_vae=args.compile_vae,
        run_config={
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "data_dir": args.data_dir,
            "hf_dataset": args.hf_dataset,
            "bucket_width": args.bucket_width,
        },
    )


if __name__ == "__main__":
    main()
