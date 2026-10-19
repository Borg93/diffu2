"""Training loop + entrypoint for Diffu (diffusers-first, Stage-1 flow matching).

Modern optimizations baked in: bf16 mixed precision (accelerate), gradient checkpointing
(diffusers), and EMA weights (diffusers EMAModel, one-pass update: diffu.ema). torch.compile optional (bucket line widths
to limit recompiles). FlashAttention comes for free via SD3's SDPA attention. Single- and
multi-GPU via `accelerate launch` (see the Makefile `train-1gpu` / `train-2gpu` targets).

Stages: 0) VAE decoder fine-tune + recon-CER gate (run separately via `diffu-vae-gate`),
1) synthetic pretrain, 2) real Swedish adaptation, 3) per-collection LoRA.

Run (prefer the Makefile, which pins CUDA_VISIBLE_DEVICES so the busy GPU 2 is never touched):
    make train-1gpu DATA_DIR=data_out          # GPU 0 only
    make train-2gpu DATA_DIR=data_out          # GPU 0 + 1

Equivalent raw command (2 GPUs):
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 \
        -m diffu.train --data-dir data_out --steps 100000 --batch-size 8
"""

from __future__ import annotations

import argparse
import math
import os
from functools import partial
from time import perf_counter
from typing import TYPE_CHECKING, Any

import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs, set_seed
from torch.utils.data import DataLoader

from .config import Config
from .data.collate import collate_lines
from .data.dataset import HandwritingLineDataset, load_manifest
from .ema import LerpEMAModel
from .model import Diffu

if TYPE_CHECKING:
    from PIL import ImageFont
    from PIL.Image import Image as PILImage

    from .data.latent_cache import LatentCache


def trainable_parameters(model: Diffu) -> list[torch.nn.Parameter]:
    """Parameters trained in Stage 1 (flow matching).

    Trained: SD3 backbone, DINOv3 style resampler (``style_mode="dino"`` only), Unifont content encoder.
    Frozen in Stage 1: the whole VAE — the Stage-1 forward never decodes, so it cannot receive a
    gradient here (and including it would trip DDP's unused-parameter check on multi-GPU). VAE
    decoder fine-tuning is Stage 0 (`diffu-vae-gate`).
    """
    params = list(model.backbone.parameters())
    if model.style is not None and not model.canvas:  # canvas mode: DINO is only REPA's frozen target
        params += list(model.style.resampler.parameters())
        params += list(model.style.pool.parameters())  # learned-query style attention-pool
    params += list(model.glyph_content.parameters())
    if model.glyph_latent is not None:  # glyph_concat conv "glyph block"
        params += list(model.glyph_latent.parameters())
    if model.fill_ratio_mlp is not None:  # fill-ratio AdaLN conditioning MLP (EMA shadows it too)
        params += list(model.fill_ratio_mlp.parameters())
    if model.repa_proj is not None:  # REPA alignment head (when cfg.aux.repa)
        params += list(model.repa_proj.parameters())
    return [p for p in params if p.requires_grad]


_REF_BATCH_KEYS = ("ref_images", "ref_latents", "ref_texts", "ref_widths")


def model_inputs(batch: dict[str, Any], device: torch.device | None = None) -> dict[str, Any]:
    """The optional model kwargs a batch carries: ``text_widths`` (glyph_pad) + the canvas reference.

    Moves tensors to ``device`` when given (the val/sample batches are not accelerate-prepared).
    """
    out: dict[str, Any] = {"text_widths": batch.get("widths")}
    for key in _REF_BATCH_KEYS:
        value = batch.get(key)
        if value is not None:
            out[key] = value.to(device) if device is not None and isinstance(value, torch.Tensor) else value
    return out


@torch.no_grad()
def evaluate_val_loss(model: Diffu, val_loader: DataLoader, device: torch.device, max_batches: int) -> float:
    """Mean flow loss on held-out (volume-disjoint) lines.

    The overfitting signal: this rising while train loss keeps falling = the model is memorizing
    the training volumes instead of learning generalizable handwriting. Averaged over several
    batches to damp the per-batch timestep noise so the trend is readable.
    """
    model.eval()
    total, seen = 0.0, 0
    try:
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            loss = model(
                batch["images"].to(device),
                batch["texts"],
                batch["style_pixel_values"].to(device),
                **model_inputs(batch, device),
            )
            total += loss.item()
            seen += 1
    finally:
        model.train()  # always restore train mode, even if a batch raises (shared module, rank-0 toggle)
    return total / max(seen, 1)


def _label_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """A label font that can render Swedish glyphs (å ä ö). DejaVu if present, else PIL's default
    (the bitmap default mangles non-ASCII, but a missing font must never crash a training run)."""
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


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

    lbl_w, cap_h = 88, 22
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


@torch.no_grad()
def save_samples(
    model: Diffu,
    batch: dict[str, Any],
    cfg: Config,
    device: torch.device,
    out_dir: str,
    step: int,
    n: int = 4,
    cfg_eval_scale: float = 0.0,
) -> tuple[str, list[PILImage], list[PILImage] | None, list[str], list[PILImage], int, list[str]]:
    """Generate n held-out lines and save a labeled ``[REAL / OURS]`` strip to eyeball progress.

    Each pair is captioned with the requested text; the real target (green REAL tag) sits above the
    model's attempt (red OURS tag) — so legibility and any overflow are visible at a glance in the
    trackio media tab without spinning up the app or decoding a 17 GB checkpoint.

    Returns ``(path, gen_pils, gen_cfg_pils, texts, real_pils, width)``. ``gen_pils`` is the raw
    (unguided) generation — also what the visual grid shows. If ``cfg_eval_scale > 0`` a SECOND,
    CFG-guided generation is run and returned as ``gen_cfg_pils`` (else ``None``); the caller reads
    both back to log gen_CER with AND without guidance, so we can see whether CFG actually helps.
    ``real_pils`` + ``width`` are returned so the caller can re-draw the grid with the reads burned in.
    """
    from .generate import to_pil

    n = min(n, len(batch["texts"]))
    texts = batch["texts"][:n]
    style_texts = batch.get("style_texts", [""] * len(batch["texts"]))[:n]
    style = batch["style_pixel_values"][:n].to(device)
    real = batch["images"][:n]
    w = real.shape[-1]
    latent_hw = (cfg.data.line_height // cfg.vae.downscale_factor, w // cfg.vae.downscale_factor)
    # glyph_pad widths + the canvas reference, sliced to the n samples (cached ref latents are train-only).
    gen_kwargs = {
        k: v[:n] for k, v in model_inputs(batch, device).items() if v is not None and k != "ref_latents"
    }
    if "ref_texts" in gen_kwargs:  # caption the sheet with what the in-canvas reference actually says
        style_texts = list(gen_kwargs["ref_texts"])

    # Sampling runs in eval mode; the try/finally guarantees train mode is restored even if a sampling
    # step raises. Otherwise the shared module is left in eval() and the next training steps silently
    # skip the CFG cond/text-dropout (it gates on `self.training`) — quietly breaking CFG training.
    model.eval()
    try:
        # The VISUAL grid is unguided: high guidance over-saturates an undertrained model's output to
        # pure white, hiding the real (on-manifold) generation. Raw output shows true progress.
        gen = model.generate(texts, style, latent_hw=latent_hw, num_steps=cfg.flow.sample_steps, **gen_kwargs)
        gen_pils = [to_pil(gen[i]) for i in range(n)]
        # The CFG variant is for MEASUREMENT only (gen_CER with guidance), not the grid — same texts/style
        # so the two CER numbers are directly comparable at this step.
        gen_cfg_pils: list[PILImage] | None = None
        if cfg_eval_scale > 0.0:
            gen_cfg = model.generate(
                texts,
                style,
                latent_hw=latent_hw,
                num_steps=cfg.flow.sample_steps,
                cfg_scale=cfg_eval_scale,
                **gen_kwargs,
            )
            gen_cfg_pils = [to_pil(gen_cfg[i]) for i in range(n)]
    finally:
        model.train()

    real_pils = [to_pil(real[i]) for i in range(n)]
    canvas = _compose_sample_grid(
        real_pils, gen_pils, list(texts), w, cfg.data.line_height, style_texts=list(style_texts)
    )
    sample_dir = os.path.join(out_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    path = os.path.join(sample_dir, f"step_{step:06d}.png")
    canvas.save(path)
    return path, gen_pils, gen_cfg_pils, list(texts), real_pils, w, list(style_texts)


def _setup_model(
    cfg: Config,
    *,
    grad_checkpoint: bool,
    compile_model: bool,
    attention_backend: str | None = None,
    compile_mode: str | None = None,
    compile_vae: bool = False,
) -> Diffu:
    """Build the model and freeze everything Stage-1 flow matching never touches.

    The VAE (encode is no_grad, decode unused) is frozen so DDP's reducer doesn't wait on gradients
    that never arrive, which crashes on step 1 under ``--multi_gpu``.
    """
    model = Diffu(cfg)
    if grad_checkpoint:
        model.backbone.enable_optimizations()  # gradient checkpointing (off = ~20% faster on big VRAM)
    if attention_backend:
        model.backbone.set_attention_backend(attention_backend)  # before compile: it is traced into the graph
    if compile_model:
        model.backbone.compile_blocks(compile_mode)  # regional torch.compile (see Backbone.compile_blocks)
    if compile_vae:  # the live encode (no --cache-latents) is ~13-23% of a step (docs/PERF_AUDIT.md)
        model.vae.compile_codec(compile_mode)
    model.vae.requires_grad_(False)
    return model


def _init_tracker(
    track: bool, is_main: bool, config: dict[str, Any], track_space: str | None, run_name: str | None = None
) -> Any:
    """Initialise trackio on the main process (live dashboard + NaN alerts); return None otherwise.

    ``run_name`` (the out-dir basename) is passed so parallel runs get distinct dashboard names —
    trackio auto-names collide when several runs launch at once with the same seed.
    """
    if not (track and is_main):
        return None
    import trackio

    trackio.init(project="diffu", name=run_name, config=config, space_id=track_space)
    return trackio


def _run_profile(
    accel: Accelerator,
    model: torch.nn.Module,
    dataloader: DataLoader,
    params: list[torch.nn.Parameter],
    out_dir: str,
    profile_steps: int,
    cond_dropout: float,
) -> Diffu:
    """Profile ``profile_steps`` real forward+backward steps and write the kernel report.

    Mirrors the Stage-1 training step (model forward under accelerate's bf16 autocast, backward,
    grad zeroing) but skips the optimizer/scheduler/EMA so the profile reflects compute only. The
    report is written to ``out_dir/profile_step_table.txt`` on the main process. Called from
    :func:`train` only when ``profile_steps > 0``; returns the unwrapped model immediately after.
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
        loss = model(
            batch["images"],
            batch["texts"],
            batch["style_pixel_values"],
            cond_dropout=cond_dropout,
            **model_inputs(batch),
        )
        accel.backward(loss)
        for p in params:
            p.grad = None

    model.train()
    report = profile_callable(
        step,
        warmup=2,
        active=profile_steps,
        trace_path=os.path.join(out_dir, "trace.json"),
        with_stack=os.environ.get("DIFFU_PROFILE_STACK") == "1",  # opt-in Python-line attribution
    )
    if accel.is_main_process:
        os.makedirs(out_dir, exist_ok=True)
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


def _save_checkpoint(accel: Accelerator, unwrapped: Diffu, out_dir: str, *, lora: bool) -> None:
    """Checkpoint the run: a few-MB LoRA adapter (``lora``) or a full ``accelerate`` state (default).

    ``save_state`` is a collective call (all ranks); the LoRA adapter is written on the main process only,
    with a barrier so the others don't race ahead of the write.
    """
    if lora:
        if accel.is_main_process:
            from .lora import save_lora

            save_lora(unwrapped, out_dir)
        accel.wait_for_everyone()
    else:
        accel.save_state(out_dir)


def train(
    cfg: Config,
    dataloader: DataLoader,
    *,
    steps: int = 100_000,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    log_every: int = 50,
    save_every: int = 2000,
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
    val_batches: int = 8,
    sample_every: int = 2000,
    log_cer: bool = True,
    profile_steps: int = 0,
    lora: bool = False,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    lora_targets: tuple[str, ...] | None = None,
    run_config: dict[str, Any] | None = None,
    attention_backend: str | None = None,
    compile_mode: str | None = None,
    compile_vae: bool = False,
) -> Diffu:
    """Run the Stage-1 flow-matching training loop.

    Args:
        cfg: Full model/data config.
        dataloader: Yields batches of ``images`` / ``texts`` / ``style_pixel_values``.
        steps: Total optimizer steps.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        log_every: Print the loss every N steps (main process only).
        save_every: Checkpoint (``accelerate save_state``) every N steps.
        out_dir: Directory for checkpoints.
        use_ema: Maintain EMA weights of the trained parameters.
        compile_model: ``torch.compile`` the SD3 transformer.
        grad_accum: Gradient accumulation steps.
        save_final: Write a final ``accelerate`` checkpoint to ``out_dir/final`` when done.
        track: Log loss to trackio (live dashboard) + fire a trackio alert on non-finite loss.
        track_space: HF Space id to sync trackio metrics to (persists for remote/cloud runs).
        profile_steps: If > 0, run torch.profiler over this many real training steps, write the
            kernel/Tensor-Core report to ``out_dir/profile_step_table.txt``, and return BEFORE the
            main training loop. 0 (default) leaves the training path unchanged.

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
    torch.set_float32_matmul_precision("high")  # TF32 on fp32 GEMMs (VAE gate / generate paths)
    set_seed(cfg.seed)  # reproducible init + data order across ranks
    model = _setup_model(
        cfg,
        grad_checkpoint=grad_checkpoint,
        compile_model=compile_model and not lora,  # LoRA must be injected BEFORE the backbone is compiled
        attention_backend=attention_backend,
        compile_mode=compile_mode,
        compile_vae=compile_vae,
    )
    if init_from is not None:  # warm-start from prior weights (strict=False) instead of random init
        from .generate import load_checkpoint

        load_checkpoint(model, init_from)
        if accel.is_main_process:
            print(f"warm-started weights from {init_from}", flush=True)

    if lora:  # Stage-3: freeze the base, train only a low-rank adapter on the backbone (diffu.lora)
        from .lora import add_lora

        n_lora = add_lora(model, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout, targets=lora_targets)
        if compile_model:
            model.backbone.compile_blocks(compile_mode)  # compile now that the adapter is in place
        if accel.is_main_process:
            base = "random init — pass --init-from for a real adapter" if init_from is None else init_from
            print(f"LoRA rank={lora_rank}: {n_lora / 1e6:.3f}M trainable adapter params (base frozen: {base})", flush=True)

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
    ema = LerpEMAModel(params, use_ema_warmup=True, foreach=True) if use_ema else None
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

    # Full experiment config to trackio so runs are comparable by every knob we sweep (batch, LR,
    # white-pad, rope, fill-ratio, grad-clip, …). run_config carries the CLI-only extras (batch_size).
    run_cfg: dict[str, Any] = {
        "steps": steps,
        "lr": lr,
        "weight_decay": weight_decay,
        "grad_accum": grad_accum,
        "max_grad_norm": max_grad_norm,
        "warmup_steps": warmup_steps,
        "min_lr": min_lr,
        "cond_dropout": cond_dropout,
        "text_dropout": text_dropout,
        "ref_dropout": ref_dropout,
        "use_ema": use_ema,
        "grad_checkpoint": grad_checkpoint,
        "repa": cfg.aux.repa,
        "ink_focal": cfg.aux.diacritic_focal_flow,
        "num_processes": accel.num_processes,
        "rope": cfg.backbone.rope,
        "fill_ratio_cond": cfg.backbone.fill_ratio_cond,
        "max_chars": cfg.cond.max_chars,
        "glyph_line": cfg.cond.glyph_line,
        "glyph_concat": cfg.cond.glyph_concat,
        "style_in_context": cfg.cond.style_in_context,
        "style_mode": cfg.cond.style_mode,
        "glyph_pad": cfg.cond.glyph_pad,
        "attention_backend": attention_backend,
        "white_pad_prob": cfg.data.white_pad_prob,
        "white_pad_max_frac": cfg.data.white_pad_max_frac,
        "line_height": cfg.data.line_height,
        "dim": cfg.backbone.dim,
        "num_layers": cfg.backbone.num_layers,
        "init_from": init_from,
        "compile": compile_model,
        "compile_mode": compile_mode,
        "compile_vae": compile_vae,
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

    unwrapped = accel.unwrap_model(model)
    if profile_steps > 0:  # off by default; returns BEFORE the main loop so training is untouched
        return _run_profile(accel, model, dataloader, params, out_dir, profile_steps, cond_dropout)
    # One fixed held-out batch on rank 0, reused for every sample strip so progress is comparable
    # step-to-step (same texts + same style references each time).
    sample_batch = next(iter(val_loader)) if val_loader is not None and accel.is_main_process else None
    recognizer = None
    if log_cer and sample_batch is not None:  # rank-0 legibility gauge: read our lines back -> gen_CER
        from .recognizer import load_recognizer

        recognizer = load_recognizer(cfg.aux.htr_recognizer_eval, accel.device)
        print(f"loaded HTR recognizer for live gen_CER ({cfg.aux.htr_recognizer_eval})", flush=True)
    model.train()
    step = start_step  # 0 for a fresh run; the resumed optimizer-step count otherwise
    last_log_t, last_log_step = perf_counter(), start_step
    while step < steps:
        seen = 0
        for batch in dataloader:
            seen += 1
            with accel.accumulate(model):
                repa_w = (
                    cfg.aux.repa_weight if step < cfg.aux.repa_stop_frac * steps else 0.0
                )  # REPA early-stop (HASTE)
                out = model(
                    batch["images"],
                    batch["texts"],
                    batch["style_pixel_values"],
                    latents=batch.get("latents"),  # cached x0 (--cache-latents); None -> live vae.encode
                    cond_dropout=cond_dropout,
                    text_dropout=text_dropout,
                    ref_dropout=ref_dropout,
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
            # sample, the barrier, and the step counter) ONLY when grads actually synced — so `step`
            # counts optimizer steps (matching the LR horizon / recog ramp / cadence), not micro-batches.
            # Diffusers' own training loops gate on `accelerator.sync_gradients` the same way. No-op at
            # grad_accum=1 (sync every iteration).
            if not accel.sync_gradients:
                continue
            if ema is not None:
                ema.step(params)
            if step % log_every == 0:
                # The losses stay on the GPU between log steps: a per-step .item() makes the CPU wait for
                # the GPU every step, which stalls kernel launching in this host-bound loop. Every rank checks
                # (so a NaN stops all ranks together); NaN is caught at most log_every steps late.
                # flow is the term comparable to val_loss (REPA is training-only); log both so the
                # train-flow vs val-flow overfitting check is apples-to-apples.
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
                msg = f"loss {loss_val:.4f}  flow {flow_val:.4f}"
                if repa_val is not None:
                    msg += f"  repa {repa_val:.4f}"
                msg += f"  lr {lr_scheduler.get_last_lr()[0]:.2e}"
                if accel.is_main_process:
                    now = perf_counter()
                    imgs_per_s = (
                        (step - last_log_step)
                        * batch["images"].shape[0]
                        * accel.num_processes
                        / max(now - last_log_t, 1e-6)
                    )
                    last_log_t, last_log_step = now, step
                    print(f"step {step}/{steps}  {msg}  {imgs_per_s:.0f} img/s", flush=True)
                    if tracker is not None:
                        lr_now = lr_scheduler.get_last_lr()[0]  # step= -> real-step x-axis
                        metrics: dict[str, float] = {"loss": loss_val, "lr": lr_now, "images_per_s": imgs_per_s}
                        if flow_val is not None:  # flow_loss is what val_loss is comparable to
                            metrics["flow_loss"] = flow_val
                        if repa_val is not None:
                            metrics["repa_loss"] = repa_val
                        tracker.log(metrics, step=step)
            if step > 0 and step % save_every == 0:
                _save_checkpoint(accel, unwrapped, os.path.join(out_dir, f"step_{step}"), lora=lora)
            if val_loader is not None and accel.is_main_process and step > 0 and step % val_every == 0:
                val_loss = evaluate_val_loss(unwrapped, val_loader, accel.device, val_batches)
                print(f"step {step}/{steps}  val_loss {val_loss:.4f}", flush=True)
                if tracker is not None:
                    tracker.log({"val_loss": val_loss}, step=step)
            if sample_batch is not None and step > 0 and step % sample_every == 0:
                # When the recognizer is loaded, also generate a CFG-guided variant so gen_CER is
                # reported WITH and WITHOUT guidance (does CFG actually help legibility here?).
                cfg_eval_scale = (
                    5.0 if recognizer is not None else 0.0
                )  # classic-CFG scale for guided gen_CER (cfg=5 measured best vs 3; see docs/PERF_AUDIT.md §0c)
                # Sample from the EMA weights (what inference/export uses), then restore the raw
                # training weights — otherwise the gauge measures noisier raw weights and understates
                # quality. EMA shadows only the trainable params, which is exactly what we swap.
                if ema is not None:
                    ema.store(params)
                    ema.copy_to(params)
                try:
                    path, gen_imgs, gen_imgs_cfg, texts, real_imgs, sample_w, style_texts = save_samples(
                        unwrapped,
                        sample_batch,
                        cfg,
                        accel.device,
                        out_dir,
                        step,
                        cfg_eval_scale=cfg_eval_scale,
                    )
                finally:
                    if ema is not None:
                        ema.restore(params)
                line = f"step {step}/{steps}  samples -> {path}"
                gen_cer: float | None = None
                gen_cer_cfg: float | None = None
                ctrl_gap: float | None = None
                preds: list[str] = []
                if recognizer is not None:  # read our lines back -> gen_CER (the legibility gauge)
                    from .model.metrics import cer
                    from .recognizer import read_lines

                    def _cer(imgs: list[PILImage], want: list[str]) -> tuple[list[str], float]:
                        # micro-average (total edits / total chars), matching eval.py. A macro mean of
                        # per-line ratios over-weights short/empty targets (empty gt scores a flat 1.0),
                        # so it read ~1.4x too high on mixed-length sample batches.
                        p = read_lines(recognizer[0], recognizer[1], imgs)
                        return p, cer(p, want)

                    preds, gen_cer = _cer(gen_imgs, texts)
                    line += f"  gen_CER {gen_cer:.2f}"
                    # Controllability: is the read closer to the REQUESTED text than to a RANDOM other
                    # text? Pair each read with a different sample's text (roll by 1) and take the CER
                    # gap. >0 = the output follows the instruction; ~0 = marginal collapse / copying (the
                    # output is no closer to what we asked than to a random line). Reuses `preds` (free).
                    if len(texts) > 1:
                        rand_texts = list(texts[1:]) + list(texts[:1])
                        ctrl_gap = cer(preds, rand_texts) - gen_cer
                        line += f"  ctrl_gap {ctrl_gap:+.2f}"
                    if gen_imgs_cfg is not None:
                        _, gen_cer_cfg = _cer(gen_imgs_cfg, texts)
                        line += f"  gen_CER_cfg {gen_cer_cfg:.2f}"
                    # redraw the grid with the reads burned in so the PNG itself shows want -> read
                    _compose_sample_grid(
                        real_imgs,
                        gen_imgs,
                        list(texts),
                        sample_w,
                        cfg.data.line_height,
                        reads=preds,
                        style_texts=style_texts,
                    ).save(path)
                print(line, flush=True)
                if tracker is not None:  # noise->ink + prompt->reading + gen_CER, by step, in trackio
                    caption = f"step {step} — top: real, bottom: generated"
                    if gen_cer is not None:
                        pairs = "  |  ".join(
                            f"style {s!r} → want {t!r} → read {p!r}"
                            for s, t, p in zip(style_texts, texts, preds, strict=False)
                        )
                        cfg_note = f" | gen_CER_cfg {gen_cer_cfg:.2f}" if gen_cer_cfg is not None else ""
                        caption = (
                            f"step {step} | gen_CER {gen_cer:.2f} (0=perfect, unguided){cfg_note} | "
                            f"image: top=real scan, bottom=ours | {pairs}"
                        )
                    payload: dict[str, object] = {"samples": tracker.Image(path, caption=caption)}
                    if gen_cer is not None:
                        payload["gen_cer"] = gen_cer
                    if gen_cer_cfg is not None:
                        payload["gen_cer_cfg"] = gen_cer_cfg
                    if ctrl_gap is not None:  # >0 = following the text; ~0 = marginal collapse / copying
                        payload["ctrl_gap"] = ctrl_gap
                    tracker.log(payload, step=step)
            # Other ranks idle through the rank-0-only val/sample blocks above; barrier here so a slow
            # sample step (CFG + TrOCR generate on rank 0) can't trip the NCCL watchdog at the next
            # all-reduce. Cheap when ranks are already in sync.
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
        # Export the EMA weights as the final model (generate.py loads a plain state-dict, so without
        # this the deployed model would be the raw, noisier weights and EMA's cost would be wasted).
        # Periodic step_N checkpoints stay raw for clean resume; only the final export is EMA-applied.
        # For a LoRA run this writes just the (EMA-applied) adapter safetensors, not a full state.
        if ema is not None:
            ema.store(params)
            ema.copy_to(params)
        _save_checkpoint(accel, unwrapped, os.path.join(out_dir, "final"), lora=lora)
        if ema is not None:
            ema.restore(params)
    if tracker is not None:
        tracker.finish()
    return accel.unwrap_model(model)


def build_dataloader(
    cfg: Config,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    *,
    split: str = "train",
    shuffle: bool = True,
    strict: bool = True,
    white_pad_prob: float = 0.0,
    bucket_width: bool = False,
    latent_cache: LatentCache | None = None,
) -> DataLoader:
    """Build a width-bucketing DataLoader over ``{data_dir}/{split}.jsonl`` (from ``make data``).

    Args:
        split: Manifest split to load (``train`` / ``val`` / ``test``).
        shuffle: Shuffle and drop the last partial batch (training). ``False`` keeps every row in
            order — used for the held-out val loader so the overfitting metric is stable.
        strict: Raise if the split has fewer than ``batch_size × num_processes`` rows. ``True`` for
            train (a too-small set hangs DDP under ``drop_last``); ``False`` for val.
        white_pad_prob: Train-only right-white-pad augmentation probability (0 = off). Pass >0 only
            for the train loader; val/test stay text-tight (white_pad_prob=0) for honest metrics.

    Raises:
        ValueError: if ``strict`` and the split has fewer than ``batch_size × num_processes`` rows.
    """
    from accelerate import PartialState

    rows = load_manifest(os.path.join(data_dir, f"{split}.jsonl"))
    ds = HandwritingLineDataset(
        rows,
        height=cfg.data.line_height,
        max_width=cfg.data.max_line_width,
        latent_cache=latent_cache,
        ref_lines=cfg.cond.style_mode == "canvas",
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
        white_pad_prob=white_pad_prob,
        white_pad_max_frac=cfg.data.white_pad_max_frac,
        # Cached-latent path: hand the collate the encoder's white-paper latent column so it can pad
        # batched latents correctly (no-op when latent_cache is None → the standard image-only batch).
        white_latent_col=latent_cache.white_column if latent_cache is not None else None,
        downscale=cfg.vae.downscale_factor,
        max_ref_width=cfg.data.max_ref_width,
    )
    if bucket_width and shuffle:
        # Group similar-width lines per batch -> ~54% padding-FLOPs reclaimed + stable shapes (lets
        # torch.compile stop recompiling). accelerate's prepared loader shards whole batches across ranks
        # (BatchSamplerShard); DDP all-reduces param-shaped grads, so per-rank differing widths are fine.
        from .data.sampler import WidthBucketBatchSampler, line_widths

        widths = line_widths(
            rows, max_width=cfg.data.max_line_width, cache_path=os.path.join(data_dir, f"{split}.widths.json")
        )
        batch_sampler = WidthBucketBatchSampler(
            widths, batch_size, shuffle=True, drop_last=True, seed=cfg.seed
        )
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Diffu (Stage-1 flow matching).")
    ap.add_argument(
        "--data-dir", default=None, help="dir with train.jsonl (from `make data`); or use --hf-dataset"
    )
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--save-every", type=int, default=10000, help="checkpoint every N steps (each ~17 GB)")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument(
        "--no-grad-checkpoint",
        action="store_true",
        help="disable grad checkpointing (~20%% faster, more VRAM)",
    )
    ap.add_argument("--compile", action="store_true")
    ap.add_argument(
        "--compile-mode",
        choices=("max-autotune-no-cudagraphs",),
        default=None,
        help="with --compile: autotune the GEMMs (+ diffusers' SD3 Inductor flags); slower first compile. "
        "CUDA-graph modes are inference-only (diffu-generate)",
    )
    ap.add_argument(
        "--compile-vae",
        action="store_true",
        help="torch.compile the VAE encode (the live x0 encode; moot with --cache-latents)",
    )
    ap.add_argument(
        "--no-repa",
        dest="repa",
        action="store_false",
        help="disable REPA representation-alignment aux loss (ON by default)",
    )
    ap.set_defaults(repa=True)
    ap.add_argument(
        "--no-fill-ratio",
        dest="fill_ratio",
        action="store_false",
        help="disable fill-ratio conditioning (ON by default)",
    )
    ap.set_defaults(fill_ratio=True)
    ap.add_argument(
        "--no-style-tokens",
        dest="style_in_context",
        action="store_false",
        help="inject style as the global pooled AdaLN vector ONLY, not as K attendable style tokens in "
        "joint attention (those let the model COPY the reference's glyphs and ignore the text). ON by default.",
    )
    ap.set_defaults(style_in_context=True)
    ap.add_argument("--track", action="store_true", help="log loss to trackio (live dashboard + NaN alerts)")
    ap.add_argument("--track-space", default=None, help="HF Space id to sync trackio to (e.g. user/diffu)")
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
        "--cond-dropout",
        type=float,
        default=0.0,
        help="drop ALL conditioning prob (full CFG: text+style); 0=off",
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
        "--style-canvas",
        action="store_true",
        help="in-context style: the reference line's clean latent sits left of the target on one canvas "
        "(no DINO style encoder); needs --glyph-line",
    )
    ap.add_argument(
        "--glyph-pad",
        action="store_true",
        help="place each line's glyph tokens over its own ink extent and pad the rest blank, instead of "
        "stretching them over the canvas; needs --glyph-line",
    )
    ap.add_argument(
        "--max-ref-width", type=int, default=None, help="--style-canvas: cap on the reference width (px)"
    )
    ap.add_argument(
        "--attention-backend",
        default=None,
        help="diffusers attention kernel, e.g. _native_cudnn / flash / flash_hub (default: torch SDPA). "
        "Needs a backward kernel for training (not sage*); benchmark on your GPU",
    )
    ap.add_argument(
        "--ink-focal", action="store_true", help="up-weight the loss on ink regions vs white paper"
    )
    ap.add_argument(
        "--init-from", default=None, help="warm-start model weights from a checkpoint (strict=False)"
    )
    ap.add_argument(
        "--resume",
        default=None,
        help="FULL resume from an accelerate step_N checkpoint dir (model+optimizer+LR+EMA+step). Unlike "
        "--init-from (weights only -> restarts optimizer/LR/step), this continues the run; use the SAME "
        "number of GPUs the checkpoint was saved with.",
    )
    ap.add_argument(
        "--profile-steps",
        type=int,
        default=0,
        help="profile N real training steps to out-dir/profile_step_table.txt, then exit (0=off)",
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
        "--cache-latents",
        action="store_true",
        help="skip the frozen VAE encode every step by feeding precomputed latents as x0 (the top remaining "
        "perf win, ~23%% of the step; docs/PERF_AUDIT.md). Builds the cache once if missing, then reads it. "
        "Requires --white-pad-prob 0 (augmentation invalidates the cache). Stores the posterior MODE, not a "
        "per-step sample -> opt-in; validate gen_CER before treating as default (like --bucket-width).",
    )
    ap.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="max characters per line (glyph content length cap; default from config, currently 128)",
    )
    ap.add_argument(
        "--glyph-line",
        action="store_true",
        help="line-level glyph content: whole-line image -> w_t column-aligned tokens + shared-column "
        "RoPE (MSRoPE), instead of per-char tokens. Smoke-validated; opt-in.",
    )
    ap.add_argument(
        "--glyph-concat",
        action="store_true",
        help="CHANNEL-CONCAT a spatial glyph latent onto the noisy latent (content in the input, not "
        "just attended — strongest coupling). Pair with --cond-dropout 0.1. EXPERIMENTAL.",
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
    ap.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (scaling; effective scale = alpha/r)")
    ap.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout")
    # --- Train straight off a Hugging Face dataset (image + text) instead of a local manifest.
    ap.add_argument(
        "--hf-dataset",
        default=None,
        help="train off a Hub datasets image+text set (e.g. Teklia/IAM-line) instead of --data-dir; "
        "needs `uv sync --extra hf`",
    )
    ap.add_argument("--hf-split", default="train", help="HF dataset split to train on")
    ap.add_argument("--hf-val-split", default="validation", help="HF split for the held-out val loader ('' to skip)")
    ap.add_argument("--hf-max-rows", type=int, default=None, help="cap the HF train split to N rows (smoke/dev)")
    ap.add_argument(
        "--htr-recognizer-eval",
        default=None,
        help="HF TrOCR id for the live gen_CER legibility gauge (default: the Swedish Lion). Pass "
        "microsoft/trocr-base-handwritten for an English corpus like IAM, else the gauge reads gibberish.",
    )
    args = ap.parse_args()

    cfg = Config()
    cfg.aux.repa = args.repa
    cfg.backbone.fill_ratio_cond = args.fill_ratio
    cfg.aux.diacritic_focal_flow = args.ink_focal
    if args.htr_recognizer_eval:  # override the default (Swedish) gauge recognizer — e.g. English for IAM
        cfg.aux.htr_recognizer_eval = args.htr_recognizer_eval
    cfg.cond.glyph_line = args.glyph_line
    cfg.cond.glyph_concat = args.glyph_concat
    cfg.cond.style_in_context = args.style_in_context
    cfg.cond.glyph_pad = args.glyph_pad
    if args.style_canvas:
        cfg.cond.style_mode = "canvas"
    if args.max_ref_width is not None:
        cfg.data.max_ref_width = args.max_ref_width
    if args.max_chars is not None:
        cfg.cond.max_chars = args.max_chars
    if args.logit_normal_mean is not None:
        cfg.flow.logit_normal_mean = args.logit_normal_mean
    cfg.data.white_pad_prob = args.white_pad_prob  # record-keeping; only the train loader uses it
    # White-pad augmentation is TRAIN-ONLY: the val loader keeps white_pad_prob=0 so val canvases stay
    # text-tight and the overfitting / gen_CER metrics are honest.
    dataloader: DataLoader
    val_loader: DataLoader | None
    if args.hf_dataset:  # train straight off a Hub image+text dataset (e.g. Teklia/IAM-line)
        from .data.hf_dataset import HFLineDataset

        if args.cache_latents:
            raise SystemExit("--cache-latents needs a --data-dir manifest; not supported with --hf-dataset")
        collate = partial(
            collate_lines,
            max_width=cfg.data.max_line_width,
            white_pad_prob=args.white_pad_prob,
            white_pad_max_frac=cfg.data.white_pad_max_frac,
            max_ref_width=cfg.data.max_ref_width,
        )
        train_ds = HFLineDataset(
            args.hf_dataset,
            split=args.hf_split,
            height=cfg.data.line_height,
            max_width=cfg.data.max_line_width,
            max_rows=args.hf_max_rows,
            ref_lines=cfg.cond.style_mode == "canvas",
        )
        dataloader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        val_loader = None
        if args.hf_val_split:  # a small held-out val loader for the overfitting / sample gauge
            try:
                val_ds = HFLineDataset(
                    args.hf_dataset,
                    split=args.hf_val_split,
                    height=cfg.data.line_height,
                    max_width=cfg.data.max_line_width,
                    max_rows=64,
                    ref_lines=cfg.cond.style_mode == "canvas",
                )
                val_loader = DataLoader(
                    val_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=partial(
                        collate_lines, max_width=cfg.data.max_line_width, max_ref_width=cfg.data.max_ref_width
                    ),
                    num_workers=args.num_workers,
                    drop_last=False,
                )
            except Exception as exc:  # noqa: BLE001 - a missing/renamed val split must not kill training
                print(f"(no HF val split {args.hf_val_split!r}: {exc})", flush=True)
    else:
        if not args.data_dir:
            raise SystemExit("provide --data-dir (a `make data` manifest) or --hf-dataset (a Hub dataset)")
        # --cache-latents: open (or one-time build) the train-split VAE latent cache. Multi-GPU: only the main
        # process builds; the others wait at the barrier, then every rank opens the same memmap read-only. The
        # cache is keyed to the encoder/geometry, so a stale one refuses to load. Val stays on live encode.
        train_latent_cache: LatentCache | None = None
        if args.cache_latents:
            from accelerate import PartialState

            from .data.latent_cache import build_latent_cache, open_cache

            if args.white_pad_prob > 0.0:
                raise SystemExit("--cache-latents needs --white-pad-prob 0 (the augmentation invalidates the cache)")
            state = PartialState()
            try:
                train_latent_cache = open_cache(cfg, args.data_dir, "train")
            except (FileNotFoundError, ValueError):
                if state.is_main_process:
                    print("--cache-latents: building the VAE latent cache (train split, one-time)…", flush=True)
                    build_latent_cache(cfg, args.data_dir, "train")
                state.wait_for_everyone()  # non-main ranks wait for the build before opening
                train_latent_cache = open_cache(cfg, args.data_dir, "train")
            if state.is_main_process:
                print(
                    f"--cache-latents: feeding {len(train_latent_cache)} cached latents as x0 (no per-step encode)",
                    flush=True,
                )
        dataloader = build_dataloader(
            cfg,
            args.data_dir,
            args.batch_size,
            args.num_workers,
            white_pad_prob=args.white_pad_prob,
            bucket_width=args.bucket_width,
            latent_cache=train_latent_cache,
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
            "lora": args.lora,
            "lora_rank": args.lora_rank,
            "hf_dataset": args.hf_dataset,
        },
    )


if __name__ == "__main__":
    main()
