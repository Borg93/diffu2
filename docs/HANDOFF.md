# Handoff — continuing on the DGX Spark (GB10), 2026-10-19

State of `main` at handoff and what to run next on a local GPU. Nothing below has been run on a GPU yet:
the cloud session had CPU only.

## 1. What changed since the original diffu (`2b7ab3f`)

| Change | Status |
|---|---|
| `--style-canvas`: reference line left of the target on one canvas (extra input channels), its transcription in the glyph tokens, loss on target columns only, `--ref-dropout` + `--style-cfg-scale` | Implemented + unit-tested. **Untrained** — style quality unproven. |
| `--glyph-pad`: glyph tokens cover only the real ink width | Implemented + tested. Alignment error 212 → 4 px (measured). Quality untrained. |
| diffusers 0.41 / accelerate 1.15 / torch 2.14.1 / transformers 5.20.1 / peft 0.21.2 | Done. |
| Attention through `dispatch_attention_fn` (`--attention-backend`) | Done. Backends not benchmarked on GPU. |
| Fused QKV (inference), one QKV matmul under compile (training), bf16 Q/K after qk-norm | Done, outputs identical. |
| VAE single-frame fast path (3× less conv work, same output to 1e-6) | Done, measured on CPU. |
| Compile modes + SD3 Inductor flags + `--compile-vae` | Done. Graph-break check: 0 breaks, 3 expected compiles, no recompiles across widths. |
| Losses read only at log steps (no per-step host sync) | Done. |
| DINOv2 fallback removed; FLUX.2 backbone experiment reverted | Done. |

Reports (private artifacts, share from the page): before/after measurements
<https://claude.ai/artifact/XAWWbmTauyQAkW84cDMRkf>, architecture review vs Qwen-Image 2.1 / FLUX.3
<https://claude.ai/artifact/6jZok7ZFYpTrPCatm3uLfM>.

## 2. Setup on the GB10

The GB10 is aarch64 with one Blackwell GPU (sm_121). torch comes from the cu130 index in `pyproject.toml`.

```bash
git clone https://github.com/Borg93/diffu2.git && cd diffu2
make install-dev
uv run python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
# expect 2.14.1+cu130 ... (12, 1)
hf auth login            # DINOv3 (REPA target / dino mode) is gated
```

**The Makefile defaults to GPU 1 (`GPUS_1 ?= 1`, from the old 3-GPU box). The GB10 has GPU 0 only — pass
`GPUS_1=0` to every make target**, or use `uv run` directly.

## 3. Run in this order

1. Tests: `make test`.
2. Smoke: `make smoke-1gpu GPUS_1=0`, then `make smoke-compile GPUS_1=0`.
3. Attention backends (picks the default to use):
   ```bash
   uv run python scripts/bench_attention.py --canvas --compile --width 1024
   ```
   Use the fastest backend that reports correct; Sage backends are inference-only (no backward).
4. Profile a few real steps and keep the trace (for the `profile-model` skill):
   ```bash
   uv run diffu-train --data-dir data_out --glyph-line --glyph-pad --style-canvas \
       --compile --profile-steps 20 --out-dir runs/profile
   # -> runs/profile/profile_step_table.txt, runs/profile/trace.json
   ```
5. VAE gate if the data changed: `make vae-gate GPUS_1=0 OUT_DIR=data_out`.
6. The retrain:
   ```bash
   uv run diffu-train --data-dir data_out --out-dir runs/canvas --glyph-line --glyph-pad --style-canvas \
       --text-dropout 0.1 --ref-dropout 0.2 --cond-dropout 0.05 --compile --compile-mode max-autotune-no-cudagraphs \
       --attention-backend <winner from step 3>
   ```
7. Sample:
   ```bash
   uv run diffu-generate --ckpt <step_dir>/ema_model.safetensors --glyph-pad --style-canvas --line-height 128 \
       --style ref_line.png --style-text "what the ref line says" --text "Göteborgs poliskammare" \
       --cfg-scale 5 --style-cfg-scale 2 --compile --compile-vae
   ```

PyTorch agent skills used in the review: `npx skills add meta-pytorch/skills` (profile-model,
debug-graph-breaks, analyze-memory-snapshot).

## 4. Open risks

- **`writer_id` is the page id.** The reference shares the target's paper, so the model can learn to copy the
  paper instead of the hand. Lines from other pages of the same writer, or writer-labelled data, fix this.
- **Copying the reference's letters** instead of writing the target text. Check with a swap test: same
  reference, different texts → the text must change; same text, different references → the hand must change.
- **No style metric yet.** Without one, style can't be compared between runs.

## 5. Why the backbone has no shared modulation

`SD3Transformer2DModel` hard-codes an `AdaLayerNormZero` per block and per stream (DiT's adaLN-Zero, which SD3
kept). We use the diffusers class as-is, so we inherited it: 24 blocks × 2 streams × (1024 → 6·1024) ≈ 300 M
of the 900 M parameters, all mapping the same timestep + fill-ratio vector. It is not configurable in that
class. Newer models compute modulation once and share it (FLUX.2, Qwen-Image 2.1, FLUX.3: ~1–3 % of params).
At our width and token count, a `Flux3Transformer2DModel` stack with the same FLOPs (821 vs 825 GFLOP per
forward) has 378 M parameters — measured; whether it writes better is untested.

## 6. Next decisions (not implemented)

No retrain needed:
- Sampling shift sweep: `FlowMatchEulerDiscreteScheduler()` in `model/diffu.py` uses shift 1. Try 1.5 / 2 / 3
  on CER (SD3's `calculate_shift` gives ≈1.9 for our 1,024 tokens).
- `UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True)` at 12–16 steps vs Euler 24.
- CFG-Zero* / APG guidance maths (diffusers guiders), computed on the target columns only.
- Skip diffusers' cache hooks: FirstBlockCache crashes on SD3's last block; MagCache / TaylorSeer break our
  `fullgraph=True` regional compile.

With a retrain:
- Reference as clean tokens at timestep 0 (FLUX.3 / Qwen-Edit) instead of noised canvas columns.
- Equal-FLOPs A/B: SD3 24 dual blocks vs `Flux3Transformer2DModel` (2 per-mode + 20 joint, width 1024),
  same data, judged on held-out-writer CER + a style metric. `Flux3Transformer2DModel` is in diffusers main
  only until 0.42 ships.
- The external baseline worth beating: FLUX.3 Klein base + LoRA (diffusers ships the DreamBooth scripts) on
  the same lines, judged the same way.
