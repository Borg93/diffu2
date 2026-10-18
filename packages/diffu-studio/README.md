# diffu-studio

A **`gradio.Server`** control panel for [Diffu](../../README.md) handwriting synthesis — single-line
inference and whole-page document generation in one app, driving `diffu` (the line model) and
`diffu-page` (layout / compositing / augmentation / labels).

## What it does

- **Single line** — type text, pick a writer style, watch the model denoise *noise → handwriting* live,
  with a TrOCR read-back + CER.
- **Whole page** — generate a full document page, streamed **line by line** (each line denoises, then
  seats; the page grows top→bottom; then the scan/aging pass), with PAGE / ALTO / JSON ground-truth
  labels emitted alongside.
- **Control panel** — every layout (`GenConfig`/`PageSpec`), augmentation (`AugmentConfig`) and sampling
  knob is a control; a seed makes any result reproducible.
- **Server + API** — one `gradio.Server` (a FastAPI app): a mounted Blocks UI plus `@app.api` **SSE**
  endpoints (`stream_line`, `stream_page`, `list_checkpoints`, `list_templates`) so a custom frontend can
  drive the same core later. OpenAPI at `/docs`.

## Run

```bash
uv run diffu-studio --ckpt checkpoints/exp_sd35_128/step_120000 --line-height 128 \
    --gallery data_out_lejonet_128/manifest.jsonl
```

`--line-height 128` is the known footgun: the released `exp_sd35_128` checkpoint was trained at 128px,
but the flag defaults to 64 — leave it off and every generation is an illegible smudge.

Device is auto-detected (`zerogpu` on HF Spaces, local `gpu`, else `cpu`); override with
`--device` or `DIFFU_STUDIO_DEVICE`. On CPU the page build shows progress instead of a live animation.

## Layout

- `core.py` — the frontend-agnostic API (no gradio, no FastAPI); everything the UI and the server share.
- `ui.py` — the slim top-level Blocks assembly: a **Studio** tab (whole-page ⟷ single-line toggle) + **Guide**.
- `views/` — one module per screen region, each owning its widgets + event wiring: `line_view` (single-line
  mode + TrOCR read-back), `page_view` (whole-page mode, knobs split by what costs a re-generation),
  `style_view` (style-reference drop target + real-crop gallery), `guide_view` (the embedded
  `diffusion_intuition` explainer).
- `controls.py` — control-bags: map a group of live Gradio widgets to a validated form at the request boundary.
- `forms.py` — frozen Pydantic value-objects for the UI state, one validated form per generation mode.
- `handlers.py` — gradio-thin services behind the events: each takes one typed form, not positional knobs.
- `server.py` — the `gradio.Server` app: Blocks UI + streaming `@app.api` SSE endpoints (a future custom
  frontend's surface; `__main__` currently launches the queued Blocks directly).
- `__main__.py` — the `diffu-studio` CLI: sets the CUDA fragmentation guard + pins one GPU *before* any
  torch import, then launches.
- `engine.py` — the stateful engine: device resolved once, models cached by weights path (capped, evicts
  oldest with a CUDA cache flush).
- `device.py` — the device backend: `zerogpu` / `gpu` / `cpu` auto-detection and pipeline adaptation.
- `checkpoints.py` — checkpoint discovery + loading, with the architecture guard (never silently generate
  gibberish from a mismatched conditioner).
- `configs.py` — bridges UI/API controls to the real `diffu-page` `GenConfig`/`AugmentConfig` (no drifting copies).
- `line.py` — single-line generation: stream the denoise (noise → handwriting), then TrOCR read-back + CER.
- `page.py` — whole-page generation: line-by-line seating via `PageComposer`, scan/aging pass, GT labels.
- `steps.py` — the value objects streamed out of the core generators (one per animation frame / seated line).
- `optimization.py` — ahead-of-time (AoT) compilation of the MMDiT backbone for HF ZeroGPU (H200).
- `theme.py` — the warm archival Gradio theme (`diffusion_intuition` look).
- `upscale.py` — PiD 4× super-resolution, a thin re-export of the shared `diffu_page.upscale` core.
- `viz.py` — draw the GT labels back over a generated page (polygons, baselines, word boxes, å ä ö-safe text).
