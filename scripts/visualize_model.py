"""Generate REAL, shape-annotated model-graph visualizations of Diffu's architecture.

Unlike hand-drawn diagrams, these are traced from the actual `nn.Module`s by running a forward
pass and recording it — so they cannot drift from the code. Every node carries the real module
name and its input/output **tensor shapes**.

  - torchview  -> static PNGs with tensor shapes on every node (the informative ones):
                    * MMDiT backbone spine (PatchEmbed -> JointTransformerBlock xN -> unpatchify)
                    * glyph-line content CNN (ResNet18 spatial trunk)
                    * glyph-latent conv block
  - torchvista -> an interactive HTML graph of the same MMDiT forward (zoom / click-to-expand).

Everything runs on CPU with tiny dummy inputs and `pretrained=False` (topology identical to the
trained model; only weight *values* differ) so NO weights download and NO GPU is touched. The MMDiT
graph uses a reduced block count for legibility (torchview rolls them to "xN"); dim/heads/patch/RoPE
are the real values.

Requirements:
  uv pip install -e ".[viz]"            # torchview + torchvista (+ visualtorch, unused here)
  graphviz `dot` on PATH                # torchview renders via graphviz; e.g. `apt-get install graphviz`

Run:  python scripts/visualize_model.py
Out:  docs/mmdit_graph.png          (backbone spine, shapes)
      docs/content_cnn_graph.png    (glyph-line ResNet18 trunk, shapes)
      docs/glyph_latent_graph.png   (glyph-latent conv block, shapes)
      docs/model_graph_mmdit.html   (interactive torchvista)
"""

from __future__ import annotations

import shutil
from pathlib import Path

import torch

from diffu.config import Config
from diffu.model.backbone import Backbone
from diffu.model.content_encoder import GlyphLatentEncoder, GlyphLineCNN
from diffu.model.rope import build_2d_rope

DOCS = Path(__file__).resolve().parent.parent / "docs"
VIZ_LAYERS = 2  # blocks built for the MMDiT graph; torchview rolls them to "xN" (real model = 24)


def _render(visual_graph, stem: str) -> None:
    visual_graph.format = "png"
    visual_graph.render(str(DOCS / stem), cleanup=True)
    print(f"[ok] {DOCS / stem}.png  ({(DOCS / (stem + '.png')).stat().st_size // 1024} KB)")


def _backbone() -> tuple[Backbone, dict, int]:
    """A real-dims Backbone (fewer blocks) + the kwargs for tracing its inner SD3 transformer + ctx dim."""
    cfg = Config()
    b = cfg.backbone.model_copy(update={"num_layers": VIZ_LAYERS})  # real dim/heads/patch/rope
    ctx = b.context_dim
    mm = Backbone(b, in_channels=cfg.vae.latent_channels, context_dim=ctx, pooled_dim=ctx,
                  out_channels=cfg.vae.latent_channels).eval()
    dev = torch.device("cpu")
    hd = b.dim // b.heads
    # Latent [1,16,8,32] -> patch 2 -> 4x16 = 64 image tokens; 16 content tokens (style pooled-only).
    cos, sin = build_2d_rope(4, 16, hd, dev, b.rope_theta)       # image 2D-RoPE freqs
    ccos, csin = build_2d_rope(1, 16, hd, dev, b.rope_theta)     # shared-column content RoPE
    kw = dict(
        hidden_states=torch.randn(1, cfg.vae.latent_channels, 8, 32),
        timestep=torch.tensor([500.0]),
        encoder_hidden_states=torch.randn(1, 16, ctx),
        pooled_projections=torch.randn(1, ctx),
        joint_attention_kwargs={"image_rotary_emb": (cos, sin),
                                "content_rotary_emb": (ccos, csin), "n_content": 16},
        return_dict=False,
    )
    return mm, kw, ctx


def mmdit_static() -> None:
    """torchview spine of the SD3 MMDiT backbone — shapes on every node. RoPE freqs precomputed so
    the joint-attention build doesn't clutter the graph; depth 1 keeps the block as one rolled node."""
    from torchview import draw_graph

    mm, kw, _ = _backbone()
    g = draw_graph(mm.transformer, input_data=kw, depth=1, expand_nested=True, roll=True,
                   graph_name="MMDiT")
    _render(g.visual_graph, "mmdit_graph")


def content_cnn_static() -> None:
    """torchview of the glyph-LINE content encoder (ResNet18 spatial trunk) — the down-sampling funnel
    with real shapes; no avgpool/fc so the width survives -> one content token per latent column."""
    from torchview import draw_graph

    cnn = GlyphLineCNN(out_dim=1024, pretrained=False).eval()  # [N,1,H,W] -> [N,1024,H/32,W/32]
    g = draw_graph(cnn, input_data=torch.randn(1, 1, 64, 512), depth=2, expand_nested=True,
                   roll=True, graph_name="GlyphLineCNN")
    _render(g.visual_graph, "content_cnn_graph")


def glyph_latent_static() -> None:
    """torchview of the glyph-latent conv block (channel-concat path) — 3x (Conv2d stride2 -> SiLU)."""
    from torchview import draw_graph

    cfg = Config()
    enc = GlyphLatentEncoder(cfg.cond, out_ch=cfg.cond.glyph_concat_channels).eval()
    g = draw_graph(enc.net, input_data=torch.randn(1, 1, 32, 512), depth=1, graph_name="GlyphLatent")
    _render(g.visual_graph, "glyph_latent_graph")


def mmdit_interactive() -> None:
    """torchvista interactive HTML of the same MMDiT forward (zoom / click any node to expand)."""
    from torchvista import trace_model

    mm, _, ctx = _backbone()
    # Full Backbone.forward (builds RoPE inline) so the interactive graph also shows the RoPE feed-in.
    inputs = (torch.randn(1, 16, 8, 32), torch.tensor([500.0]), torch.randn(1, 16, ctx),
              torch.randn(1, ctx), 16)
    out = DOCS / "model_graph_mmdit.html"
    with torch.no_grad():
        trace_model(mm, inputs, show_non_gradient_nodes=False, collapse_modules_after_depth=1,
                    height=900, export_path=str(out))
    print(f"[ok] {out}  ({out.stat().st_size // 1024} KB, interactive)")


if __name__ == "__main__":
    torch.manual_seed(0)
    if shutil.which("dot") is None:
        print("[warn] graphviz `dot` not on PATH — the torchview PNGs will fail. "
              "Install graphviz (e.g. `apt-get install graphviz` / `brew install graphviz`).")
    for name, fn in (("mmdit-static", mmdit_static), ("content-cnn", content_cnn_static),
                     ("glyph-latent", glyph_latent_static), ("mmdit-interactive", mmdit_interactive)):
        try:
            fn()
        except Exception as e:  # one failure shouldn't sink the rest
            import traceback
            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()
