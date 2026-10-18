"""Whole-page mode: compose many lines into a historical page, then age/damage/upscale it.

Knobs are split into two clearly-labelled sections so it is obvious what costs a re-generation:
**① Generation** (layout, sampling, content) applies only on *Generate*; **② Look & aging** (spacing, aging,
damage, binding, ink) updates the cached page *instantly* — no model, no GPU. Presence effects are **checkboxes**
(deterministic on/off for this page); only genuine-strength effects stay sliders. Everything is gathered into a
:class:`PageControls` bag so the handlers take one typed form.
"""

import gradio as gr
from diffu_page.layout import TEMPLATES

from diffu_studio.controls import AugmentControls, PageControls
from diffu_studio.device import gpu_task
from diffu_studio.engine import StudioEngine
from diffu_studio.handlers import recompose_view, reseat_view, stream_page_view, upscale_view

_INK_COLORS = ["", "#20232e", "#2a2320", "#1a2a3a", "#3a2a1a"]  # "" = the varied default palette
# Derived from the layout registry (same as config.Template) so every template — incl. letter/register/
# ledger/landskap/blank_leaf — is selectable, and adding one stays a single edit in layout.TEMPLATES.
_TEMPLATES = list(TEMPLATES)


def build_page_view(
    *, engine: StudioEngine, default_ckpt: str, ref_state: gr.Image, text: gr.Textbox
) -> None:
    """Build every whole-page widget + preview and wire Generate / re-composite / re-seat / upscale.

    ``ref_state`` and ``text`` are the shared inputs built once in the top card (both modes reuse them).
    """
    aug: dict[str, gr.components.Component] = {}  # scan-look widgets, keyed by AugmentForm field name

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):  # noqa: SIM117 — 50 % settings column; combining would dedent every tab
            # ── ① Generation — applies on Generate (re-renders the handwriting/layout) ──
            gr.Markdown("### ① Generation &nbsp;·&nbsp; set these, then press **✨ Generate** ↻")
            with gr.Tabs():
                with gr.Tab("📐 Layout"), gr.Group():
                    with gr.Row():
                        surprise = gr.Checkbox(
                            value=False,
                            label="🎲 Surprise me (corpus mix)",
                            info="Sample the whole layout in corpus proportions (prose/letter/register/"
                            "ledger/…). Template + layout knobs are IGNORED while on; reroll via Seed.",
                        )
                        spread_frac = gr.Slider(
                            0.0,
                            1.0,
                            value=0.35,
                            step=0.05,
                            label="Spread fraction",
                            info="Surprise only: chance a prose-like page comes out as a two-page opening.",
                        )
                    with gr.Row():
                        template = gr.Dropdown(
                            _TEMPLATES,
                            value="single_column",
                            label="Template",
                            info="All registry templates — prose (single/multi column), tables (semi_table/"
                            "register/ledger/landskap), key_value, letter, spread, blank_leaf.",
                        )
                        line_height = gr.Slider(
                            24,
                            140,
                            value=100,
                            step=2,
                            label="Line height",
                            info="Rendered height of one line — GT scale ≈100px.",
                        )
                    with gr.Row():
                        n_columns = gr.Slider(
                            1,
                            4,
                            value=2,
                            step=1,
                            label="Columns",
                            info="Text columns (multi_column / columns templates).",
                        )
                        width = gr.Slider(
                            600, 2480, value=1240, step=20, label="Width", info="Page width (px)."
                        )
                    height = gr.Slider(
                        800, 3508, value=1754, step=20, label="Height", info="Page height (px; A4≈1240×1754)."
                    )
                    with gr.Accordion("📄 ALTO/PAGE XML layout (optional)", open=False):
                        xml_file = gr.File(
                            label="Upload an XML layout to replay a real page's exact line boxes",
                            file_types=[".xml"],
                            type="filepath",
                        )
                with gr.Tab("🎛️ Sampling"), gr.Group():
                    with gr.Row():
                        steps = gr.Slider(
                            4,
                            50,
                            value=24,
                            step=1,
                            label="Steps",
                            info="Denoise iterations per line — higher is smoother but slower.",
                        )
                        cfg = gr.Slider(
                            0.0,
                            12.0,
                            value=5.0,
                            step=0.5,
                            label="CFG",
                            info="Text-following strength; ~5 is the legible sweet spot.",
                        )
                    with gr.Row():
                        seed = gr.Number(
                            value=0,
                            label="Seed",
                            precision=0,
                            info="Re-rolls WHERE aging/damage lands. 🎲 Reroll updates instantly.",
                        )
                        reroll = gr.Button("🎲 Reroll look", variant="secondary", size="sm")
                    formats = gr.CheckboxGroup(
                        ["page", "alto", "json"],
                        value=["page", "alto", "json"],
                        label="Label formats",
                        info="Which ground-truth label formats to emit.",
                    )
                    with gr.Row():
                        lora = gr.Textbox(
                            value="",
                            label="LoRA adapter",
                            placeholder="runs/iam_lora/final",
                            info="Optional Stage-3 adapter (dir or Hub id) applied on top of the checkpoint.",
                        )
                        lora_scale = gr.Slider(
                            0.0,
                            1.5,
                            value=1.0,
                            step=0.05,
                            label="LoRA scale",
                            info="Adapter strength — 0 = base checkpoint, 1 = full.",
                        )
                with gr.Tab("📝 Content"), gr.Group():
                    with gr.Row():
                        furniture_prob = gr.Checkbox(
                            value=False, label="Tables/furniture", info="Add ruled ledger tables."
                        )
                        typeset_furniture = gr.Checkbox(
                            value=False, label="Typeset numbers", info="Printed type, not handwritten."
                        )
                        binding_prob = gr.Checkbox(
                            value=False, label="Book binding", info="Spine + gutter (applies on Generate)."
                        )
                    with gr.Row():
                        insertion_prob = gr.Slider(
                            0.0,
                            0.2,
                            value=0.0,
                            step=0.005,
                            label="Insertions/line",
                            info="Per-line rate of an inserted (interlinear) word.",
                        )
                        deletion_prob = gr.Slider(
                            0.0,
                            0.2,
                            value=0.0,
                            step=0.005,
                            label="Deletions/line",
                            info="Per-line rate of a struck-through deletion.",
                        )
                    page_number = gr.Textbox(
                        label="Page number", value="", info="Force a page number (blank = none)."
                    )
                with gr.Tab("🧩 Layout detail"), gr.Group():
                    with gr.Row():
                        header_scale = gr.Slider(
                            1.0,
                            2.0,
                            value=1.6,
                            step=0.05,
                            label="Header scale",
                            info="Header size relative to the body (never smaller).",
                        )
                        marg_scale = gr.Slider(
                            0.5,
                            1.0,
                            value=0.95,
                            step=0.05,
                            label="Marginalia scale",
                            info="Marginalia size relative to the body (never bigger).",
                        )
                    with gr.Row():
                        margin_ratio = gr.Slider(
                            0.02,
                            0.12,
                            value=0.06,
                            step=0.005,
                            label="Page margin",
                            info="Margin as a fraction of the smaller page side.",
                        )
                        px_per_char = gr.Slider(
                            10.0,
                            20.0,
                            value=15.0,
                            step=0.5,
                            label="Px per char",
                            info="Character-width estimate used to size line slots / wrapping.",
                        )
                    with gr.Row():
                        drop_cap = gr.Checkbox(
                            value=False, label="Drop cap", info="Enlarge the opening initial into a versal."
                        )
                        catchword = gr.Textbox(
                            label="Catchword",
                            value="",
                            info="Next leaf's first word, bottom-outer corner (blank = off).",
                        )
                    with gr.Row():
                        section_headings = gr.Textbox(
                            label="Section headings",
                            value="",
                            info="Comma-separated rubrics inserted between paragraphs (blank = off).",
                        )
                        margin_keys = gr.Textbox(
                            label="Margin keys",
                            value="",
                            info="Comma-separated keyed margin notes, one per paragraph (blank = off).",
                        )
                    with gr.Row():
                        enum_prob = gr.Slider(
                            0.0,
                            1.0,
                            value=0.0,
                            step=0.05,
                            label="Enumerators",
                            info="Per-paragraph chance of an article-enumerator prefix.",
                        )
                        enum_marker = gr.Dropdown(
                            ["§", "N:o"], value="§", label="Enum marker", info="The enumerator glyph."
                        )
                    underline_prob = gr.Slider(
                        0.0,
                        1.0,
                        value=0.0,
                        step=0.05,
                        label="Underlines",
                        info="Per-body-line chance of an underline (names/dates) — labelled GT.",
                    )
                    with gr.Row():
                        table_rows = gr.Slider(
                            6,
                            30,
                            value=14,
                            step=1,
                            label="Table rows",
                            info="Rows for the table templates (semi_table / register / ledger).",
                        )
                        table_cols = gr.Slider(
                            2,
                            8,
                            value=3,
                            step=1,
                            label="Table columns",
                            info="Columns for the table templates.",
                        )
                    kv_label_frac = gr.Slider(
                        0.2,
                        0.5,
                        value=0.34,
                        step=0.01,
                        label="KV label width",
                        info="key_value template: label-column width as a fraction of the body.",
                    )

            # ── ② Look & aging — updates the cached page instantly (no model, no re-generation) ──
            gr.Markdown(
                "### ② Look &amp; aging &nbsp;·&nbsp; updates **instantly** 🎨 &nbsp;·&nbsp; "
                "checkboxes = effect on/off · sliders = strength · spacing re-seats ⚡"
            )
            with gr.Tabs():
                with gr.Tab("↔️ Spacing ⚡"), gr.Group(), gr.Row():
                    line_gap = gr.Slider(
                        -0.5,
                        1.0,
                        value=0.18,
                        step=0.02,
                        label="Line gap",
                        info="Vertical gap between lines (fraction of line height). Negative = lines overlap "
                        "— ascenders/descenders interleave like a real manuscript. Re-seats instantly.",
                    )
                    para_gap = gr.Slider(
                        0.0,
                        2.0,
                        value=0.4,
                        step=0.05,
                        label="Paragraph gap",
                        info="Extra gap between paragraphs. Re-seats instantly.",
                    )
                with gr.Tab("🟤 Aging"), gr.Group():
                    with gr.Row():
                        aug["rotate_max"] = gr.Slider(
                            0.0,
                            6.0,
                            value=1.0,
                            step=0.5,
                            label="Rotation °",
                            info="Max page rotation, degrees.",
                        )
                        aug["wash_max"] = gr.Slider(
                            0.0, 1.0, value=0.2, step=0.05, label="Damp wash", info="Max damp-wash staining."
                        )
                    with gr.Row():
                        aug["ink_fade_max"] = gr.Slider(
                            0.0,
                            0.5,
                            value=0.12,
                            step=0.02,
                            label="Ink fade",
                            info="Max ink fade (thinner strokes).",
                        )
                        aug["vignette_max"] = gr.Slider(
                            0.0,
                            0.6,
                            value=0.42,
                            step=0.02,
                            label="Platen vignette",
                            info="Max edge darkening.",
                        )
                    with gr.Row():
                        aug["bleed_max"] = gr.Slider(
                            0.0,
                            0.4,
                            value=0.13,
                            step=0.02,
                            label="Verso bleed-through",
                            info="Ink from the back page.",
                        )
                        aug["skew"] = gr.Slider(
                            0.0,
                            0.03,
                            value=0.008,
                            step=0.001,
                            label="Perspective skew",
                            info="Max keystone skew.",
                        )
                    with gr.Row():
                        aug["microfilm_prob"] = gr.Checkbox(
                            value=False, label="Microfilm", info="Render as grayscale microfilm."
                        )
                        aug["foxing_tint"] = gr.ColorPicker(
                            value="#ffc773", label="Foxing tint", info="Hue of the foxing / age stains."
                        )
                with gr.Tab("💥 Damage"), gr.Group():
                    aug["edge_damage_max"] = gr.Slider(
                        0.0,
                        0.02,
                        value=0.007,
                        step=0.001,
                        label="Edge fray",
                        info="Max edge fraying / tearing.",
                    )
                    with gr.Row():
                        aug["bed_prob"] = gr.Checkbox(
                            value=False, label="Scanner-bed", info="Dark scanner-bed border."
                        )
                        aug["clasp_prob"] = gr.Checkbox(
                            value=True, label="Page-holder clasp", info="Clasp over an edge."
                        )
                    with gr.Row():
                        aug["fold_prob"] = gr.Checkbox(
                            value=True, label="Fold / tide-line", info="Crease or water tide-line."
                        )
                        aug["dust_prob"] = gr.Checkbox(
                            value=True, label="Dust + scratches", info="Dust specks + scratches."
                        )
                    with gr.Row():
                        aug["glove_prob"] = gr.Checkbox(
                            value=True, label="Gloved fingertip", info="Fingertip at an edge."
                        )
                        aug["hole_prob"] = gr.Checkbox(
                            value=False, label="Worm / punch holes", info="Holes in the margin."
                        )
                    aug["curl_prob"] = gr.Checkbox(
                        value=True, label="Curled corner", info="Lifted/curled page corner."
                    )
                with gr.Tab("📚 Binding"), gr.Group():
                    with gr.Row():
                        aug["spine_shadow_max"] = gr.Slider(
                            0.0,
                            1.0,
                            value=0.72,
                            step=0.05,
                            label="Spine shadow",
                            info="Max shadow near the gutter.",
                        )
                        aug["spine_warp_max"] = gr.Slider(
                            0.0,
                            0.3,
                            value=0.15,
                            step=0.01,
                            label="Spine warp",
                            info="Max page warp toward the spine.",
                        )
                    aug["gutter_blur_max"] = gr.Slider(
                        0.0, 6.0, value=3.0, step=0.2, label="Gutter blur", info="Max blur in the gutter."
                    )
                    gr.Markdown(
                        "<small>Binding shape needs the **Book binding** checkbox (Content tab) + Generate; "
                        "these tune its look.</small>"
                    )
                with gr.Tab("🖋️ Ink & marks"), gr.Group():
                    aug["ink_color"] = gr.Dropdown(
                        _INK_COLORS, value="", label="Ink colour", info="Ink colour (blank = varied default)."
                    )
                    with gr.Row():
                        aug["correction_prob"] = gr.Checkbox(
                            value=True, label="Ink corrections", info="Struck-through fixes."
                        )
                        aug["stamp_prob"] = gr.Checkbox(
                            value=True, label="Stamp / seal", info="Archival stamp or seal."
                        )
                    with gr.Row():
                        aug["paraph_prob"] = gr.Checkbox(
                            value=False, label="Scribal paraph", info="A looping flourish."
                        )
                        aug["margin_mark_prob"] = gr.Checkbox(
                            value=False, label="Margin marks", info="Reference marks in the margin."
                        )

            # ── Generate — at the BOTTOM of the settings column, below every knob (incl. Spacing) ──
            with gr.Row():
                with gr.Column(scale=0, min_width=180):
                    go = gr.Button("✨ Generate", variant="primary")  # modest button, not a full-width bar
                with gr.Column(scale=5):
                    status = gr.Markdown()
        with gr.Column(scale=1):  # 50 % — preview
            with gr.Tabs():
                with gr.Tab("Page"):
                    page_img = gr.Image(height=560, show_label=False)
                    # the line currently denoising — only meaningful on the Page tab, while generating
                    active_img = gr.Image(label="Current line (while generating)", height=88)
                with gr.Tab("Labels overlay"):
                    overlay_img = gr.Image(height=560, show_label=False)
                    labels_file = gr.File(label="GT labels (PAGE / ALTO / JSON)", file_count="multiple")
                with gr.Tab("🔍 Upscale 4×"):
                    gr.Markdown("4× super-resolution of the current page with PiD (GPU Space only).")
                    upscaled = gr.ImageSlider(label="Original ↔ 4× (PiD)", type="pil", height=520)
                    upscale_status = gr.Markdown()
                    with gr.Row(), gr.Column(scale=0, min_width=200):  # button at the bottom, like Generate
                        upscale_btn = gr.Button("🔍 Upscale 4×", variant="primary")
            render_state = gr.State()  # model inks + layout from Generate — reused by the instant paths

    augment = AugmentControls(aug)
    controls = PageControls(
        {
            "text": text,
            "template": template,
            "n_columns": n_columns,
            "steps": steps,
            "cfg_scale": cfg,
            "seed": seed,
            "formats": formats,
            "width": width,
            "height": height,
            "line_height": line_height,
            "line_gap": line_gap,
            "para_gap": para_gap,
            "xml_layout": xml_file,
            "furniture_prob": furniture_prob,
            "typeset_furniture": typeset_furniture,
            "insertion_prob": insertion_prob,
            "deletion_prob": deletion_prob,
            "page_number": page_number,
            "binding_prob": binding_prob,
            "header_scale": header_scale,
            "marg_scale": marg_scale,
            "margin_ratio": margin_ratio,
            "px_per_char": px_per_char,
            "drop_cap": drop_cap,
            "catchword": catchword,
            "section_headings": section_headings,
            "margin_keys": margin_keys,
            "enum_prob": enum_prob,
            "enum_marker": enum_marker,
            "underline_prob": underline_prob,
            "table_rows": table_rows,
            "table_cols": table_cols,
            "kv_label_frac": kv_label_frac,
            "surprise": surprise,
            "spread_frac": spread_frac,
        },
        augment,
    )

    # "Surprise me" overrides the hand-set layout — grey out the template dropdown so that's visible (the
    # other layout knobs are documented as ignored in the checkbox label). Cheap in-process toggle, no model.
    surprise.change(
        lambda on: gr.update(interactive=not on),
        inputs=[surprise],
        outputs=[template],
        show_progress="hidden",
    )

    # ZeroGPU fork-safety: a @spaces.GPU call PICKLES its args to reach the worker fork, so GPU-wrapped
    # handlers must receive plain VALUES in a fixed order — a {component: value} dict would pickle the
    # component objects (and with them the whole Blocks graph, incl. local closures → crash). The ordered
    # list is zipped back into the dict the controls expect INSIDE the handler (components arrive via fork
    # memory, not pickle). Non-GPU handlers (recompose/reseat/reroll) run in-process and keep dict inputs.
    gen_inputs: list = [ref_state, lora, lora_scale, *controls.components()]

    def on_generate(*vals):
        data = dict(zip(gen_inputs, vals, strict=True))
        yield from stream_page_view(
            engine,
            controls.read(data),
            ref=data[ref_state],
            ckpt=default_ckpt,
            lora=(data[lora] or "").strip() or None,
            lora_scale=float(data[lora_scale]),
        )

    def on_recompose(data):
        return recompose_view(data[render_state], augment.read(data), data[formats], seed=int(data[seed]))

    def on_reseat(data):
        return reseat_view(
            data[render_state], augment.read(data), data[line_gap], data[para_gap], seed=int(data[seed])
        )

    def on_reroll(data):
        import random  # noqa: PLC0415 — local so a fresh look each click; module-level RNG would be shared state

        fresh = random.randint(0, 999_999)
        return [fresh, *recompose_view(data[render_state], augment.read(data), data[formats], seed=fresh)]

    go.click(
        gpu_task(on_generate),
        inputs=gen_inputs,
        outputs=[page_img, active_img, overlay_img, status, labels_file, render_state],
    )
    # ② Look knobs re-composite the CACHED inks (no model): strength SLIDERS on .release, checkbox toggles +
    # pickers + formats + seed on .change. line/para gap re-SEAT instead. render_state carries the cache along.
    recompose_in: set = {render_state, formats, seed} | augment.components()
    recompose_out = [page_img, overlay_img, labels_file, status]
    for slider in augment.sliders():
        slider.release(on_recompose, inputs=recompose_in, outputs=recompose_out)
    for toggle in [*augment.toggles(), formats, seed]:
        toggle.change(on_recompose, inputs=recompose_in, outputs=recompose_out)
    reroll.click(
        on_reroll,
        inputs={render_state, formats} | augment.components(),
        outputs=[seed, page_img, overlay_img, labels_file, status],
    )

    reseat_in: set = {render_state, line_gap, para_gap, seed} | augment.components()
    reseat_out = [page_img, overlay_img, labels_file, status, render_state]
    for spacing in (line_gap, para_gap):
        spacing.release(on_reseat, inputs=reseat_in, outputs=reseat_out)

    # upscale_view already takes the page positionally — wire it directly (fork-safe: only the image travels).
    upscale_btn.click(
        gpu_task(upscale_view, duration=90), inputs=[page_img], outputs=[upscaled, upscale_status]
    )
