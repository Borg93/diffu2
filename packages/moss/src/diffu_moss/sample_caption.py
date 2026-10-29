"""A reference-drawn caption in the forms MOSS-Image draws from and trains on.

The page pipeline captions a crop drawn from a reference image as FLUX.3 image-edit records are captioned
(``diffu_page.captions.restyle``): the paragraph takes the hand, ink and paper "of <ref_image_0>" and every handwritten
line row is written "in the reference's hand". In the plain form (v3, v3c) that is all the caption says of the look; in
the described form (v3d) it also names the reference's measured hand, ink and paper: the opening's clause after the
citation (slant, pen, letters and spacing, the ink and paper by name and hex), the scan's greyscale, the ink of the
handwriting and of every line and fragment row, the paper row's colour and hex. MOSS's prompt has no image token, so:

- :func:`with_sample`: the citations name "the sample image", for a drawing with the reference image beside the prompt;
- :func:`without_sample`: the citations dropped and every other word kept, for a drawing from the prompt alone;
- :func:`without_look`: a described caption's look words dropped: exactly the plain caption of the same crop, so the
  look of a drawing from it must come from the sample image.

A caption that cites no reference (a page drawn in a named style) is left as it is by the first two.
:func:`sample_layout_prompt` compiles a caption in the form for a drawing with or without the sample image as MOSS's
Layout prompt, as the page pipeline's ``MossLayoutPromptStep`` compiles a crop and a page pair is trained on.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

from diffu_moss.layout_prompt import layout_prompt

CITATIONS = (  # how a caption cites its reference image, and the words that name the sample image instead
    # after the archive a named caption names (v3b, the page pairs): dropped with its comma
    (", in the hand, ink and paper of <ref_image_0>", ", in the hand, ink and paper of the sample image"),
    (" in the hand, ink and paper of <ref_image_0>", " in the hand, ink and paper of the sample image"),
    (" in the reference's hand", " in the sample image's hand"),
)
DESCRIBED_OPENING = re.compile(
    r"^(?P<lead>Create a flat, evenly lit )(?:greyscale )?(?P<page>archival scan of a new [^<]+? in the hand, ink and "
    r"paper of <ref_image_0>): [^.]+\.(?P<paper> The paper <paper_1>)"
)
DESCRIBED_INK_OPENING = re.compile(r"(?P<lines> carries \d+ lines of cursive handwriting) in [a-z -]+? ink")
DESCRIBED_PAPER_ROW = re.compile(
    r"Aged [a-z -]+ paper \(#[0-9A-Fa-f]{6}\) of a bound archival volume(?:, in a greyscale scan)?\."
)
DESCRIBED_INK_LINE = re.compile(r", [a-z -]*cursive in the reference's hand, in [a-z -]+ ink\.$")
DESCRIBED_INK_FRAGMENT = re.compile(r"handwritten line in [a-z -]+ ink, cut off")
PLAIN_PAPER_ROW = "Aged paper of a bound archival volume."
PLAIN_INK_LINE = ", cursive script in the reference's hand and ink."
PLAIN_INK_FRAGMENT = "handwritten line, cut off"


def with_sample(caption: str) -> str:
    """``caption`` for a drawing with its reference image beside the prompt: its citations name the sample image."""
    for cited, named in CITATIONS:
        caption = caption.replace(cited, named)
    return caption


def without_sample(caption: str) -> str:
    """``caption`` for a drawing from the prompt alone: its citations of the reference dropped, their words kept."""
    for cited, _ in CITATIONS:
        caption = caption.replace(cited, "")
    return caption


def sample_layout_prompt(caption: str, texts: Mapping[str, str], size: tuple[int, int], *, sample: bool) -> str:
    """``caption`` as MOSS's Layout prompt (:func:`diffu_moss.layout_prompt.layout_prompt`: each line of ``texts`` by
    its row id, on a canvas of ``size`` (width, height) px), its citations naming the sample image when the drawing is
    given it (``sample``) and dropped when not.

    Raises:
        ValueError: If the caption has no layout list, or a line's row does not quote its text.
    """
    return layout_prompt(with_sample(caption) if sample else without_sample(caption), texts, size)


def without_look(caption: str) -> str:
    """A described reference caption (v3d's form) with its look words dropped: the plain caption (v3c's form) of the
    same crop, its text, layout and citations unchanged.

    Raises:
        ValueError: If the caption is not a described reference caption: its opening describes no look, or its
            handwriting's ink or its paper row is not in the described form.
    """
    start = caption.find(" [{")
    if start < 0:
        raise ValueError(f"no layout list in caption: {caption[:120]!r}")
    paragraph, rows = caption[:start], json.loads(caption[start + 1 :])
    paragraph, opened = DESCRIBED_OPENING.subn(r"\g<lead>\g<page>.\g<paper>", paragraph)
    if not opened:
        raise ValueError(f"the caption's opening describes no look to drop: {caption[:160]!r}")
    paragraph, inked = DESCRIBED_INK_OPENING.subn(r"\g<lines>", paragraph)
    papers = [row for row in rows if row["id"] == "paper_1" and DESCRIBED_PAPER_ROW.fullmatch(row["desc"])]
    if inked != 1 or len(papers) != 1:
        raise ValueError(f"the caption's handwriting ink or paper row is not described: {caption[:160]!r}")
    for row in rows:
        if row is papers[0]:
            row["desc"] = PLAIN_PAPER_ROW
        else:
            row["desc"] = DESCRIBED_INK_FRAGMENT.sub(
                PLAIN_INK_FRAGMENT, DESCRIBED_INK_LINE.sub(PLAIN_INK_LINE, row["desc"])
            )
    return f"{paragraph} {json.dumps(rows, ensure_ascii=False)}"
