"""A template marked from its scan: each line its PAGE file does not say how was made read as printed or handwritten
(:mod:`diffu_page.marking.print_hand`), and its ruled lines found on the scan (:mod:`diffu_page.marking.rules`) when
the file has none. One stage, stateless but for the classifier it is given: a page's layout and scan in, the marked
layout out."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from diffu_page.layout import Layout, Production
from diffu_page.layout.page_xml import find_scan
from diffu_page.marking.lines import line_images, on_scan, read_scan
from diffu_page.marking.print_hand import PRINTED_CHANCE, PrintHandClassifier
from diffu_page.marking.rules import find_rules


def marked(layout: Layout, scan: np.ndarray, *, classifier: PrintHandClassifier) -> Layout:
    """``layout`` marked from its grey ``scan``: every line without a production of its own read by ``classifier``,
    printed when it is at least :data:`PRINTED_CHANCE` sure and else handwritten, every line with one kept as it is;
    and the ruled lines found on the scan when the layout has none (a PAGE file's ``SeparatorRegion`` s are kept)."""
    unstated = [k for k, line in enumerate(layout.lines) if line.production is None]
    chances = classifier([line_images(scan, on_scan(layout.lines[k].box, layout, scan)) for k in unstated])
    productions = {
        k: Production.PRINTED if chance >= PRINTED_CHANCE else Production.HANDWRITTEN
        for k, chance in zip(unstated, chances, strict=True)
    }
    lines = [
        line.model_copy(update={"production": productions[k]}) if k in productions else line
        for k, line in enumerate(layout.lines)
    ]
    rules = layout.rules or find_rules(scan, layout)
    return layout.model_copy(update={"lines": lines, "rules": rules})


def template_scan(template: Path) -> np.ndarray:
    """The grey scan a template's PAGE file describes, beside it (:func:`diffu_page.layout.page_xml.find_scan`).

    Raises:
        FileNotFoundError: If there is no scan beside the template.
    """
    scan = find_scan(template)
    if scan is None:
        raise FileNotFoundError(
            f"{template} has no scan beside it to be marked from (its imageFilename, one folder up)"
        )
    return read_scan(scan)
