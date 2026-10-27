"""A ground-truth template marked from its scan, for a PAGE file that says neither which of its lines are printed nor
where its ruled lines run: each line read as printed or handwritten by a small CNN (:mod:`print_hand`, trained by
:mod:`train` on :mod:`diffu_page.data.print_hand_set`'s lines) and the page's ruled lines found with OpenCV
(:mod:`rules`), so a drawing prompt draws print as print and the form's grid (:func:`marked`)."""

from diffu_page.marking.mark import marked, template_scan
from diffu_page.marking.print_hand import PRINTED_CHANCE, PrintHandClassifier, PrintHandNet
from diffu_page.marking.rules import find_rules

__all__ = ["PRINTED_CHANCE", "PrintHandClassifier", "PrintHandNet", "find_rules", "marked", "template_scan"]
