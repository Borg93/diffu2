"""A drawn crop's ground truth: its lines and ruled lines as ALTO or PAGE XML, every line a judge could not confirm
tagged ``unclear``. Pure functions of the plan and the judge's verdicts, run after the judge (:mod:`diffu_page.judge`)."""

from __future__ import annotations

from diffu_page.crops.geometry import SLOT_ABOVE_BASELINE, SLOT_BELOW_BASELINE, pixel_box
from diffu_page.judge import LineVerdict
from diffu_page.labels import LabelFormat, LineLabel, PageLabels, Rect, export
from diffu_page.plan import CropPlan


def line_labels(plan: CropPlan, verdicts: list[LineVerdict], max_line_cer: float) -> list[LineLabel]:
    """The plan's lines as labels in crop pixels: the slot as the polygon, the baseline across it where the plan puts
    it (0.7 of the way down the slot for a record without one), a table cell's line in the table region with its
    cell, and a line whose verdict fails (over ``max_line_cer``, or a number read otherwise) tagged ``unclear``."""
    order_of = {index: order for order, index in enumerate(plan.reading_order)}
    labels: list[LineLabel] = []
    for index, (line, verdict) in enumerate(zip(plan.lines, verdicts, strict=True)):
        x0, y0, x1, y1 = (round(v) for v in pixel_box(line.bbox, (plan.width, plan.height)))
        if line.baseline is not None:
            baseline_y = round(line.baseline * plan.height / 1000)
        else:
            baseline_y = round(y0 + SLOT_ABOVE_BASELINE / (SLOT_ABOVE_BASELINE + SLOT_BELOW_BASELINE) * (y1 - y0))
        labels.append(
            LineLabel(
                text=line.text,
                polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                baseline=[(x0, baseline_y), (x1, baseline_y)],
                region=line.kind,
                block=line.block,
                order=order_of[index],
                tag="" if verdict.passes(max_line_cer) else "unclear",
                production=line.production,
                cell=line.cell,
            )
        )
    return labels


def rule_rects(plan: CropPlan) -> list[Rect]:
    """The plan's ruled lines in crop pixels."""
    return [
        (round(x0), round(y0), round(x1), round(y1))
        for x0, y0, x1, y1 in (pixel_box(rule, (plan.width, plan.height)) for rule in plan.rules)
    ]


def crop_labels(
    plan: CropPlan, verdicts: list[LineVerdict], *, max_line_cer: float, label_format: LabelFormat, image_name: str
) -> tuple[str, int]:
    """The crop's labels as ``label_format`` XML naming ``image_name``, and how many of its lines are tagged
    ``unclear``."""
    lines = line_labels(plan, verdicts, max_line_cer)
    page = PageLabels(width=plan.width, height=plan.height, image=image_name, lines=lines, rules=rule_rects(plan))
    return export(page, label_format), sum(line.tag == "unclear" for line in lines)
