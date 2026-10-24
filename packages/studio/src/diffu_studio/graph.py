"""The studio's canvas: the gr.Workflow graph of a plan node feeding one draw node per drawer.

The references are declared in the order the app view's endpoint takes them (template, text, style, seed, steps), and
the first drawer's crops come first, so the endpoint is ``/<first drawer>_crops``.
"""

from __future__ import annotations

from typing import Any

from diffu_page.captions.prompt import PageStyle

type Node = dict[str, Any]

_COLUMN = 340  # px between the canvas' columns: references, plan, draw nodes, results
_ROW = 150  # px between the nodes of a column


def _port(port_id: str, label: str, port_type: str, **extra: Any) -> dict[str, Any]:
    return {"id": port_id, "label": label, "type": port_type, **extra}


def _reference(node_id: str, label: str, port_type: str, default: object, row: int) -> Node:
    node: Node = {
        "id": node_id,
        "label": label,
        "role": "reference",
        "asset_type": port_type,
        "inputs": [_port("in", label, port_type)],
        "outputs": [_port("out", label, port_type)],
        "x": 0,
        "y": row * _ROW,
    }
    if default is not None:
        node["data"] = {"out": default}
    return node


def _subject(node_id: str, label: str, port_type: str, row: int) -> Node:
    return {
        "id": node_id,
        "label": label,
        "role": "subject",
        "asset_type": port_type,
        "inputs": [_port("in", label, port_type)],
        "outputs": [_port("out", label, port_type)],
        "x": 3 * _COLUMN,
        "y": row * _ROW,
    }


def _edge(source: tuple[str, str], target: tuple[str, str], port_type: str) -> dict[str, str]:
    return {
        "id": f"{source[0]}.{source[1]}->{target[0]}.{target[1]}",
        "from_node_id": source[0],
        "from_port_id": source[1],
        "to_node_id": target[0],
        "to_port_id": target[1],
        "type": port_type,
    }


def studio_graph(drawers: list[str], style: PageStyle) -> dict[str, Any]:
    """The canvas for ``drawers`` (by name), its style reference set to ``style``."""
    references = [
        _reference("ref_template", "Template", "file", None, 0),
        _reference("ref_text", "Text (blank: the template's own)", "text", "", 1),
        _reference("ref_style", "Style", "json", style.model_dump(mode="json"), 2),
        _reference("ref_seed", "Seed", "number", 0, 3),
        _reference("ref_steps", "Steps", "number", 50, 4),
    ]
    plan = {
        "id": "op_plan",
        "label": "Plan the page",
        "role": "operator",
        "kind": "fn",
        "fn": "plan",
        "inputs": [
            _port("template", "Template", "file", required=True),
            _port("text", "Text", "text"),
            _port("style", "Style", "json", required=True),
        ],
        "outputs": [_port("plans", "Crop plans", "json", output_index=0)],
        "x": _COLUMN,
        "y": _ROW,
    }
    draws: list[Node] = []
    subjects: list[Node] = []
    edges = [
        _edge(("ref_template", "out"), ("op_plan", "template"), "file"),
        _edge(("ref_text", "out"), ("op_plan", "text"), "text"),
        _edge(("ref_style", "out"), ("op_plan", "style"), "json"),
    ]
    for row, name in enumerate(drawers):
        draw_id = f"op_draw_{name}"
        draws.append(
            {
                "id": draw_id,
                "label": f"Draw with {name}",
                "role": "operator",
                "kind": "fn",
                "fn": f"draw_{name}",
                "inputs": [
                    _port("plans", "Crop plans", "json", required=True),
                    _port("seed", "Seed", "number"),
                    _port("steps", "Steps", "number"),
                ],
                "outputs": [
                    _port("sheet", "Crops", "image", output_index=0),
                    _port("labels", "Crops and labels (zip)", "file", output_index=1),
                ],
                "x": 2 * _COLUMN,
                "y": 2 * row * _ROW,
            }
        )
        subjects.append(_subject(f"sub_{name}_crops", f"{name} crops", "image", 2 * row))
        subjects.append(_subject(f"sub_{name}_labels", f"{name} labels", "file", 2 * row + 1))
        edges += [
            _edge(("op_plan", "plans"), (draw_id, "plans"), "json"),
            _edge(("ref_seed", "out"), (draw_id, "seed"), "number"),
            _edge(("ref_steps", "out"), (draw_id, "steps"), "number"),
            _edge((draw_id, "sheet"), (f"sub_{name}_crops", "in"), "image"),
            _edge((draw_id, "labels"), (f"sub_{name}_labels", "in"), "file"),
        ]
    return {
        "schema_version": "2",
        "name": "diffu studio",
        "references": references,
        "operators": [plan, *draws],
        "subjects": subjects,
        "edges": edges,
    }
