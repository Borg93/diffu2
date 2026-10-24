"""The studio's canvas: the gr.Workflow graph of a plan node feeding one draw node per drawer.

The graph is gr.Workflow's schema version 2 (references, operators, subjects and the edges between their ports) as
models: an edge takes its type from the ports it joins and refuses two ports of different types. The references are
declared in the order the app view's endpoint takes them (template, text, style, seed, steps), and the first drawer's
crops come first, so the endpoint is ``/<first drawer>_crops``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from diffu_page.captions.prompt import PageStyle

PLAN = "plan"  # the plan node's bound function
_COLUMN = 340  # px between the canvas' columns: references, plan, draw nodes, results
_ROW = 150  # px between the nodes of a column


def draw_fn(drawer: str) -> str:
    """The bound function of ``drawer``'s draw node."""
    return f"draw_{drawer}"


class PortType(StrEnum):
    """What a port carries."""

    FILE = "file"
    TEXT = "text"
    JSON = "json"
    NUMBER = "number"
    IMAGE = "image"


class Port(BaseModel):
    """A node's input or output. ``required``: an operator input its function cannot run without; ``output_index``: an
    operator output's place in what its function returns."""

    model_config = ConfigDict(frozen=True)

    id: str
    label: str
    type: PortType
    required: bool | None = None
    output_index: int | None = None


class Node(BaseModel):
    """A canvas node: its ports and where it sits on the canvas."""

    model_config = ConfigDict(frozen=True)

    id: str
    label: str
    inputs: list[Port]
    outputs: list[Port]
    x: int
    y: int


class Reference(Node):
    """An input value of the canvas; ``data`` holds its default."""

    role: Literal["reference"] = "reference"
    asset_type: PortType
    data: dict[str, JsonValue] | None = None


class Operator(Node):
    """A function of the studio, bound by the name ``fn``."""

    role: Literal["operator"] = "operator"
    kind: Literal["fn"] = "fn"
    fn: str


class Subject(Node):
    """A result shown on the canvas."""

    role: Literal["subject"] = "subject"
    asset_type: PortType


class Edge(BaseModel):
    """A wire from one node's output port to another's input port."""

    model_config = ConfigDict(frozen=True)

    id: str
    from_node_id: str
    from_port_id: str
    to_node_id: str
    to_port_id: str
    type: PortType


class WorkflowGraph(BaseModel):
    """A gr.Workflow canvas in schema version 2."""

    schema_version: Literal["2"] = "2"
    name: str
    references: list[Reference]
    operators: list[Operator]
    subjects: list[Subject]
    edges: list[Edge]


def _find(ports: list[Port], port_id: str, node: Node) -> Port:
    port = next((port for port in ports if port.id == port_id), None)
    if port is None:
        raise ValueError(f"{node.id} has no port {port_id!r}")
    return port


def edge(source: Node, output: str, target: Node, input_: str) -> Edge:
    """The wire from ``source``'s output port ``output`` to ``target``'s input port ``input_``.

    Raises:
        ValueError: If a node lacks the port, or the two ports carry different types.
    """
    out, into = _find(source.outputs, output, source), _find(target.inputs, input_, target)
    if out.type is not into.type:
        raise ValueError(f"{source.id}.{output} carries {out.type}, but {target.id}.{input_} takes {into.type}")
    return Edge(
        id=f"{source.id}.{output}->{target.id}.{input_}",
        from_node_id=source.id,
        from_port_id=output,
        to_node_id=target.id,
        to_port_id=input_,
        type=out.type,
    )


def _reference(node_id: str, label: str, port_type: PortType, *, default: JsonValue, row: int) -> Reference:
    return Reference(
        id=node_id,
        label=label,
        asset_type=port_type,
        inputs=[Port(id="in", label=label, type=port_type)],
        outputs=[Port(id="out", label=label, type=port_type)],
        x=0,
        y=row * _ROW,
        data=None if default is None else {"out": default},
    )


def _subject(node_id: str, label: str, port_type: PortType, *, row: int) -> Subject:
    return Subject(
        id=node_id,
        label=label,
        asset_type=port_type,
        inputs=[Port(id="in", label=label, type=port_type)],
        outputs=[Port(id="out", label=label, type=port_type)],
        x=3 * _COLUMN,
        y=row * _ROW,
    )


def studio_graph(drawers: list[str], style: PageStyle) -> WorkflowGraph:
    """The canvas for ``drawers`` (by name), its style reference set to ``style``."""
    template = _reference("ref_template", "Template", PortType.FILE, default=None, row=0)
    text = _reference("ref_text", "Text (blank: the template's own)", PortType.TEXT, default="", row=1)
    style_reference = _reference("ref_style", "Style", PortType.JSON, default=style.model_dump(mode="json"), row=2)
    seed = _reference("ref_seed", "Seed", PortType.NUMBER, default=0, row=3)
    steps = _reference("ref_steps", "Steps", PortType.NUMBER, default=50, row=4)
    plan = Operator(
        id="op_plan",
        label="Plan the page",
        fn=PLAN,
        inputs=[
            Port(id="template", label="Template", type=PortType.FILE, required=True),
            Port(id="text", label="Text", type=PortType.TEXT),
            Port(id="style", label="Style", type=PortType.JSON, required=True),
        ],
        outputs=[Port(id="plans", label="Crop plans", type=PortType.JSON, output_index=0)],
        x=_COLUMN,
        y=_ROW,
    )
    draws: list[Operator] = []
    subjects: list[Subject] = []
    edges = [
        edge(template, "out", plan, "template"),
        edge(text, "out", plan, "text"),
        edge(style_reference, "out", plan, "style"),
    ]
    for row, name in enumerate(drawers):
        draw = Operator(
            id=f"op_draw_{name}",
            label=f"Draw with {name}",
            fn=draw_fn(name),
            inputs=[
                Port(id="plans", label="Crop plans", type=PortType.JSON, required=True),
                Port(id="seed", label="Seed", type=PortType.NUMBER),
                Port(id="steps", label="Steps", type=PortType.NUMBER),
            ],
            outputs=[
                Port(id="sheet", label="Crops", type=PortType.IMAGE, output_index=0),
                Port(id="labels", label="Crops and labels (zip)", type=PortType.FILE, output_index=1),
            ],
            x=2 * _COLUMN,
            y=2 * row * _ROW,
        )
        crops = _subject(f"sub_{name}_crops", f"{name} crops", PortType.IMAGE, row=2 * row)
        labels = _subject(f"sub_{name}_labels", f"{name} labels", PortType.FILE, row=2 * row + 1)
        draws.append(draw)
        subjects += [crops, labels]
        edges += [
            edge(plan, "plans", draw, "plans"),
            edge(seed, "out", draw, "seed"),
            edge(steps, "out", draw, "steps"),
            edge(draw, "sheet", crops, "in"),
            edge(draw, "labels", labels, "in"),
        ]
    return WorkflowGraph(
        name="diffu studio",
        references=[template, text, style_reference, seed, steps],
        operators=[plan, *draws],
        subjects=subjects,
        edges=edges,
    )
