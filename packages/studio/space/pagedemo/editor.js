// The demo's page editor. Gradio runs this once as the gr.HTML component's js_on_load, with `element` (the component's
// root), `props` (its value and the templates), `trigger` and `watch` in scope. The value is the page as JSON
// (pagedemo/document.py: aspect, lines, boxes with id, kind, rect [x0, y0, x1, y1] on 0-1000, text, printed) and,
// from Python only, "result": the drawn page as a data URI, which the browser keeps and never sends back.
// `props.templates` holds the templates as JSON, by name: a page each, loaded here without a trip to the server.

const GRID = 1000;
const CHARS_PER_ASPECT = 3.7; // the pipeline's rule: characters a line takes per unit of its width over its height
const SIDE_MARGIN = 0.15; // of a line pitch: what the caption adds to each side of a line's box
const SPLIT_MARK = "¬";
const MIN_SPLIT = 3;
const MIN_WIDTH = 24;
const THINNEST = 4;
const KINDS = {
  text: ["Text", "#2f6fb5"],
  heading: ["Heading", "#7a3fb0"],
  note: ["Margin note", "#2e8a57"],
  number: ["Page number", "#c06a1b"],
  rule: ["Ruled line", "#555b66"],
};
const HANDLES = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];
const FINGERS = window.matchMedia("(pointer: coarse)").matches; // a touch screen: boxes come from + Box, not a drag
const REMEMBERED = 60; // the edits Undo goes back over

const q = (selector) => element.querySelector(selector);
const stage = q(".pe-stage");
const layer = q(".pe-layer");
const picture = q(".pe-result");
const hint = q(".pe-hint");
const templateSelect = q(".pe-template");
const undoButton = q(".pe-undo");
const fileInput = q(".pe-file");
const shape = q(".pe-shape");
const linesInput = q(".pe-lines");
const linesValue = q(".pe-lines-value");
const resultButton = q(".pe-toggle-result");
const boxesButton = q(".pe-toggle-boxes");
const inspector = q(".pe-inspector");
const kindSelect = q(".pe-kind");
const textArea = q(".pe-text");
const room = q(".pe-room");

let page = { aspect: 0.7, lines: 30, boxes: [] };
let result = null;
let showResult = true;
let showBoxes = true;
let selected = null;
let drag = null;
let ghost = null;
let committed = null;
let pending = null;
let typing = false;
let sliding = false;
const past = [];

const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const pitch = () => GRID / page.lines;
const boxOf = (id) => page.boxes.find((box) => box.id === id);
const tidy = (rect) => rect.map((value) => Math.round(value * 100) / 100);

// Python's round: a half goes to the even neighbour.
function roundHalfEven(value) {
  const rounded = Math.round(value);
  return Math.abs(value % 1) === 0.5 && rounded % 2 !== 0 ? rounded - 1 : rounded;
}

function slotCount(box) {
  return Math.max(1, Math.floor(((box.rect[3] - box.rect[1]) / GRID) * page.lines + 0.25));
}

// The characters a line of the box takes: its slot's width over its height, the sides inset as the page is laid out.
function lineBudget(box) {
  let across = ((box.rect[2] - box.rect[0]) / GRID) * page.aspect * page.lines;
  if (across > 6 * SIDE_MARGIN) across -= 2 * SIDE_MARGIN;
  return Math.max(1, roundHalfEven(CHARS_PER_ASPECT * across));
}

// The box's text written into its line slots as the pipeline writes it (diffu_page's refill): words fill a line up to
// its budget, a word that overruns is split with the mark when both halves keep MIN_SPLIT characters, and a line
// break of the author's starts a new line. Returns the lines that fit and how many words do not.
function wrap(box) {
  const count = slotCount(box);
  const limit = lineBudget(box);
  const paragraphs = box.text
    .replace(/\r/g, "\n")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => line.split(/\s+/));
  let words = [];
  const next = () => {
    if (!words.length) words = paragraphs.shift() || [];
    if (!words.length) return null;
    let line = "";
    while (words.length) {
      const word = words[0];
      const candidate = line ? `${line} ${word}` : word;
      if (candidate.length <= limit) {
        line = candidate;
        words.shift();
        continue;
      }
      const left = limit - line.length - (line ? 1 : 0) - SPLIT_MARK.length;
      const splitsWell = left >= MIN_SPLIT && word.length - left >= MIN_SPLIT;
      if (line && !splitsWell) return line;
      const cut = Math.max(left, 1);
      words[0] = word.slice(cut);
      const head = word.slice(0, cut) + SPLIT_MARK;
      return line ? `${line} ${head}` : head;
    }
    return line;
  };
  const lines = [];
  let over = 0;
  for (let guard = 0; guard < 3000; guard += 1) {
    const line = next();
    if (line === null) break;
    if (lines.length < count) lines.push(line);
    else over += line.split(" ").length;
  }
  return { count, limit, lines, over };
}

function nextId(kind) {
  for (let number = 1; ; number += 1) {
    if (!boxOf(`${kind}_${number}`)) return `${kind}_${number}`;
  }
}

// A page as the editor holds it, from anything read as JSON: its numbers kept in range, its boxes those well formed.
function cleaned(value) {
  const boxes = Array.isArray(value.boxes) ? value.boxes : [];
  return {
    aspect: clamp(Number(value.aspect) || 0.7, 0.4, 2.5),
    lines: clamp(Math.round(Number(value.lines)) || 30, 8, 60),
    boxes: boxes
      .filter((box) => box && typeof box.id === "string" && Array.isArray(box.rect) && box.rect.length === 4)
      .map((box) => ({
        id: box.id,
        kind: box.kind in KINDS ? box.kind : "text",
        rect: tidy(box.rect.map(Number)),
        text: typeof box.text === "string" ? box.text : "",
        printed: Boolean(box.printed),
      })),
  };
}

function load() {
  let value = {};
  try {
    value = JSON.parse(props.value || "{}");
  } catch (error) {
    value = {};
  }
  if ("result" in value) {
    result = value.result || null;
    showResult = true;
  }
  page = cleaned(value);
  if (!boxOf(selected)) selected = null;
}

// The page as it is now kept for Undo, before an edit changes it.
function remember() {
  const now = JSON.stringify(page);
  if (past[past.length - 1] !== now) past.push(now);
  if (past.length > REMEMBERED) past.shift();
  undoButton.disabled = false;
}

function undo() {
  const last = past.pop();
  undoButton.disabled = !past.length;
  if (last === undefined) return;
  page = cleaned(JSON.parse(last));
  if (!boxOf(selected)) selected = null;
  layout();
  commit(true);
  render();
}

// Another page in place of this one (a template, a file opened): the drawn page under the old one goes with it.
function replacePage(value) {
  remember();
  page = cleaned(value);
  result = null;
  selected = null;
  showBoxes = true;
  layout();
  commit(true);
  render();
}

function commit(now) {
  clearTimeout(pending);
  const send = () => {
    pending = null;
    committed = JSON.stringify(page);
    props.value = committed;
    trigger("change");
  };
  if (now) send();
  else pending = setTimeout(send, 300);
}

function layout() {
  stage.style.aspectRatio = String(page.aspect);
  stage.style.maxWidth = `calc(84vh * ${page.aspect})`;
  const options = [...shape.options];
  const nearest = options.reduce((best, option) =>
    Math.abs(Number(option.value) - page.aspect) < Math.abs(Number(best.value) - page.aspect) ? option : best,
  );
  shape.value = nearest.value;
  linesInput.value = String(clamp(page.lines, Number(linesInput.min), Number(linesInput.max)));
  linesValue.textContent = String(page.lines);
}

function place(node, rect) {
  const [x0, y0, x1, y1] = rect;
  node.style.left = `${x0 / 10}%`;
  node.style.top = `${y0 / 10}%`;
  node.style.width = `${(x1 - x0) / 10}%`;
  node.style.height = `${(y1 - y0) / 10}%`;
}

function boxNode(box, slotPx) {
  const [name, color] = KINDS[box.kind];
  const node = document.createElement("div");
  node.className = "pe-box";
  node.dataset.id = box.id;
  node.style.setProperty("--c", color);
  node.classList.toggle("is-selected", box.id === selected);
  node.classList.toggle("is-printed", box.printed && box.kind !== "rule");
  node.classList.toggle("is-rule", box.kind === "rule");
  node.classList.toggle("at-top", box.rect[1] < 2 * pitch());
  place(node, box.rect);
  let label = name;
  if (box.kind === "rule") {
    const line = document.createElement("div");
    line.className = "pe-rule";
    const flat = (box.rect[2] - box.rect[0]) * page.aspect >= box.rect[3] - box.rect[1];
    if (flat) Object.assign(line.style, { left: "0", right: "0", top: "calc(50% - 1px)", height: "2px" });
    else Object.assign(line.style, { top: "0", bottom: "0", left: "calc(50% - 1px)", width: "2px" });
    node.appendChild(line);
  } else {
    const written = wrap(box);
    for (let k = 0; k < written.count; k += 1) {
      const slot = document.createElement("div");
      slot.className = "pe-slot";
      slot.classList.toggle("is-empty", !written.lines[k]);
      slot.style.top = `${k * slotPx}px`;
      slot.style.height = `${slotPx}px`;
      slot.style.fontSize = `${Math.max(6, slotPx * 0.5)}px`;
      slot.textContent = written.lines[k] || "";
      node.appendChild(slot);
    }
    if (box.printed) label += " · printed";
    if (written.over) label += ` · ${written.over} word${written.over === 1 ? "" : "s"} do not fit`;
    node.classList.toggle("is-over", written.over > 0);
  }
  const tag = document.createElement("span");
  tag.className = "pe-tag";
  tag.textContent = label;
  node.appendChild(tag);
  for (const direction of HANDLES) {
    const handle = document.createElement("span");
    handle.className = "pe-handle";
    handle.dataset.dir = direction;
    node.appendChild(handle);
  }
  return node;
}

function renderBoxes() {
  const slotPx = stage.clientHeight / page.lines;
  layer.replaceChildren(...page.boxes.map((box) => boxNode(box, slotPx)));
  if (ghost) layer.appendChild(ghost);
}

function renderInspector() {
  const box = boxOf(selected);
  inspector.classList.toggle("is-idle", !box || !showBoxes);
  if (!box) {
    inspector.style.removeProperty("--c");
    return;
  }
  inspector.style.setProperty("--c", KINDS[box.kind][1]);
  kindSelect.value = box.kind;
  textArea.classList.toggle("is-unused", box.kind === "rule");
  if (textArea.value !== box.text) textArea.value = box.text;
  if (box.kind === "rule") {
    room.textContent = "a straight line along the box's longer side";
    room.title = room.textContent;
    room.classList.remove("is-over");
    return;
  }
  const written = wrap(box);
  const parts = [`${written.count} line${written.count === 1 ? "" : "s"}`, `about ${written.limit} characters a line`];
  if (written.over) parts.push(`${written.over} word${written.over === 1 ? "" : "s"} do not fit`);
  room.textContent = (written.over ? [parts[2], parts[0], parts[1]] : parts).join(" · ");
  room.title = room.textContent;
  room.classList.toggle("is-over", written.over > 0);
}

function render() {
  layer.classList.toggle("is-hidden", !showBoxes);
  const shown = Boolean(result) && showResult;
  stage.classList.toggle("has-result", shown);
  picture.hidden = !shown;
  if (shown && picture.getAttribute("src") !== result) picture.setAttribute("src", result);
  resultButton.hidden = !result;
  resultButton.textContent = showResult ? "Hide page" : "Show page";
  boxesButton.textContent = showBoxes ? "Hide boxes" : "Show boxes";
  renderBoxes();
  renderInspector();
  if (!showBoxes) hint.textContent = "The boxes are hidden. Show them to change the page.";
  else if (!page.boxes.length) hint.textContent = `${FINGERS ? "Tap + Box to add a box" : "Drag on the page to draw a box"}, or pick a template. A box is a block of lines.`;
  else hint.textContent = `Drag a box to move it, a ${FINGERS ? "corner" : "handle"} to resize it${FINGERS ? "" : "; Delete removes it"}. Every line is drawn at one height: the Lines slider sets it.`;
}

function select(id) {
  selected = id;
  render();
}

function gridPoint(event) {
  const bounds = stage.getBoundingClientRect();
  return [
    clamp(((event.clientX - bounds.left) / bounds.width) * GRID, 0, GRID),
    clamp(((event.clientY - bounds.top) / bounds.height) * GRID, 0, GRID),
  ];
}

// A rectangle a pointer drew or resized, kept on the page: a text box at least MIN_WIDTH wide and a whole number of
// line pitches tall, measured from the edge the pointer does not hold; a rule only kept from vanishing.
function snapped(rect, held, kind) {
  let [x0, y0, x1, y1] = rect;
  const narrowest = kind === "rule" ? THINNEST : MIN_WIDTH;
  if (x1 - x0 < narrowest) {
    if (held.includes("w")) x0 = x1 - narrowest;
    else x1 = x0 + narrowest;
  }
  if (kind === "rule") {
    if (y1 - y0 < THINNEST) {
      if (held.includes("n")) y0 = y1 - THINNEST;
      else y1 = y0 + THINNEST;
    }
  } else {
    const step = pitch();
    const fromBottom = held.includes("n");
    const most = Math.max(1, Math.floor((fromBottom ? y1 : GRID - y0) / step + 1e-6));
    const count = clamp(Math.round((y1 - y0) / step), 1, most);
    if (fromBottom) y0 = y1 - count * step;
    else y1 = y0 + count * step;
  }
  const shiftX = x0 < 0 ? -x0 : x1 > GRID ? GRID - x1 : 0;
  const shiftY = y0 < 0 ? -y0 : y1 > GRID ? GRID - y1 : 0;
  return tidy([x0 + shiftX, y0 + shiftY, x1 + shiftX, y1 + shiftY]);
}

function moved(rect, dx, dy) {
  const [x0, y0, x1, y1] = rect;
  const shiftX = clamp(dx, -x0, GRID - x1);
  const shiftY = clamp(dy, -y0, GRID - y1);
  return tidy([x0 + shiftX, y0 + shiftY, x1 + shiftX, y1 + shiftY]);
}

function resized(rect, held, x, y, kind) {
  let [x0, y0, x1, y1] = rect;
  if (held.includes("w")) x0 = Math.min(x, x1 - 1);
  if (held.includes("e")) x1 = Math.max(x, x0 + 1);
  if (held.includes("n")) y0 = Math.min(y, y1 - 1);
  if (held.includes("s")) y1 = Math.max(y, y0 + 1);
  return snapped([x0, y0, x1, y1], held, kind);
}

stage.addEventListener("pointerdown", (event) => {
  if (event.button !== 0 || !showBoxes) return;
  const handle = event.target.closest(".pe-handle");
  const node = event.target.closest(".pe-box");
  const at = gridPoint(event);
  if (node && boxOf(node.dataset.id)) {
    const box = boxOf(node.dataset.id);
    drag = { mode: handle ? handle.dataset.dir : "move", id: box.id, at, rect: [...box.rect], moved: false };
    select(box.id);
  } else if (event.pointerType === "touch") {
    select(null); // a finger on the bare page scrolls it
    return;
  } else {
    drag = { mode: "new", at, rect: null, moved: false };
    select(null);
  }
  stage.setPointerCapture(event.pointerId);
  stage.focus({ preventScroll: true });
  event.preventDefault();
});

stage.addEventListener("pointermove", (event) => {
  if (!drag) return;
  const [x, y] = gridPoint(event);
  const [startX, startY] = drag.at;
  if (!drag.moved && Math.hypot(x - startX, y - startY) < 4) return;
  if (!drag.moved) remember();
  drag.moved = true;
  if (drag.mode === "new") {
    const held = `${y < startY ? "n" : "s"}${x < startX ? "w" : "e"}`;
    drag.rect = snapped([Math.min(startX, x), Math.min(startY, y), Math.max(startX, x), Math.max(startY, y)], held, "text");
    if (!ghost) {
      ghost = document.createElement("div");
      ghost.className = "pe-ghost";
      layer.appendChild(ghost);
    }
    place(ghost, drag.rect);
    return;
  }
  const box = boxOf(drag.id);
  if (!box) return;
  box.rect = drag.mode === "move" ? moved(drag.rect, x - startX, y - startY) : resized(drag.rect, drag.mode, x, y, box.kind);
  renderBoxes();
  renderInspector();
});

function endDrag() {
  if (!drag) return;
  const done = drag;
  drag = null;
  if (ghost) {
    ghost.remove();
    ghost = null;
  }
  if (done.mode === "new") {
    if (done.moved && done.rect) {
      const box = { id: nextId("text"), kind: "text", rect: done.rect, text: "", printed: false };
      page.boxes.push(box);
      selected = box.id;
      commit(true);
      render();
      textArea.focus({ preventScroll: true });
    }
    return;
  }
  if (done.moved) commit(true);
  render();
}

stage.addEventListener("pointerup", endDrag);
stage.addEventListener("pointercancel", endDrag);

function remove(id) {
  remember();
  page.boxes = page.boxes.filter((box) => box.id !== id);
  if (selected === id) selected = null;
  commit(true);
  render();
}

const ARROWS = { ArrowUp: [0, -1], ArrowDown: [0, 1], ArrowLeft: [-1, 0], ArrowRight: [1, 0] };
stage.addEventListener("keydown", (event) => {
  const box = boxOf(selected);
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    undo();
    event.preventDefault();
    return;
  }
  if (!box || !showBoxes) return;
  if (event.key === "Delete" || event.key === "Backspace") {
    remove(box.id);
  } else if (event.key === "Escape") {
    select(null);
  } else if (event.key in ARROWS) {
    const reach = event.shiftKey ? 10 : 2;
    remember();
    box.rect = moved(box.rect, ARROWS[event.key][0] * reach, ARROWS[event.key][1] * reach);
    commit(false);
    render();
  } else {
    return;
  }
  event.preventDefault();
});

textArea.addEventListener("input", () => {
  const box = boxOf(selected);
  if (!box) return;
  if (!typing) remember();
  typing = true;
  box.text = textArea.value;
  commit(false);
  renderBoxes();
  renderInspector();
});
textArea.addEventListener("blur", () => {
  typing = false;
  if (pending !== null) commit(true); // nothing typed is left waiting when a button is pressed next
});

kindSelect.addEventListener("change", () => {
  const box = boxOf(selected);
  if (!box || !(kindSelect.value in KINDS)) return;
  remember();
  box.kind = kindSelect.value;
  box.id = nextId(box.kind);
  selected = box.id;
  if (box.kind === "rule") box.printed = false;
  else box.rect = snapped(box.rect, "se", box.kind);
  commit(true);
  render();
});

q(".pe-delete").addEventListener("click", () => {
  if (boxOf(selected)) remove(selected);
});

q(".pe-copy").addEventListener("click", () => {
  const box = boxOf(selected);
  if (!box) return;
  remember();
  const tall = box.rect[3] - box.rect[1];
  const down = box.rect[3] + pitch() + tall <= GRID ? tall + pitch() : 0;
  const copy = { ...box, id: nextId(box.kind), rect: moved(box.rect, down ? 0 : 20, down) };
  page.boxes.push(copy);
  selected = copy.id;
  commit(true);
  render();
});

q(".pe-clear").addEventListener("click", () => {
  remember();
  page.boxes = [];
  selected = null;
  result = null;
  showBoxes = true;
  commit(true);
  render();
});

resultButton.addEventListener("click", () => {
  showResult = !showResult;
  render();
});

boxesButton.addEventListener("click", () => {
  showBoxes = !showBoxes;
  render();
});

shape.addEventListener("change", () => {
  remember();
  page.aspect = Number(shape.value);
  layout();
  commit(true);
  render();
});

linesInput.addEventListener("input", () => {
  if (!sliding) remember();
  sliding = true;
  page.lines = Number(linesInput.value);
  linesValue.textContent = String(page.lines);
  commit(false);
  render();
});
linesInput.addEventListener("change", () => {
  sliding = false;
});

undoButton.addEventListener("click", undo);

q(".pe-add").addEventListener("click", () => {
  remember();
  const step = pitch();
  const lowest = page.boxes.reduce((bottom, box) => Math.max(bottom, box.rect[3]), 60 - step);
  const top = lowest + 4 * step <= GRID - 40 ? lowest + step : 60;
  const box = { id: nextId("text"), kind: "text", rect: tidy([120, top, 870, top + 3 * step]), text: "", printed: false };
  page.boxes.push(box);
  selected = box.id;
  showBoxes = true;
  commit(true);
  render();
  textArea.focus({ preventScroll: true });
});

if (FINGERS) q(".pe-idle").textContent = "Tap a box to write its text, or + Box to add one.";

let templates = {};
try {
  templates = JSON.parse(props.templates || "{}");
} catch (error) {
  templates = {};
}
for (const name of Object.keys(templates)) templateSelect.add(new Option(name, name));
templateSelect.addEventListener("change", () => {
  const chosen = templates[templateSelect.value];
  templateSelect.value = "";
  if (chosen) replacePage(chosen);
});

q(".pe-save").addEventListener("click", () => {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([JSON.stringify(page, null, 1)], { type: "application/json" }));
  link.download = "page.json";
  link.click();
  URL.revokeObjectURL(link.href);
});

q(".pe-load").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  fileInput.value = "";
  if (!file) return;
  try {
    const read = JSON.parse(await file.text());
    const value = read && read.page && Array.isArray(read.page.boxes) ? read.page : read; // a drawn page's record too
    if (!value || !Array.isArray(value.boxes)) throw new Error("no boxes");
    replacePage(value);
  } catch (error) {
    hint.textContent = "That file is not a page saved here.";
  }
});

watch("value", () => {
  if (props.value === committed) return; // an edit of our own coming back
  committed = props.value;
  load();
  layout();
  render();
});

new ResizeObserver(() => render()).observe(stage);
committed = props.value;
load();
layout();
render();
