// RaceView: replays three readers at their measured pace, on the manuscript and as text, then lists Jev's scores.
// Gradio runs this once with `element`, `props` and `watch` in scope; `props.value` is a Race payload (app.py).
// Every text is set with textContent: readings are model output and are never parsed as HTML.

const SVG = "http://www.w3.org/2000/svg";
const LANES = [
  { key: "trocr", name: "Swedish Lion TrOCR", how: "the region as its text lines, cut out beforehand; one batch" },
  { key: "plain", name: "MiniCPM-OCR reader alone", how: "the region as one image; one token per reader pass" },
  { key: "dspark", name: "Reader + DSpark draft", how: "the region as one image; up to 16 tokens per reader pass" },
];
const SPEEDS = [
  { label: "▶ Race them side by side", speed: 1, main: true },
  { label: "½ speed", speed: 0.5 },
  { label: "¼ speed", speed: 0.25 },
];
const still = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const state = { run: null, mode: "live", speed: 1, raceStart: 0, started: {}, frame: 0 };
const derived = new WeakMap(); // per lane payload: what drawing needs, worked out once

function make(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function shape(tag, attributes) {
  const node = document.createElementNS(SVG, tag);
  for (const [name, value] of Object.entries(attributes || {})) node.setAttribute(name, value);
  return node;
}

function payload() {
  let value = props.value;
  if (typeof value === "string") {
    try {
      value = JSON.parse(value);
    } catch (error) {
      return null;
    }
  }
  return value && typeof value === "object" && value.run ? value : null;
}

function percent(rate) {
  return rate === null || rate === undefined ? "" : ` · CER ${(100 * rate).toFixed(1)} %`;
}

// A reader lane's pieces as drawing needs them: when each came, the characters read by then, and its passes
// (pieces that became known together).
function readerFacts(lane) {
  let facts = derived.get(lane);
  if (facts) return facts;
  const times = [];
  const characters = [];
  const passes = [];
  let total = 0;
  for (const [at, text] of lane.pieces) {
    total += text.length;
    times.push(at);
    characters.push(total);
    const last = passes[passes.length - 1];
    if (last && last.at === at) last.tokens += 1;
    else passes.push({ at, tokens: 1 });
  }
  facts = { times, characters, total: Math.max(1, total), passes };
  derived.set(lane, facts);
  return facts;
}

// The manuscript with an outline per line and a highlighter fill that grows as the line is read.
function page(race, key) {
  const box = make("div", "rv-page");
  if (!race.image) {
    box.hidden = true;
    return box;
  }
  box.style.setProperty("--rv-aspect", String(race.aspect || 1));
  const image = make("img");
  image.src = race.image;
  image.alt = "The region being read";
  box.append(image);
  const ink = shape("svg", { class: "rv-ink", viewBox: "0 0 1 1", preserveAspectRatio: "none" });
  const clips = shape("defs");
  ink.append(clips);
  (race.outlines || []).forEach((outline, i) => {
    const xs = outline.map((point) => point[0]);
    const left = Math.min(...xs);
    const id = `rv-${race.run}-${key}-${i}`;
    const clip = shape("clipPath", { id });
    const cut = shape("rect", { x: left, y: 0, width: 0, height: 1 });
    cut.dataset.span = String(Math.max(...xs) - left);
    clip.append(cut);
    clips.append(clip);
    const points = outline.map((point) => point.join(",")).join(" ");
    const fill = shape("polygon", { class: "rv-fill", points, "clip-path": `url(#${id})` });
    const edge = shape("polygon", { class: "rv-edge", points });
    edge.dataset.line = String(i);
    ink.append(fill, edge);
  });
  box.append(ink);
  return box;
}

// How much of each line is read: `amount(i)` gives a fraction for line i.
function inkLines(column, amount) {
  const cuts = column.querySelectorAll(".rv-ink clipPath rect");
  cuts.forEach((cut, i) => {
    const fraction = Math.min(1, Math.max(0, amount(i)));
    cut.setAttribute("width", String(fraction * Number(cut.dataset.span)));
  });
}

// The skeleton, built again whenever Gradio re-renders the template or a new run starts.
function ensure(race) {
  let root = element.querySelector(".rv-root");
  if (!root) {
    root = make("div", "rv-root");
    element.appendChild(root);
  }
  root.dataset.theme = document.body.classList.contains("dark") ? "dark" : "light";
  const built = root.dataset.run === race.run && root.dataset.image === (race.image || "");
  if (built && root.querySelector(".rv-lanes")) return root;
  root.dataset.run = race.run;
  root.dataset.image = race.image || "";
  const track = make("div", "rv-track");
  const lanes = make("div", "rv-lanes");
  for (const lane of LANES) {
    const row = make("div", "rv-row");
    row.dataset.lane = lane.key;
    const name = make("div", "rv-name");
    name.append(make("i", "rv-swatch"), make("span", "", lane.name));
    const bar = make("div", "rv-bar");
    bar.append(make("div", "rv-fillbar"));
    row.append(name, bar, make("div", "rv-time", "waiting"));
    track.append(row);

    const column = make("section", "rv-lane");
    column.dataset.lane = lane.key;
    const head = make("header", "rv-head");
    head.append(make("i", "rv-swatch"), make("b", "", lane.name));
    const body = make("div", "rv-text");
    body.dataset.n = "0";
    column.append(head, make("div", "rv-how", lane.how), make("div", "rv-meta", "\u00a0"), page(race, lane.key));
    if (lane.key !== "trocr") column.append(make("div", "rv-passes"));
    column.append(body);
    if (lane.key === "dspark") {
      const legend = make("div", "rv-legend");
      legend.append(
        make("span", "rv-draft", "kept from the draft"),
        make("span", "", " · "),
        make("span", "rv-own", "the reader's own token"),
      );
      column.append(legend);
    }
    lanes.append(column);
  }
  const controls = make("div", "rv-controls");
  controls.hidden = true;
  for (const choice of SPEEDS) {
    const button = make("button", choice.main ? "rv-button rv-main" : "rv-button", choice.label);
    button.type = "button";
    button.dataset.speed = String(choice.speed);
    controls.append(button);
  }
  root.replaceChildren(
    make("div", "rv-status"),
    track,
    make("p", "rv-summary"),
    controls,
    lanes,
    make("div", "rv-jev"),
  );
  return root;
}

function elapsed(key, now) {
  if (still) return Infinity;
  if (state.mode === "race") return ((now - state.raceStart) / 1000) * state.speed;
  return state.started[key] === undefined ? 0 : (now - state.started[key]) / 1000;
}

function countUntil(times, at, from) {
  let n = Math.min(from, times.length);
  while (n < times.length && times[n] <= at) n += 1;
  while (n > 0 && times[n - 1] > at) n -= 1;
  return n;
}

function drawPasses(column, facts, at, finished) {
  const strip = column.querySelector(".rv-passes");
  if (strip.childElementCount !== facts.passes.length) {
    strip.replaceChildren(
      ...facts.passes.map((pass) => {
        const block = make("i");
        block.style.flexGrow = String(pass.tokens);
        block.title = `${pass.tokens} token${pass.tokens === 1 ? "" : "s"} in this pass`;
        return block;
      }),
    );
  }
  strip.classList.toggle("rv-dense", facts.passes.length > 90);
  facts.passes.forEach((pass, i) => {
    strip.children[i].classList.toggle("rv-on", finished || pass.at <= at);
  });
}

function drawReader(column, race, lane, at, finished, key) {
  const body = column.querySelector(".rv-text");
  const facts = readerFacts(lane);
  let shown = Number(body.dataset.n || 0);
  const target = finished ? lane.pieces.length : countUntil(facts.times, at, shown);
  if (target < shown || body.dataset.kind !== "reader") {
    body.replaceChildren();
    body.dataset.kind = "reader";
    shown = 0;
  }
  for (let i = shown; i < target; i += 1) {
    const piece = lane.pieces[i];
    if (!piece[1]) continue;
    const cls = key === "dspark" ? (piece[2] ? "rv-draft" : "rv-own") : "";
    body.append(make("span", cls, piece[1]));
  }
  body.dataset.n = String(target);
  drawPasses(column, facts, at, finished);
  // the highlighter follows the share of the text read, along the lines' own shares of the transcription
  const read = target === 0 ? 0 : facts.characters[target - 1] / facts.total;
  const shares = race.shares || [];
  inkLines(column, (i) => {
    const from = i === 0 ? 0 : shares[i - 1];
    return (read - from) / Math.max(1e-9, shares[i] - from);
  });
  const meta = column.querySelector(".rv-meta");
  if (finished) {
    const extra = lane.acceptance ? ` · ${lane.acceptance.toFixed(1)} tokens kept per pass` : "";
    meta.textContent = `${lane.tokens} tokens in ${lane.passes} reader passes${extra}${percent(lane.cer)}`;
  } else {
    meta.textContent = `${target} of ${lane.pieces.length} tokens`;
  }
}

function drawTrocr(column, lane, at, finished) {
  const body = column.querySelector(".rv-text");
  const step = finished ? lane.steps.length : countUntil(lane.steps, at, Number(body.dataset.n || 0));
  if (body.dataset.kind !== "trocr" || body.childElementCount !== lane.lines.length) {
    body.replaceChildren(
      ...lane.lines.map((line, i) => {
        const row = make("div", "rv-line");
        row.dataset.line = String(i);
        return row;
      }),
    );
    body.dataset.kind = "trocr";
  }
  const reached = (line) => (step === 0 ? 0 : line.ends[Math.min(step, line.ends.length) - 1]);
  lane.lines.forEach((line, i) => {
    body.children[i].textContent = line.text.slice(0, reached(line)) || " ";
  });
  body.dataset.n = String(step);
  inkLines(column, (i) => (lane.lines[i] ? reached(lane.lines[i]) / Math.max(1, lane.lines[i].text.length) : 0));
  const meta = column.querySelector(".rv-meta");
  meta.textContent = finished
    ? `${lane.lines.length} lines in ${lane.steps.length} decoding steps${percent(lane.cer)}`
    : `step ${step} of ${lane.steps.length}, every line at once`;
}

function candidate(name, key, text, score, exact, picked) {
  const box = make("div", picked ? "rv-cand rv-picked" : "rv-cand");
  box.dataset.lane = key;
  const meter = make("span", "rv-meter");
  const fill = make("i");
  fill.style.width = `${Math.round(100 * score)}%`;
  meter.append(fill);
  box.append(
    make("span", "rv-k", name),
    make("span", "rv-reading", text || "(empty)"),
    meter,
    make("span", "rv-score", score.toFixed(2)),
    make("span", "rv-tag", exact ? "= ground truth" : "≠ ground truth"),
  );
  if (picked) box.append(make("span", "rv-pick", "Jev's pick"));
  return box;
}

function drawJev(root, race) {
  const box = root.querySelector(".rv-jev");
  const rows = race.jev || [];
  if (!race.jev_total) {
    box.replaceChildren();
    return;
  }
  let list = box.querySelector(".rv-jlist");
  if (!list) {
    list = make("div", "rv-jlist");
    const intro =
      "Each line is read on its own by TrOCR and by the MiniCPM reader. Jev looks at the line and at each reading " +
      "drawn as an image, and scores from 0 to 1 whether the reading is exactly what the line says.";
    box.replaceChildren(
      make("h4", "rv-jhead", "Jev, line by line"),
      make("p", "rv-jintro", intro),
      make("p", "rv-jnote"),
      list,
    );
  }
  for (let i = list.childElementCount; i < rows.length; i += 1) {
    const row = rows[i];
    const item = make("div", "rv-jrow");
    item.dataset.line = String(row.index);
    const image = make("img", "rv-jimg");
    image.src = row.image;
    image.alt = `Line ${row.index + 1} of the region`;
    image.loading = "lazy";
    const truth = make("div", "rv-truth");
    truth.append(make("span", "rv-k", "ground truth"), make("span", "rv-reading", row.truth));
    const pickMinicpm = row.minicpm_score > row.trocr_score;
    item.append(
      image,
      truth,
      candidate("TrOCR", "trocr", row.trocr, row.trocr_score, row.trocr_exact, !pickMinicpm),
      candidate("MiniCPM", "plain", row.minicpm, row.minicpm_score, row.minicpm_exact, pickMinicpm),
    );
    list.append(item);
  }
  box.querySelector(".rv-jnote").textContent =
    race.jev_note || `Jev has scored ${rows.length} of ${race.jev_total} lines…`;
}

function draw(now) {
  state.frame = 0;
  const race = payload();
  if (!race) return;
  const root = ensure(race);
  const status = root.querySelector(".rv-status");
  status.textContent = race.status || "";
  status.hidden = !race.status;
  const summary = root.querySelector(".rv-summary");
  summary.textContent = race.summary || "";
  summary.hidden = !race.summary;
  root.querySelector(".rv-controls").hidden = !race.done || still;

  const scale = Math.max(0.001, ...LANES.map((lane) => (race[lane.key] ? race[lane.key].seconds : 0)));
  let running = false;
  for (const lane of LANES) {
    const data = race[lane.key];
    const row = root.querySelector(`.rv-row[data-lane="${lane.key}"]`);
    const column = root.querySelector(`.rv-lane[data-lane="${lane.key}"]`);
    if (!data) {
      const skipped = race.done || (lane.key === "trocr" && race.plain);
      row.querySelector(".rv-time").textContent = skipped ? "not run" : "waiting";
      column.classList.toggle("rv-skipped", Boolean(skipped));
      continue;
    }
    const at = Math.min(Math.max(0, elapsed(lane.key, now)), data.seconds);
    const finished = at >= data.seconds;
    running = running || !finished;
    row.querySelector(".rv-fillbar").style.width = `${(100 * at) / scale}%`;
    row.querySelector(".rv-time").textContent = `${at.toFixed(2)} s`;
    if (lane.key === "trocr") drawTrocr(column, data, at, finished);
    else drawReader(column, race, data, at, finished, lane.key);
  }
  drawJev(root, race);
  if (running) state.frame = requestAnimationFrame(draw);
}

function sync() {
  const race = payload();
  if (!race) return;
  const now = performance.now();
  if (race.run !== state.run) {
    Object.assign(state, { run: race.run, mode: "live", speed: 1, started: {} });
  }
  for (const lane of LANES) {
    if (race[lane.key] && state.started[lane.key] === undefined) state.started[lane.key] = now;
  }
  if (state.frame) cancelAnimationFrame(state.frame);
  draw(now);
}

element.addEventListener("click", (event) => {
  const button = event.target.closest("[data-speed]");
  if (!button) return;
  Object.assign(state, { mode: "race", speed: Number(button.dataset.speed), raceStart: performance.now() });
  if (state.frame) cancelAnimationFrame(state.frame);
  state.frame = requestAnimationFrame(draw);
});

// Pointing at a line (on a manuscript, in TrOCR's text or in Jev's list) lights it up everywhere.
function light(event, on) {
  const target = event.target.closest ? event.target.closest("[data-line]") : null;
  if (!target) return;
  for (const node of element.querySelectorAll(`[data-line="${target.dataset.line}"]`)) {
    node.classList.toggle("rv-hot", on);
  }
}
element.addEventListener("mouseover", (event) => light(event, true));
element.addEventListener("mouseout", (event) => light(event, false));

watch("value", sync);
sync();
