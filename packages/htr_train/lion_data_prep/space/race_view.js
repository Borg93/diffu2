// RaceView: replays three readers from the same start at their measured pace, on the manuscript and as text; shows
// the same region's timings on our server; then lists Jev's scores line by line.
// Gradio runs this once with `element`, `props` and `watch` in scope; `props.value` is a Race payload (app.py).
// Every text is set with textContent: readings are model output and are never parsed as HTML.

const SVG = "http://www.w3.org/2000/svg";
const LANES = [
  { key: "trocr", name: "Swedish Lion TrOCR", how: "the region as its text lines, cut out beforehand; one line after the other" },
  { key: "plain", name: "MiniCPM-OCR reader alone", how: "the region as one image; one token per reader pass" },
  { key: "dspark", name: "Reader + DSpark draft", how: "the region as one image; up to 16 tokens per reader pass" },
];
// The stretches timed here: each lane's own, and TrOCR's lines once more as one batch.
const BARS = [
  { key: "trocr", lane: "trocr", name: "TrOCR · one line after the other", seconds: (race) => race.trocr && race.trocr.seconds },
  { key: "batch", lane: "trocr", name: "TrOCR · its lines as one batch", seconds: (race) => race.trocr && race.trocr.batch_seconds },
  { key: "plain", lane: "plain", name: "MiniCPM-OCR reader alone", seconds: (race) => race.plain && race.plain.seconds },
  { key: "dspark", lane: "dspark", name: "Reader + DSpark draft", seconds: (race) => race.dspark && race.dspark.seconds },
];
// The same stretches on our server (Race.server), the reader served by vLLM.
const SERVER = [
  { field: "trocr_single", lane: "trocr", name: "TrOCR · one line after the other" },
  { field: "trocr_batch", lane: "trocr", name: "TrOCR · its lines as one batch" },
  { field: "vllm", lane: "plain", name: "Reader alone · in vLLM" },
  { field: "vllm_dspark", lane: "dspark", name: "Reader + DSpark · in vLLM" },
];
const SPEEDS = [
  { label: "▶ Race them again", speed: 1, main: true },
  { label: "½ speed", speed: 0.5 },
  { label: "¼ speed", speed: 0.25 },
];
const JEV_INTRO =
  "Each line is read on its own by TrOCR and by the MiniCPM-OCR reader. Marked characters differ from the ground " +
  "truth (the line as a person transcribed it); ‸ stands for missing ones. Jev looks at the line and at a reading " +
  "drawn as an image, and scores from 0 to 1 how sure it is that the reading is exactly what the line says. Where " +
  "the two readings differ, the one Jev scores higher is its pick. Jev catches a reading that is plainly another " +
  "text far better than one that is a letter or two off (see How it works), so a high score does not prove a " +
  "reading right.";
const still = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const state = { run: null, raced: false, speed: 1, raceStart: 0, frame: 0 };
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

function write(root, selector, text) {
  const node = root.querySelector(selector);
  node.textContent = text || "";
  node.hidden = !text;
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

function barRow(key, lane, name) {
  const row = make("div", "rv-row");
  row.dataset.bar = key;
  row.dataset.lane = lane;
  const label = make("div", "rv-name");
  label.append(make("i", "rv-swatch"), make("span", "", name));
  const bar = make("div", "rv-bar");
  bar.append(make("div", "rv-fillbar"));
  row.append(label, bar, make("div", "rv-time"));
  return row;
}

function lanesOf(race) {
  const lanes = make("div", "rv-lanes");
  for (const lane of LANES) {
    const column = make("section", "rv-lane");
    column.dataset.lane = lane.key;
    const head = make("header", "rv-head");
    head.append(make("i", "rv-swatch"), make("b", "", lane.name));
    const body = make("div", "rv-text");
    body.dataset.n = "0";
    column.append(head, make("div", "rv-how", lane.how), make("div", "rv-meta", " "), page(race, lane.key));
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
  return lanes;
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

  const live = make("section", "rv-group rv-live");
  const track = make("div", "rv-track");
  for (const bar of BARS) track.append(barRow(bar.key, bar.lane, bar.name));
  const controls = make("div", "rv-controls");
  controls.hidden = true;
  for (const choice of SPEEDS) {
    const button = make("button", choice.main ? "rv-button rv-main" : "rv-button", choice.label);
    button.type = "button";
    button.dataset.speed = String(choice.speed);
    controls.append(button);
  }
  const timed = make("div", "rv-timed");
  timed.append(track, controls);
  const split = make("div", "rv-split");
  split.append(timed, make("p", "rv-summary"));
  live.append(make("h4", "rv-title", "Here, live: one stream each, in PyTorch on half a GPU"), split);

  const server = make("section", "rv-group rv-server");
  const serverTrack = make("div", "rv-track");
  for (const bar of SERVER) serverTrack.append(barRow(bar.field, bar.lane, bar.name));
  server.append(make("h4", "rv-title"), serverTrack, make("p", "rv-said"), make("p", "rv-note"));

  const batched = make("section", "rv-group rv-batched");
  const batchedTitle = "Everything batched, on our server: lines read a second";
  batched.append(make("h4", "rv-title", batchedTitle), make("div", "rv-tiles"), make("p", "rv-note"));

  const pair = make("div", "rv-pair");
  pair.append(server, batched);
  root.replaceChildren(
    make("div", "rv-notice"),
    make("div", "rv-status"),
    live,
    lanesOf(race),
    pair,
    make("section", "rv-group rv-jev"),
  );
  return root;
}

// Seconds into the replay: none before every reader is timed, all of them where motion is unwelcome.
function elapsed(now) {
  if (!state.raced) return 0;
  if (still) return Infinity;
  return Math.max(0, ((now - state.raceStart) / 1000) * state.speed); // a frame's clock can precede the start
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

// TrOCR reads its lines one after the other: each line's steps carry their own times.
function drawTrocr(column, lane, at, finished) {
  const body = column.querySelector(".rv-text");
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
  let current = lane.lines.length;
  const reached = lane.lines.map((line, i) => {
    if (finished) return line.text.length;
    const steps = countUntil(line.steps, at, 0);
    if (steps < line.steps.length && current === lane.lines.length) current = i;
    return steps === 0 ? 0 : line.ends[steps - 1];
  });
  lane.lines.forEach((line, i) => {
    body.children[i].textContent = line.text.slice(0, reached[i]) || " ";
  });
  inkLines(column, (i) => (lane.lines[i] ? reached[i] / Math.max(1, lane.lines[i].text.length) : 0));
  const meta = column.querySelector(".rv-meta");
  if (finished) {
    const steps = lane.lines.reduce((sum, line) => sum + line.steps.length, 0);
    meta.textContent = `${lane.lines.length} lines in ${steps} decoding steps${percent(lane.cer)}`;
  } else {
    meta.textContent = `line ${Math.min(current + 1, lane.lines.length)} of ${lane.lines.length}`;
  }
}

function drawServer(root, race, scale) {
  const box = root.querySelector(".rv-server");
  const server = race.server;
  box.hidden = !server;
  root.querySelector(".rv-pair").classList.toggle("rv-single", !server);
  if (server) {
    box.querySelector(".rv-title").textContent =
      `The same region on our server: a whole ${server.gpu}, the reader served by vLLM`;
    for (const bar of SERVER) {
      const row = box.querySelector(`[data-bar="${bar.field}"]`);
      row.querySelector(".rv-fillbar").style.width = `${(100 * server[bar.field]) / scale}%`;
      row.querySelector(".rv-time").textContent = `${server[bar.field].toFixed(2)} s`;
    }
    box.querySelector(".rv-said").textContent = race.server_said || "";
    box.querySelector(".rv-note").textContent =
      `Measured beforehand (${server.measured}): one request at a time, each image new to the server, the median ` +
      `of ${server.runs} runs. This Space gets half of such a GPU and decodes in PyTorch, since ZeroGPU cannot ` +
      "host vLLM. These bars and the live ones share one scale.";
  }
  const batched = root.querySelector(".rv-batched");
  const rates = race.batched || [];
  batched.hidden = !rates.length;
  const tiles = batched.querySelector(".rv-tiles");
  if (tiles.childElementCount !== rates.length) {
    tiles.replaceChildren(
      ...rates.map((rate) => {
        const tile = make("div", "rv-tile");
        tile.dataset.lane = rate.lane;
        const what = make("div", "rv-what");
        what.append(make("div", "rv-tilename", rate.name), make("div", "rv-how", rate.how));
        const figure = make("div", "rv-rate", rate.per_second.toFixed(rate.per_second < 10 ? 1 : 0));
        figure.append(make("span", "rv-unit", " lines a second"));
        tile.append(what, figure);
        return tile;
      }),
    );
  }
  batched.querySelector(".rv-note").textContent = race.batched_note || "";
}

function reading(name, key, data, picked) {
  const row = make("div", picked ? "rv-jr rv-picked" : "rv-jr");
  row.dataset.lane = key;
  const text = make("span", "rv-reading");
  for (const [piece, agrees] of data.marks) text.append(make("span", agrees ? "" : "rv-bad", piece));
  if (!data.marks.length) text.textContent = "(empty)";
  const meter = make("span", "rv-meter");
  const fill = make("i");
  fill.style.width = `${Math.round(100 * data.score)}%`;
  meter.append(fill);
  const score = make("span", "rv-scorebox");
  score.append(meter, make("span", "rv-score", data.score.toFixed(3)));
  row.append(
    make("span", "rv-k", name),
    text,
    make("span", data.exact ? "rv-right" : "rv-wrong", data.exact ? "✓ right" : "✗ wrong"),
    score,
    make("span", "rv-pick", picked ? "Jev's pick" : ""),
  );
  return row;
}

function drawJev(root, race) {
  const box = root.querySelector(".rv-jev");
  const rows = race.jev || [];
  box.hidden = !race.jev_total;
  if (!race.jev_total) return;
  let list = box.querySelector(".rv-jlist");
  if (!list) {
    list = make("div", "rv-jlist");
    const header = make("div", "rv-jr");
    for (const name of ["Reader", "Its reading of the line", "Right?", "Jev's score, 0 to 1", ""]) {
      header.append(make("span", "", name));
    }
    const head = make("div", "rv-jrow rv-jhead");
    head.append(make("div", "rv-jlinehead", "The line, and what Jev's scores amount to"), header);
    const fold = make("details", "rv-fold");
    fold.append(make("summary", "rv-foldhead"), make("p", "rv-jintro", JEV_INTRO), head, list);
    box.replaceChildren(make("h4", "rv-title", "Jev, the reader's proof-reader"), make("p", "rv-jnote"), fold);
  }
  for (let i = list.childElementCount; i < rows.length; i += 1) {
    const row = rows[i];
    const item = make("div", "rv-jrow");
    item.dataset.line = String(row.index);
    const line = make("div", "rv-jline");
    const image = make("img", "rv-jimg");
    image.src = row.image;
    image.alt = `Line ${row.index + 1} of the region`;
    image.loading = "lazy";
    const verdict = make("span", "rv-verdict", row.verdict);
    verdict.dataset.kind = row.verdict.startsWith("Jev picks the right") ? "good" : row.verdict === "both right" ? "" : "bad";
    line.append(image, verdict);
    const truth = make("div", "rv-jr rv-truth");
    truth.append(make("span", "rv-k", "ground truth"), make("span", "rv-reading", row.truth));
    const readings = make("div", "rv-jreads");
    readings.append(
      truth,
      reading("TrOCR", "trocr", row.trocr, row.pick === "trocr"),
      reading("MiniCPM-OCR", "plain", row.minicpm, row.pick === "minicpm"),
    );
    item.append(line, readings);
    list.append(item);
  }
  box.querySelector(".rv-jnote").textContent =
    race.jev_note || `Jev has scored ${rows.length} of ${race.jev_total} lines…`;
  box.querySelector(".rv-foldhead").textContent =
    `Line by line: both readings and Jev's scores (${rows.length} of ${race.jev_total} lines)`;
}

function draw(now) {
  state.frame = 0;
  const race = payload();
  if (!race) return;
  const root = ensure(race);
  write(root, ".rv-status", race.status);
  write(root, ".rv-notice", race.notice);
  write(root, ".rv-summary", race.summary);
  root.querySelector(".rv-controls").hidden = !race.ready || still;

  const server = race.server;
  const known = BARS.map((bar) => bar.seconds(race) || 0);
  if (server) known.push(...SERVER.map((bar) => server[bar.field]));
  const scale = Math.max(0.001, ...known);
  const at = elapsed(now);
  let running = false;
  for (const bar of BARS) {
    const row = root.querySelector(`.rv-live [data-bar="${bar.key}"]`);
    const total = bar.seconds(race);
    const fill = row.querySelector(".rv-fillbar");
    const time = row.querySelector(".rv-time");
    if (!total || !state.raced) {
      const skipped = !total && (race.done || (bar.lane === "trocr" && race.plain));
      fill.style.width = "0";
      time.textContent = total ? "timed" : skipped ? "not run" : "waiting";
      continue;
    }
    const shown = Math.min(at, total);
    running = running || shown < total;
    fill.style.width = `${(100 * shown) / scale}%`;
    time.textContent = `${shown.toFixed(2)} s`;
  }
  for (const lane of LANES) {
    const data = race[lane.key];
    const column = root.querySelector(`.rv-lane[data-lane="${lane.key}"]`);
    column.classList.toggle("rv-skipped", !data && Boolean(race.done || (lane.key === "trocr" && race.plain)));
    if (!data) continue;
    if (!state.raced) {
      column.querySelector(".rv-meta").textContent = "timed; the replay starts when all are";
      continue;
    }
    const shown = Math.min(at, data.seconds);
    if (lane.key === "trocr") drawTrocr(column, data, shown, shown >= data.seconds);
    else drawReader(column, race, data, shown, shown >= data.seconds, lane.key);
  }
  drawServer(root, race, scale);
  drawJev(root, race);
  if (running) state.frame = requestAnimationFrame(draw);
}

function sync() {
  const race = payload();
  if (!race) return;
  if (race.run !== state.run) Object.assign(state, { run: race.run, raced: false, speed: 1 });
  // every reader is timed: they start together
  if (race.ready && !state.raced) Object.assign(state, { raced: true, speed: 1, raceStart: performance.now() });
  if (state.frame) cancelAnimationFrame(state.frame);
  draw(performance.now());
}

element.addEventListener("click", (event) => {
  const button = event.target.closest("[data-speed]");
  if (!button) return;
  Object.assign(state, { raced: true, speed: Number(button.dataset.speed), raceStart: performance.now() });
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
