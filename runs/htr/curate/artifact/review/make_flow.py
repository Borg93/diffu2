"""The curation flow as one inline SVG (flow.svg), drawn on a grid; styled by the page's classes."""
W, MX, MW, SX, SW = 780, 20, 430, 510, 250
parts = []
def box(x, y, w, h, cls): parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="{cls}"/>')
def text(x, y, s, cls="f-t", anchor="start"): parts.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{s}</text>')
def arrow(x1, y1, x2, y2, label=None, cls="f-a", lx=None, ly=None, la="start"):
    parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{cls}"/>')
    if x1 == x2: parts.append(f'<polygon points="{x2 - 5},{y2 - 8} {x2 + 5},{y2 - 8} {x2},{y2}" class="{cls}-h"/>')
    elif x2 > x1: parts.append(f'<polygon points="{x2 - 8},{y2 - 5} {x2 - 8},{y2 + 5} {x2},{y2}" class="{cls}-h"/>')
    else: parts.append(f'<polygon points="{x2 + 8},{y2 - 5} {x2 + 8},{y2 + 5} {x2},{y2}" class="{cls}-h"/>')
    if label: text(lx if lx is not None else x1 + 8, ly if ly is not None else (y1 + y2) / 2 + 4, label, "f-l", la)
def chip(x, y, label, cls):
    w = 9 + 6.4 * len(label); box(x - w, y, w, 18, f"f-chip {cls}"); text(x - w / 2, y + 13, label, f"f-chip-t {cls}", "middle")
def stage(n, y, title, lines, status, cls, h=70):
    box(MX, y, MW, h, "f-box")
    parts.append(f'<circle cx="{MX + 22}" cy="{y + 22}" r="12" class="f-num"/>'); text(MX + 22, y + 26.5, str(n), "f-num-t", "middle")
    text(MX + 42, y + 27, title, "f-h")
    for k, line in enumerate(lines): text(MX + 42, y + 46 + 16 * k, line, "f-s")
    chip(MX + MW - 10, y + 10, status, cls)
def side(y, title, line, cls, h=48):
    box(SX, y, SW, h, f"f-side {cls}"); text(SX + 12, y + 20, title, "f-h2"); text(SX + 12, y + 37, line, "f-s")
cx = MX + MW / 2
# the store
box(MX, 10, MW, 50, "f-store"); text(MX + 16, 31, "3,154,257 scans in two archives", "f-h"); text(MX + 16, 49, "each with the old YOLO + TrOCR labels (htrflow, 2024)", "f-s")
arrow(cx, 60, cx, 92, "every scan, two image models", lx=cx + 10)
stage(1, 92, "Describe every scan", ["two image models give each scan a vector;", "small classifiers turn it into columns"], "done", "c-done")
side(92, "Your 17,686 marks", "and a VL model's answers on 100,000", "c-in", 52); arrow(SX, 118, MX + MW, 118, "teach", lx=SX - 8, ly=112, la="end")
arrow(cx, 162, cx, 196, "class, table, print, language, cluster …", lx=cx + 10)
stage(2, 196, "Decide which scans to use", ["an opening of handwritten running text?", "kept: 2,245,502 scans"], "done", "c-done")
side(186, "Tables · 331,109 scans", "kept apart for a table model", "c-apart"); arrow(MX + MW, 210, SX, 210)
side(242, "Odd, print, single, no page", "left out (about 400,000)", "c-out"); arrow(MX + MW, 250, SX, 266)
arrow(cx, 266, cx, 300, "openings to cut", lx=cx + 10)
stage(3, 300, "Cut each opening at its fold", ["a placer finds the dark column between the pages", "4,237,487 single pages built"], "done", "c-done")
arrow(cx, 370, cx, 404, "single pages", lx=cx + 10)
stage(4, 404, "Read every page with our reader", ["Nemotron Parse, fine-tuned on your ground truth", "1,139,745 read so far · 13.6 pages a second"], "1 of 4 sets", "c-run")
arrow(cx, 474, cx, 508, "page image + reader's text + old label", lx=cx + 10)
stage(5, 508, "Check every page", ["reader against old label · loops · turned · size", "writing layer and Jev: being measured"], "running", "c-run")
side(498, "Turned · 815 pages", "turn back, read again", "c-apart"); arrow(MX + MW, 522, SX, 522)
side(554, "Looped, near-empty", "left out (about 5 %)", "c-out"); arrow(MX + MW, 562, SX, 578)
arrow(cx, 578, cx, 612, "pages that pass", lx=cx + 10)
# the choice
box(MX, 612, MW, 104, "f-box"); parts.append(f'<circle cx="{MX + 22}" cy="{634}" r="12" class="f-num"/>'); text(MX + 22, 638.5, "6", "f-num-t", "middle")
text(MX + 42, 639, "Choose each page's label", "f-h"); chip(MX + MW - 10, 622, "being tested", "c-test")
box(MX + 14, 654, 196, 50, "f-opt"); text(MX + 26, 674, "the old label", "f-h2"); text(MX + 26, 692, "tried: 6.92 %", "f-s")
box(MX + 220, 654, 196, 50, "f-opt c-lean"); text(MX + 232, 674, "our reader's reading", "f-h2"); text(MX + 232, 692, "E60: result Tuesday", "f-s")
side(640, "Blind review, 107 pages", "reader better on 60, label on 5", "c-in", 52); arrow(SX, 666, MX + MW, 666)
arrow(cx, 716, cx, 750, "pool pages with their labels", lx=cx + 10)
stage(7, 750, "Train, then test", ["pretrain on pool pages, fine-tune on 49,491", "GT pages, score 200 test pages: best 6.59 %"], "repeats", "c-done")
H = 830
svg = (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="The curation flow: scans are described by small classifiers, chosen, cut at the fold, read by our reader, checked, given a label, and used to pretrain before the fine-tune on ground truth." class="flow">' + "".join(parts) + "</svg>")
open("flow.svg", "w").write(svg); print(len(svg), "characters")
