---
title: diffu archive pages
emoji: 📜
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
python_version: "3.12"
short_description: Draw archive pages from boxes, text and a sample page
license: apache-2.0
startup_duration_timeout: 1h
models:
  - Gabriel/diffu-moss-above
  - OpenMOSS-Team/MOSS-Image-Base
  - Riksarkivet/trocr-base-handwritten-hist-swe-2
---

# diffu archive pages

Draws a synthetic archive page from a layout and a sample page, with
[Gabriel/diffu-moss-above](https://huggingface.co/Gabriel/diffu-moss-above), diffu's fine-tune of MOSS-Image for Swedish
archival handwriting. The page comes back with its labels: ALTO and PAGE XML of every line, its text and its place.

1. **Lay out the page.** Start from a template (a letter, court minutes, a register, a certificate) or drag boxes
   onto a blank page. A box is a block of lines: text, a heading, a margin note or a page number; a ruled line is a
   box too. Every line of a page is drawn at one height, which the Lines slider sets.
2. **Write its text.** Select a box and write. The page shows how the text falls into the box's lines, and how much
   has no room. A line break of yours starts a new line.
3. **Give a sample page.** A scan of handwriting whose paper, ink and hand the page takes: one of the examples, or
   your own.
4. **Draw.** The model writes the text you gave it, line for line. Every drawn line is then read back by an HTR
   model, and a line that does not read back as written is tagged `unclear` in the labels.

## What to expect

Measured on the templates, drawn fast at 2 megapixels from the four sample pages (20 Nov 2026):

- **The text is written as given.** Of each page's lines, 13 of 13 to 26 of 27 read back as written.
- **The paper and the kind of scan follow the sample**: colour or greyscale, the paper's tone.
- **The ink and the pen follow it only in part.** The ink comes out darker and the pen heavier than the sample's,
  most of all from a sample in faint ink.
- **The hand mostly does not.** The model writes a hand of the period, and its letter forms change little from one
  sample to the next.
- **Long ruled lines are drawn** (a register's columns); a short line under a word often is not, and printed text
  comes out handwritten.
- The model was trained on Swedish archival handwriting of the 16th to 19th centuries at 2 megapixels. Other
  languages, modern hands and smaller pages are outside what it was trained on; a 1 megapixel page draws faster and
  a little less surely.
- A page's prompt holds every line's text and box, and the text encoder reads 4,096 tokens: about sixty lines.
- **Fast** draws with a distilled student of the model, in 8 steps without guidance; it is experimental. **Guided**
  draws with the model itself and takes several times the GPU time. The status line under the button says how much
  GPU time a page will ask for.

## How it is built

- `app.py`: the Gradio app. The page is planned on CPU, drawn on the GPU (ZeroGPU), read back and labelled.
- `pagedemo/editor.js`, `editor.html`, `editor.css`: the page editor, a Gradio custom HTML component.
- `pagedemo/document.py`, `planning.py`, `sample.py`: the editor's page as the layout, the prompt and the sample the
  model was trained on, through diffu's own page pipeline.
- `pagedemo/drawing.py`: the model, loaded with diffu's pipeline at pinned revisions.

The code is in [Borg93/diffu2](https://github.com/Borg93/diffu2) (`packages/studio/space`), with the pipeline it
runs on.

## Sample pages

The example samples are cut from scans in the Swedish National Archives' (Riksarkivet) public HTR datasets on the Hub:

- `gota-hovratt-1687.jpg`, `gota-hovratt-1701.jpg`: Göta hovrätt, Brottsmålsprotokoll, 1687 and 1701-1706
  ([Riksarkivet/gota_hovratt_seg](https://huggingface.co/datasets/Riksarkivet/gota_hovratt_seg))
- `standernas-deputation-1765.jpg`: Frihetstidens utskott, Riksens ständers stora deputations handlingar, volume 2,
  1765-1766
  ([Riksarkivet/frihetstidens_utskottshandlingar](https://huggingface.co/datasets/Riksarkivet/frihetstidens_utskottshandlingar))
- `jonkopings-radhusratt-1584-1595.jpg`: Jönköpings rådhusrätts och magistrats arkiv, Domböcker 1, 1584-1595
  ([Riksarkivet/jonkopings_radhusratts_och_magistrat_seg](https://huggingface.co/datasets/Riksarkivet/jonkopings_radhusratts_och_magistrat_seg))
