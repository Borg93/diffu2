# Diffu — what we're doing (in plain words)

> A jargon-free companion to the technical [`ARCHITECTURE.md`](ARCHITECTURE.md).
> If you read one page, read this one.

## The one-sentence goal

We're building a machine that **invents realistic Swedish handwriting images** — so we can
create *unlimited* practice material to train **other** AI that **reads** old handwritten documents.

## Why bother?

- AI that reads handwriting (called **HTR**) learns from examples: an image of a handwritten line
  **+ what it says** (the typed-out text).
- Real old Swedish handwriting (e.g. 1800s Göteborg police records) *with* typed-out text is
  **scarce and expensive** to produce.
- If we can **generate** thousands of fake-but-realistic handwritten lines whose text we already
  know, we give the reader-AI far more to practice on → it reads the *real* documents better.
- **Success = the reader-AI makes fewer mistakes** (a lower "Character Error Rate") after training
  on our fakes. That scoreboard — not whether the images look pretty — is the only thing that counts.

## What the machine takes in and puts out

- **You give it:** (1) some **text** to write (e.g. `Göteborg`), and (2) a **style sample** — a small
  snippet of someone's handwriting to imitate.
- **It gives back:** an image of that text, written in that handwriting style, with correct Swedish
  letters (å ä ö).

## How does it learn? (the simple version)

Think of teaching an art student to forge handwriting:

1. We take a **real** handwritten line and **smear it into random noise**.
2. The student (the AI) practices **removing the noise, step by step, to get the handwriting back** —
   while being told *what text* it should say and *whose style* to copy.
3. Repeat on millions of real examples → the student learns to start from **pure noise** and
   **paint brand-new handwriting on command**.

That "remove noise to reveal an image" trick is the whole idea behind a **diffusion model**.
Everything else is just *telling it what to write and in which style*.

## Where are we now?

- Current state always lives in [`STATUS.md`](STATUS.md).
- The released model is [`Gabriel/diffu_sd35_128`](https://huggingface.co/Gabriel/diffu_sd35_128) —
  it writes legible Swedish handwriting that a recognizer reads back at ~2–5 % character error.
