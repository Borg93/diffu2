### What is timed

All three read the same region, one after the other on the same GPU (ZeroGPU: half an RTX PRO 6000 Blackwell). Each
is timed from the images to the text, preprocessing included, after a warm-up that is not timed:

- **Swedish Lion TrOCR** (Riksarkivet's Swedish Lion Libre 2026, large, character-based) cannot read a region. It
  gets the region as its text lines, cut out beforehand along the ground truth's line outlines, and reads them all in
  one batch, greedily, in float32 as Riksarkivet runs it. Finding the lines is a model of its own in practice; that
  step is not timed here.
- **MiniCPM-OCR reader alone** reads the region as one image, one token per pass, greedily.
- **Reader + DSpark draft**: the draft proposes 15 tokens at a time; the reader checks them in one pass and keeps the
  ones it agrees with, plus one of its own. Greedy decoding, so the text is the reader's own.

### What you see

Each reader notes when every token became known, and the page replays its reading at that pace: first as each reading
arrives, then side by side from the replay buttons. Nothing is sent to the page inside a timed stretch, so the display
slows no reader down.

- On each manuscript a highlighter follows the reader: TrOCR's moves along every line at once (a decoding step adds
  one character to each line of the batch), the reader's goes line by line.
- The strip above a reader's text has one block per reader pass, as wide as the tokens that pass produced: many thin
  ones alone, a few wide ones with the draft.
- In the DSpark lane, shaded text was proposed by the draft and kept; underlined text is the reader's own token, one
  per pass. Long shaded runs are where the time is saved.
- Point at a line (on a manuscript, in TrOCR's text or in Jev's list) to see it lit up everywhere.

### Why the draft makes it faster

Decoding one token at a time runs the whole reader for each token. A check of 16 tokens costs about the same pass, and
the reader keeps about 8 of them on average (the acceptance length). The draft is small (5 layers) and reads the
reader's own hidden states, so its guesses are cheap and usually right. It helps most when one region is read at a time
and its text is long; reading many requests at once keeps the GPU busy anyway.

### How it runs here

vLLM runs the draft natively, but vLLM cannot run on ZeroGPU, so this Space decodes in PyTorch (`dspark_hf.py`).
Profiled with `torch.profiler`, eager decoding was launch-bound: about 1,500 small kernels a pass, the GPU busy 12 % of
the time. So:

1. Each reader pass is replayed from a **CUDA graph** (transformers' StaticCache, the attention mask built on the GPU),
   for the reader alone and with the draft alike, which removes most of that launch overhead.
2. The draft's whole proposal, its Markov sampling included, is **one graph**.
3. **One reader pass per draft block.** The reader is hybrid: its linear-attention layers keep a recurrent state that
   cannot be cut back after rejected guesses. The checking pass keeps each such layer's inputs for the block, and the
   state is rebuilt over the accepted tokens only (a few kernels instead of a second pass).

Measured on our server (a full RTX PRO 6000 with nothing else on it, 12 dev regions and 20 test lines): the regions
took 12.4 s with the reader alone and 3.2 s with the draft (**3.9×**); the lines 2.8 s and 2.2 s (1.3×). The reader
alone writes the same tokens as transformers' own `generate`.

vLLM is still faster than this Space's PyTorch: on the same GPU and regions it took 5.9 s with the reader alone and
2.2 s with the draft, about 2× and 1.4× faster.

### The same models in vLLM

On 1,000 test lines and 200 dev regions, one GPU per server (the draft's model card):

| | one at a time | 64 at a time |
|---|---|---|
| lines: reader alone → with DSpark | 13.3 → 18.2 lines/s (1.37×) | 74.6 → 82.2 lines/s (1.10×) |
| regions: reader alone → with DSpark | 1.95 → 4.08 regions/s (2.09×) | 12.4 → 13.2 regions/s (1.07×) |

Acceptance length 8.72; the character error rate is the same with and without the draft.

### Jev

Jev compares a line's image with a candidate reading drawn as an image (`@RENDERER@`) and returns a score from 0 to 1.
Here each line is read on its own by TrOCR and by the MiniCPM reader, and Jev scores both; keeping, line by line, the
reading Jev scores higher is one way to combine two readers.

Jev catches gross mismatches well and near-misses badly. On 2,000 test pairs it tells the correct text from rule-made
wrong candidates (dropped or swapped words, another line's text) with an AUROC of 0.988, but from the reader's own
wrong readings, a letter or two off, with only 0.624: those were 1.2 % of its training pairs. So a high score here
does not prove a reading right.

### Limits

- The examples are from GT-Lejonet's dev volumes; other hands and periods read worse.
- Near-greedy decoding can repeat a phrase to the token limit on dense pages (a loop); such a reading is wrong from
  where the loop starts.
- Timings vary with the GPU slice and the queue. A fresh GPU worker captures its CUDA graphs first (a few seconds, not
  counted in the timings shown).
