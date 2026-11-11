### What is timed

All three read the same region, one after the other on the same GPU (ZeroGPU: half an RTX PRO 6000 Blackwell), with
**one stream each and nothing batched**. Each is timed from the images to the text, preprocessing included, after a
warm-up that is not timed:

- **Swedish Lion TrOCR** (Riksarkivet's Swedish Lion Libre 2026, large, character-based) cannot read a region. It
  gets the region as its text lines, cut out beforehand along the ground truth's line outlines, and reads them one
  after the other, greedily, in float32 as Riksarkivet runs it. Finding the lines is a model of its own in practice;
  that step is not timed here.
- **MiniCPM-OCR reader alone** reads the region as one image, one token per pass, greedily.
- **Reader + DSpark draft**: the draft proposes 15 tokens at a time; the reader checks them in one pass and keeps the
  ones it agrees with, plus one of its own. Greedy decoding, so the text is the reader's own.

### Comparing them fairly

A batch lets a GPU decode many sequences side by side. TrOCR can batch a region's lines. The reader reads a region as
one sequence, which has nothing to batch; what it batches is whole regions, in vLLM, which this Space cannot run.
Racing a batch of 8 to 18 lines against one sequence would compare two different things, so there are three views,
each with one rule for everyone:

- **Nothing batched** (the race): TrOCR reads its lines one after the other, the reader one region.
- **TrOCR's lines as one batch** is timed too and drawn as the second bar, so nothing is hidden: on these regions it
  is about as fast as the reader alone in this Space's PyTorch, and 2 to 4 times slower than the reader with its
  draft.
- **Everything batched** (the three rates under the race): on our server, TrOCR in batches of 32 lines against the
  reader with 64 requests at a time in vLLM, which reads about ten times as many lines a second.

### What you see

Each reader notes when every token became known. Once all three are timed, the page replays their readings from the
same start, each at its measured pace; the buttons replay it, also at half and quarter speed. Nothing is sent to the
page inside a timed stretch, so the display slows no reader down.

- On each manuscript a highlighter follows the reader: TrOCR's goes down the lines one by one, a character per
  decoding step; the reader's follows its tokens through the region.
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

### The same region on our server, and vLLM

The outlined bars under the race were measured beforehand on our server (`measure_reference.py`, `reference.json`):
one whole RTX PRO 6000 Blackwell with nothing else running on it, TrOCR in transformers as here, the reader served by
vLLM alone and with its draft, one request at a time, the median of 5 runs. vLLM keeps the processed form of an image
it has seen, which saves 0.05 to 0.1 s a region, so every request sent the region with one pixel changed by one
level: new to the server, as a region is in production. Over the six example regions:

| a region read by | seconds |
|---|---|
| TrOCR, one line after the other | 1.78 – 3.52 |
| TrOCR, its lines as one batch | 0.72 – 1.86 |
| the reader in vLLM | 0.50 – 0.74 |
| the reader with its draft in vLLM | 0.32 – 0.50 |

The reader in vLLM is 1.4 to 2.6 times as fast as TrOCR's one batch, and 2.3 to 3.7 times with its draft. On six
other regions and the same whole GPU, this Space's PyTorch decoding was as fast as vLLM with the draft and 1.2 to 1.8
times slower alone; the live bars are slower again because ZeroGPU gives half a GPU. The reader alone writes the same
tokens as transformers' own `generate`.

What vLLM adds is reading many requests at once. On the first 1,000 test lines and 200 dev regions, one RTX PRO
6000 Blackwell:

| | one at a time | many at a time |
|---|---|---|
| lines: Swedish Lion TrOCR | 4.3 lines/s (batch 1) | 6.5 lines/s (batch 32) |
| lines: reader alone → with DSpark | 13.3 → 18.2 lines/s (1.37×) | 67 → 77 lines/s (64 at a time) |
| regions: reader alone → with DSpark | 1.95 → 4.08 regions/s (2.09×) | 11.9 → 12.7 regions/s (64 at a time) |

The 64-at-a-time rows are from a run where every image was new to the servers (two servers on one GPU, asked in
turn); with each server on a GPU of its own they were 75 → 82 lines/s and 12.4 → 13.2 regions/s. Acceptance length
8.72; the character error rate is the same with and without the draft. OpenBMB's deployment guide also serves the
reader and a DSpark draft with SGLang (through a pull request still open there in October 2026); we have measured
vLLM only.

### Jev

Jev compares a line's image with a candidate reading drawn as an image (`@RENDERER@`) and returns a score from 0 to 1.
Here each line is read on its own by TrOCR and by the MiniCPM reader, and Jev scores both; keeping, line by line, the
reading Jev scores higher is one way to combine two readers. A reading's errors are the characters it adds, drops or
gets wrong against the ground truth. The fold under the race is titled with what Jev did: on how many of the lines
where one reading has fewer errors it picked that one. Inside, the character error rate over the region of each
reader, of Jev's picks, and of the best possible picks (the reading with fewer errors on every line, the most any
chooser could get); then one row per line: each reading with its wrong characters marked, its errors and Jev's score
in percent, Jev's pick shaded and marked ✓ when it has fewer errors than the other reading, ✗ when it has more. Click
a row to see the line.

Jev catches gross mismatches well and near-misses badly. On 2,000 test pairs it tells the correct text from rule-made
wrong candidates (dropped or swapped words, another line's text) with an AUROC of 0.988, but from the reader's own
wrong readings, a letter or two off, with only 0.624: those were 1.2 % of its training pairs. So a high score here
does not prove a reading right.

### Limits

- The examples are drawn at random (seeded, one each from six collections) from GT-Lejonet's test pages, which no
  model here trained on. Our dev volumes would not do: they are cut out of the training pages, which TrOCR trained on
  (6 epochs) and our reader did not. TrOCR chose its checkpoint on the test pages, which flatters it a little. Other
  hands and periods read worse.
- TrOCR gets each line cut along the ground truth's outline. In practice a line detector finds the lines, and its
  mistakes would count against TrOCR; here they cannot.
- Near-greedy decoding can repeat a phrase to the token limit on dense pages (a loop); such a reading is wrong from
  where the loop starts.
- Timings vary with the GPU slice and the queue. A fresh GPU worker captures its CUDA graphs first (a few seconds, not
  counted in the timings shown).
- With the draft the reader's text can differ from its text alone by a token where two tokens were nearly tied: a
  check of 16 tokens rounds differently from a pass of one. vLLM shows the same between 1 and 64 requests at a time.
