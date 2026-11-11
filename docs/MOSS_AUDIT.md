# MOSS-Image: the audit of 11 November 2026

Why the page's look follows the sample page and its handwriting does not: every part of the training and the drawing
checked against the code and against measurements, the one fault found, and what follows from it. The runs it speaks
of are in [MOSS_RUNS.md](MOSS_RUNS.md); how they are run, in [MOSS_HPC.md](MOSS_HPC.md).

## In plain words

Think of the model as a scribe. He is handed a sample page and a text, and asked for a new page that looks as if the
sample's writer had written it. What comes back has the sample's paper and ink, and the text is right, but the
handwriting is the scribe's own tidy hand, whatever sample he was given.

The audit asked three things: was he taught wrongly, does he look at the sample at all, and what would make him
copy the hand.

- **One real fault in how the lessons were dealt out, now fixed.** Each lesson is a page in one of three forms: with
  the sample alone, with the sample and a few words on the look, or without a sample. The forms were meant to fall at
  random. They did not: whole pages always came without a sample (or with the look words), and the sample alone was
  only ever shown with small windows of a page. The planner is fixed and the 2 MP lesson set is built again.
- **The teaching itself is right.** The noise he learns to remove, what he is asked to predict and how he is marked
  are the same as in MOSS-Image's own trainer, line for line.
- **He glances at the sample once, at the very start.** While a page is still almost all noise he takes the paper
  and the ink from the sample; after that, swapping, blurring or removing the sample changes nothing he draws. Of all
  the looking he does while drawing, between 0.07 % and 0.29 % goes to the sample, though it is close to half of what
  lies in front of him.
- **The lessons barely reward the hand.** Knowing a scribe's hand makes his guess of a noisy line better by at most
  4.5 %, and only while the page is nearly all noise; once the strokes show through the noise the page itself tells
  him the hand and the sample is worth nothing. So the marks he gets hardly push him to study the sample's letters.

## 1. The fault: each pair's prompt form was tied to its kind (fixed)

A cache gives every pair one prompt form, in fixed shares (the content mix: the sample alone 60 %, the sample with the
look in words 20 %, no sample 20 %). `assign_variants` shuffled the list of forms with `random.Random(seed)`, and
`planned_entries` shuffles the pairs with the same seed: two lists of one length shuffled by one stream take the same
order, so every pair got the form at its own place in the manifest, and a page-pair manifest lists its whole pages
before its windows. With `--limit` the two lists differ in length and the forms fall at random, which is why the
Spark's pilots (`--limit 12000`) never showed it.

Counted on Leonardo with `moss/check_mix.py` (it fails a tied cache and passes a mixed one):

| Cache | No sample | The sample alone | The sample with look words |
|---|---|---|---|
| `cache_full_b` (what B trained on) | 28,676 whole pages, 0 windows | 0 pages, 86,027 windows (all 43,014 look changes) | 20,197 pages, 8,479 windows |
| the first 2 MP cache | 28,675 pages, 0 windows | 0 pages, 86,025 windows (all 43,012 look changes) | 20,194 pages, 8,481 windows |
| the Spark's pilot cache (`--limit`) | 795 pages, 1,605 windows | 2,449 pages, 4,751 windows | 860 pages, 1,540 windows |

So B never saw a whole page with the sample alone, never a window without a sample, and never a look change on a
whole page. The validation and held-out caches hold every pair in every form and are right.

Fixed in `c824264`: the forms are shuffled by a stream of their own (`random.Random(f"forms/{seed}")`). A new test
plans a cache without a limit and fails on the old code. The 2 MP cache is built again with the fix and takes the
first build's place before the queued 2 MP run can start (MOSS_RUNS.md, the notes of 11 Nov).

## 2. The training maths (verified, no fault)

Read against KohakuMossTrainer's source and MOSS-Image's report, line for line:

- the noisy page is `(1 - sigma) * page + sigma * noise`, the target the velocity `noise - page`, the loss their mean
  squared error with no weighting by noise level;
- the noise levels are MOSS-Image's own draw (`diffu_moss.train.sigmas`: median 0.78, quartiles 0.64 and 0.89; 34 %
  of the draws above 0.85, 11 % below 0.5);
- the sample's tokens are given clean (the modulation of timestep 0) and take no loss.

The package's tests pass on the fixed code, and both type checkers report no errors.
## 3. What the model reads from its sample (measured)

All on B at step 7,000 (the published EMA weights), on pages of volumes held out from its training (24 pairs of 15
volumes, 19 windows and 5 whole pages; none from a volume B trained on), unless another model is named. The scripts
and their outputs are in `~/diffu_queue/audit` on the Spark.

**The loss with the sample swapped, blurred or removed** (`ref_probe.py`: the flow-matching error at fixed noise
levels, the same noise for every variant; how much worse than with the page's own sample, in the text lines):

| The sample given | sigma 0.30 | 0.50 | 0.70 | 0.85 | 0.95 |
|---|---|---|---|---|---|
| another volume's, same kind (colour or grey) | 0.00 % | 0.00 % | -0.02 % | +0.02 % | +0.50 % (±0.15) |
| another volume's, other kind | +0.01 % | +0.02 % | +0.01 % | +0.12 % | +1.94 % (±0.32) |
| its own, blurred until no stroke is left | +0.04 % | +0.05 % | +0.07 % | +0.18 % | +0.86 % (±0.27) |
| none | 0.00 % | 0.00 % | -0.01 % | +0.02 % | +0.61 % (±0.18) |

Below sigma 0.85 nothing about the sample matters; at 0.95 its kind (colour or grey) matters most, its strokes
little. Sampling spends 24 of its 50 steps above 0.85, so the look is set there and carried down.

**Where the attention goes** (`attn_probe.py`, 6 pairs, blocks 2-41): of the drawn tokens' attention 0.29 % lies on
the sample at sigma 0.95, 0.11 % at 0.8 and 0.07 % at 0.5, though the sample is ~45 % of all tokens (the prompt takes
~29 %). Of that sliver, 8-10 % falls within one cell of the token's own place on the page (an even spread would give
0.28 %): the model looks at the sample where MOSS-Image was pretrained to look at an image it edits, at the same
spot.

**Why the same spot** (the geometry): MOSS-Image lays a reference over the drawn image. Each reference token takes the
(row, column) of the drawn token under it, one step along a fourth position axis; that step turns 2 of the 64 pairs of
rotary phases by more than half a radian, so a reference token looks to attention like its drawn twin moved by one
cell (mean cosine 0.957). An image to edit holds, at each spot, what is drawn at that spot. A sample page holds
other lines there.

**What knowing the hand is worth** (`adapter_gap.py`: B against B with a scribe's adapter merged, on 12 pages of that
scribe the adapter never trained on; the change of the error in the text lines):

| Adapter | sigma 0.15 | 0.30 | 0.50 | 0.70 | 0.85 | 0.95 |
|---|---|---|---|---|---|---|
| the 1715 clerk's (carries his hand in drawings) | +0.02 % | -0.05 % | -0.25 % | -1.18 % | -2.24 % | -4.53 % (±0.65) |
| Påhlman's (does not) | 0.00 % | 0.00 % | -0.01 % | -0.04 % | -0.04 % | +0.35 % (±0.20) |

Knowing the hand pays where the page is still mostly noise and nowhere else, and at most 4.5 % of the error in the
lines; B takes 0.5-0.6 % from its sample there. The probe agrees with the eye: the adapter that carries a hand shows
it, the one that does not shows nothing.

**A model taught with hand changes does read strokes** (the Spark's pilot, 2,000 steps, half its sample-form pairs
given the same change of stroke weight, slant and width on page and sample; 12 pairs): blurring its sample costs it
+3.08 % (±0.64) in the lines at sigma 0.95, +0.92 % at 0.85, +0.35 % at 0.70 and +0.19 % at 0.50, three to five
times B's, and more than taking the sample away (+0.59 %). So the sample's path can carry strokes when the lesson
rewards it; what that pilot was rewarded for is the change it was given, and another volume's sample still costs it
only +0.85 %.

**B does not read what a reference shows, even when it shows the answer** (`strip_probe.py`: the page's own upper
lines given as the reference, clean, each token at its own place on the page, and the error read on the upper lines
themselves and on the lower lines; against no reference):

| The reference given | On the upper lines it shows, sigma 0.85 | 0.95 | On the lower lines, 0.85 | 0.95 |
|---|---|---|---|---|
| the page's own sample page (as trained) | -0.07 % | -0.54 % | -0.03 % | -0.56 % (±0.20) |
| the page's own upper lines, in place | -0.34 % | -1.10 % (±0.16) | -0.07 % | -0.68 % (±0.19) |
| the same on the drawn image's own plane | -0.57 % | -1.40 % (±0.17) | -0.07 % | -0.74 % (±0.19) |
| another volume's upper lines, in place | 0.00 % | +0.43 % | +0.04 % | +0.46 % (±0.17) |

A reference that holds the very lines being drawn, at the very places they are drawn, lowers B's error on them by
1.1-1.4 % at the highest noise, 0.3-0.6 % at 0.85 and under 0.25 % below. B takes the tone of its reference and
nothing finer: the same scribe's lines beside the ones it draws help it no more than a sample page does (-0.68 %
against -0.56 %), and another volume's lines there cost what another volume's sample costs. The same probe on the
released Base and on the pilot taught with hand changes is running.

## 4. Why the look follows and the hand does not

Put together:

1. **What the lesson pays for.** The look pays in the loss at high noise, where a sample's kind (grey or colour) is
   worth 2 %; the hand pays 4.5 % at most, at the same noise only, and only if the sample's writer is the page's.
   B's samples were any page of the volume (often another scribe's), so for B the hand paid less than that.
2. **What the model was built to do with a reference.** MOSS-Image lays the reference over the page, and was
   pretrained to edit: to look at the same spot. That carries a sample's tone to the page; it does not carry letters,
   which sit elsewhere on a sample. B's attention to its sample is a sliver, a tenth of it at the same spot.
3. **What B then learned.** To take the tone and stop: it reads nothing finer from a reference, not even the answer
   laid in place. The pilot taught with hand changes shows the path itself can carry strokes when they pay.
4. **The fault of section 1** tied whole pages to the forms without the sample alone, so B never trained the form its
   users draw whole pages in. It is not why the hand fails (the Spark's pilots were free of it and their hand fails
   too), but it skewed what B learned from a sample on whole pages.

So the hand needs a lesson that pays for it (a sample that is the page's own scribe for certain; changes of hand
that only the sample can explain) and perhaps a place for the sample that suits finding letters rather than editing
(beside the page, not over it). The sibling pilot tests both: section 6.

## 5. Speed and memory at 2 MP (the PyTorch team's skills)

The profile-model, analyze-memory-snapshot and debug-graph-breaks skills
([meta-pytorch/skills](https://github.com/meta-pytorch/skills); the installed plugin is the repository's files, byte
for byte) applied to the 2 MP training step on 8 A100s (job 64198262) and to the DiT's blocks:

- **The profile** confirms the 1 MP reading of 7 Nov (the GPUs busy 94-96 %, communication alone ~2.3 s of a 19.5 s
  step, no launch overhead: kernels under 50 µs are 0.3 % of kernel time) and shows what it missed: all ranks gather
  each block's weights together, so a round runs at the pace of its largest sample. A rank holding two samples
  without a sample page (8.5 thousand tokens each) computed for 7.3 s of a 19.5 s step and waited in NCCL for the
  rest. Simulated on the 2 MP cache's own sizes, a rank computes 84 % of a step's compute time on average.
  **Fix, built and measured:** `--size-block N` sorts each pass's samples by size inside shuffled blocks of N rounds,
  so a round's samples are of like size. On the 8 A100s (job 64201211, the probe cache, the same samples a pass):
  15.2 s a step on average against 19.1, the losses the same (0.5997 against 0.6000 over steps 7-30); simulated on
  the full cache's sizes, 16.2 s against 18.7. On for the 27,000-step run: about 17 of its 137 hours.
- **The memory snapshot**: 21.5 GiB of tensors at the step's peak, 31-32 of the GPU's 64 GB in use; a third of the
  peak is what stays between steps (weights, their average, Muon's momentum), the rest the saved block inputs
  (3.9 GiB), the gathered weights and one block's activations. Half the reserved memory is free but in small pieces,
  which costs nothing at this size. The free half of the GPU buys little: a block kept uncheckpointed costs ~5 GiB
  for 1/42 of the recompute, about 3 % a step for all that fit.
- **Graph breaks**: none. The joint block compiles whole, drawing and training (bfloat16 autocast, gradient
  checkpointing), in three variants (with a sample, without, in training). Whether compiled blocks make a training
  step faster under FSDP2 is a queued job; in the profile the elementwise kernels a compiler fuses are 13 % of
  kernel time.

## 6. What changes

Done:

- The planner's fault fixed (`c824264`), the 2 MP cache built again (running; the queued run waits for it), the run
  log and the inference guide corrected.
- `--size-block` in the trainer (`1195a7c`), measured on GPUs and on for the 2 MP run.
- Both type checkers back at zero (ty had renamed a rule; pyrefly lacked a config entry).

Running (MOSS_RUNS.md, the audit's runs):

- **The sibling pilot**, two arms from B's step 7,000 on 8,576 pairs whose sample is the other window of the
  target's own page: the sample over the page (as now) and above it. Judged by the loss with the right and the wrong
  scribe's sample, and by drawings from both beside the real windows.
- Per-scribe adapters in the named form on B and on Base (the route that already works for one scribe).

Ranked next, by what the measurements say:

1. If either pilot arm reads the hand: samples by the same scribe for the big run. The 2 MP set already picks the
   nearest page; sibling windows can be added without encoding anything again (their latents and prompts are in the
   cache already).
2. Hand changes at scale (they made the pilot read strokes), on same-scribe samples.
3. If only the `above` arm reads it: the sample's place as a trained setting of the model, in training and drawing.
4. Guidance that pushes toward the sample through trained branches only (today's third pass runs the negative
   prompt, which the fine-tune never trains).
5. A reward on the hand itself (a writer network on the drawn page), if the loss alone keeps paying too little.

## 7. Where things are

On the Spark, `~/diffu_queue/audit`: the probes (`ref_probe.py`, `attn_probe.py`, `adapter_gap.py`, `strip_probe.py`,
`form_probe.py`, `continue_probe.py`), their outputs and `summarize.py`; the trace and snapshot readers
(`trace_summary.py`, `peak_replay.py`, `graph_breaks_check.py`); the pilot's evaluation (`swap_draw.py`,
`swap_score.py`). `~/diffu_queue/leonardo`: every job script and helper run on Leonardo (`moss/` there).
