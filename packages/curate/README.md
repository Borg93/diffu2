# diffu-curate

Curating real archive scans with machine-made labels (the pools) into training sets. A pool is a folder of volumes of
ALTO files an old pipeline wrote (YOLO regions and lines, TrOCR), beside a folder of their scans; this machine's two
are named in `diffu_curate.settings` (`svea`, `andraarkiv`) and moved by `DIFFU_CURATE_*`.

Every scan of a pool is a row of one Lance table that checks add their scores to. A check scores; it drops nothing. A
training set is selected from the scores afterwards, so a threshold is set, and reset, without scoring again.

    diffu-curate features --pool svea --table runs/htr/curate/scans.lance --workers 24
    diffu-curate features --pool andraarkiv --table runs/htr/curate/scans.lance --workers 24
    diffu-curate calibration --pool andraarkiv --out runs/htr/curate/calibration_andraarkiv
    diffu-curate sample --pool svea --scans 34000 --out /mnt/scratch/gabriel/htr/svea_pilot

- `features`: what a scan's ALTO file alone says (who wrote it, the page's confidence, regions, lines, characters,
  line heights and density, lines with far less or more text than the page's usual for their width, the outlines'
  cover, repeated lines and words). Resumable by volume. Blind to writing the old pipeline did not find.
- `calibration`: on the ground-truth scans a pool also holds and the old pipeline labelled, old lines matched to true
  ones: per line its error, per scan the lines missed and astray and the outlines' cover of each other
  (`scans.lance`, `pages.lance`, `lines.lance`). These scans are easier than a pool at large.
- `sample`: a random sample staged as `diffu-reader-set --xml` reads a ground truth, every volume the ground truth is
  in left out. Build the set with `--split-from-gt-test-lists`: the trainers refuse a real set without a split.

State, findings and the backlog: docs/HTR_RUNS.md.
