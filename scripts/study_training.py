"""Study a Diffu training run straight from trackio's SQLite store.

trackio logs each step's metrics as a JSON blob in ``metrics.metrics``. This reads the
loss / lr / val_loss trajectory for one run, prints a downsampled table + a summary (best
val-loss step, final loss, where the LR schedule bottomed out), and maps the best-val step
to the nearest checkpoint on disk so you know which one to load in diffu-studio.

    uv run --no-sync python scripts/study_training.py                  # most recent run
    uv run --no-sync python scripts/study_training.py --run gentle-meadow-5
    uv run --no-sync python scripts/study_training.py --ckpt-root data_out/run3
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

DEFAULT_DB = Path.home() / ".cache/huggingface/trackio/diffu.db"


def _latest_run(con: sqlite3.Connection) -> str:
    row = con.execute(
        "SELECT run_name FROM metrics GROUP BY run_name ORDER BY MAX(timestamp) DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise SystemExit("no runs found in the metrics table")
    return str(row[0])


def _series(con: sqlite3.Connection, run: str) -> list[tuple[int, dict[str, float]]]:
    rows = con.execute(
        "SELECT step, metrics FROM metrics WHERE run_name = ? ORDER BY step, id", (run,)
    ).fetchall()
    series: list[tuple[int, dict[str, float]]] = []
    for step, blob in rows:
        try:
            series.append((int(step), json.loads(blob)))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return series


def _nearest_checkpoint(ckpt_root: Path, step: int) -> Path | None:
    steps = []
    for path in ckpt_root.glob("step_*"):
        head, _, tail = path.name.partition("_")
        if head == "step" and tail.isdigit():
            steps.append((int(tail), path))
    if not steps:
        return None
    return min(steps, key=lambda s: abs(s[0] - step))[1] / "model.safetensors"


def main() -> None:
    ap = argparse.ArgumentParser(description="Study a Diffu run from trackio's SQLite DB.")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--run", default=None, help="run_name (default: most recently active)")
    ap.add_argument("--ckpt-root", type=Path, default=Path("data_out/run3"))
    ap.add_argument("--every", type=int, default=0, help="print every Nth loss point (0 = auto ~30 rows)")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    run = args.run or _latest_run(con)
    series = _series(con, run)
    if not series:
        raise SystemExit(f"no metric rows for run {run!r}")

    keys = sorted({k for _, metrics in series for k in metrics})
    print(f"run: {run}   points: {len(series)}   metrics: {keys}")

    loss = [(s, m["loss"]) for s, m in series if "loss" in m]
    lr_by_step = {s: m["lr"] for s, m in series if "lr" in m}
    vals = [(s, m[k]) for s, m in series for k in m if "val" in k.lower() and "loss" in k.lower()]

    every = args.every or max(1, len(loss) // 30)
    print("\n  step       loss        lr")
    for i, (step, value) in enumerate(loss):
        if i % every and i != len(loss) - 1:
            continue
        lr_at = lr_by_step.get(step, float("nan"))
        print(f"{step:>7}   {value:9.4f}   {lr_at:.2e}")

    if vals:
        best_step, best_val = min(vals, key=lambda x: x[1])
        last_step, last_val = vals[-1]
        print(
            f"\nval_loss: {len(vals)} pts | best {best_val:.4f} @ {best_step} | last {last_val:.4f} @ {last_step}"
        )
        nearest = _nearest_checkpoint(args.ckpt_root, best_step)
        if nearest is not None:
            print(f"-> best checkpoint to load: {nearest}")
    else:
        print("\nno val_loss logged for this run")

    if lr_by_step:
        floor_step = min(lr_by_step, key=lambda s: lr_by_step[s])
        peak = max(lr_by_step.values())
        print(
            f"LR: peak {peak:.2e} -> floor {lr_by_step[floor_step]:.2e} @ step {floor_step} "
            "(the cosine schedule effectively ends here)"
        )


if __name__ == "__main__":
    main()
