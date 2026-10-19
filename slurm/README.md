# Running Diffu on CINECA Leonardo (Slurm)

Test locally on the GB10 first; these scripts then run the same commands on Leonardo's Booster partition
(4× A100 64 GB + 32 cores per node). Login, 2FA and general Slurm/filesystem help: the `cineca-hpc` Claude
skill. Facts these scripts rely on: GPU jobs need `--partition=boost_usr_prod` and an account; QOS
`boost_qos_dbg` (fast, short) or `boost_qos_bprod` (production, up to 24 h); compute nodes have **no
internet**; keep everything in `$WORK` / `$FAST` (`$HOME` is 50 GB).

| File | What |
|---|---|
| `env.sh` | job environment: repo + caches on `$WORK`, offline Hub, the driver check |
| `setup_login.sh` | once, on a login node: uv, the venv, every Hub model/dataset the jobs read |
| `smoke.sbatch` | 1 GPU, debug QOS: tests + the dino / canvas / compile smoke runs |
| `train.sbatch` | 1 node, 4 GPUs: `diffu-train` via `accelerate --multi_gpu`, `--resume auto` |
| `chain.sh` | N chained `train.sbatch` jobs for runs longer than the 24 h walltime |

## Once

```bash
# on a login node (see the cineca-hpc skill for `step ssh login`)
echo 'export SBATCH_ACCOUNT=<project>' >> ~/.bashrc       # your project account (saldo -b)
cd $WORK && git clone https://github.com/Borg93/diffu2.git && cd diffu2
uv run hf auth login                                      # DINOv3 is gated
bash slurm/setup_login.sh
```

Data: manifests store image paths relative to the repo (the jobs `cd $WORK/diffu2`), so `data_iam/` belongs in
the checkout. Build it there with `uv run --no-sync python scripts/prepare_iam_writers.py --out-dir data_iam`
(login node: it fetches two small DiffusionPen JSONs), or copy it from the GB10 onto `$FAST` (faster I/O) and
link it in:

```bash
rsync -a data_iam/ <user>@login.leonardo.cineca.it:<FAST>/data_iam/  # from the GB10 (1.6 GB); <FAST> = `echo $FAST` there
ln -s $FAST/data_iam $WORK/diffu2/data_iam                             # on Leonardo
```

## Run

```bash
sbatch slurm/smoke.sbatch                         # ~10 min on the debug QOS
OUT_DIR=$WORK/runs/canvas slurm/chain.sh 3 \
    --data-dir data_iam --style-canvas --glyph-pad \
    --text-dropout 0.1 --ref-dropout 0.2 --cond-dropout 0.05 \
    --max-line-width 3072 --max-ref-width 2048 --drop-wide-lines \
    --bucket-width --no-grad-checkpoint --compile --batch-size 8 \
    --htr-recognizer-eval microsoft/trocr-base-handwritten --track
squeue -u $USER;  tail -f slurm-diffu-train-<jobid>.out
```

`--batch-size` is per GPU (4 per node). Each chained job resumes from the newest complete `step_N`, so keep
`--save-every` well under one job's worth of steps. If a batch doesn't fit in an A100's 64 GB, drop
`--no-grad-checkpoint` (slower; `--compile` still works) or lower `--batch-size`.

## Monitoring with trackio

Compute nodes can't reach a Space, so `--track` logs locally (under `$HF_HOME/trackio`, on `$WORK`). From a login
node: `uv run trackio list alerts --project diffu --json`, `uv run trackio get metric --project diffu --run canvas
--metric val_loss --json`, or push to a private Space with `uv run trackio sync --project diffu --space-id
<user>/diffu-trackio`.

## If the driver check fails

torch comes from the CUDA 13.0 (cu130) wheel index, which needs an NVIDIA driver ≥ 580. `env.sh` stops a job
on an older driver. Then run inside a container whose CUDA forward-compatibility libraries cover the gap
(A100s support CUDA forward compatibility): pull an NGC PyTorch image on the login node
(`singularity pull docker://nvcr.io/nvidia/pytorch:<tag>`; `module load profile/global singularity/3.9-pro`)
and `singularity exec --nv` the same commands.
