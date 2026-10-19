# Environment for the Diffu Slurm jobs on CINECA Leonardo — sourced by every slurm/*.sbatch (and handy in an
# interactive `srun --pty bash` shell). Compute nodes have NO internet: the venv, the Hub models and the
# datasets must already be on $WORK (slurm/setup_login.sh, run once on a login node). No `set -e` here: an
# interactive shell that sources this file must survive a failing command; the scripts set it themselves.

export DIFFU_ROOT="${DIFFU_ROOT:-$WORK/diffu2}"  # the repo checkout ($HOME is only 50 GB)
export PATH="$HOME/.local/bin:$PATH"  # uv
export UV_CACHE_DIR="$WORK/.cache/uv" UV_PYTHON_INSTALL_DIR="$WORK/.local/share/uv/python"
export HF_HOME="$WORK/.cache/huggingface" TORCH_HOME="$WORK/.cache/torch"  # trackio logs under HF_HOME too
if [ -n "${SLURM_JOB_ID:-}" ]; then  # on a compute node: read the pre-fetched caches only
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    # torch's cu130 wheels need an NVIDIA driver >= 580 (CUDA 13); fail here, not deep inside a run.
    driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    if [ "${driver%%.*}" -lt 580 ]; then
        echo "NVIDIA driver $driver < 580: the cu130 torch wheels can't run on this node (see slurm/README.md)" >&2
        return 1
    fi
fi
cd "$DIFFU_ROOT"
