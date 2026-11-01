# syntax=docker/dockerfile:1.11
#
# The inference image: diffu-runner (diffu-run, diffu-run-export), the page pipeline it draws with (MOSS-Image, the
# TrOCR judge) and Ray, for the head and the GPU workers of a KubeRay cluster. No model weights: they are staged on a
# volume and mounted (docs/INFERENCE.md). Build from the repo root:
#
#   docker buildx build -f .docker/inference.dockerfile --platform linux/amd64 \
#     --build-arg BUILD_DATE=$(date -u +%FT%TZ) --build-arg VCS_REF=$(git rev-parse HEAD) \
#     --build-arg VERSION=$(git describe --always) -t <registry>/diffu-inference:<tag> .
#
# torch comes from the lock's cu130 index; its CUDA libraries (cuBLAS, cuDNN, NCCL, ...) are pip packages in the venv,
# so the base is plain Ubuntu and the host needs only an NVIDIA driver for CUDA 13 (580 or newer).

# ── uv: the uv binary (0.12.18, by digest), copied into the builder only ─────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:0.12.18@sha256:3adc3706091ce7c2fe595e669628caedd6d951551b92b258b7e7dbe06d9440bc AS uv

# ── base: Ubuntu 24.04 (by digest) with what the runtime loads (OpenMP, CA roots, tini), wget for KubeRay probes ──
FROM ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852 AS base
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]
# Reason: the digest-pinned base fixes the release; apt resolves that release's current security updates.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 tini wget \
    && rm -rf /var/lib/apt/lists/*

# ── builder: a managed Python and the locked workspace (diffu-runner and what it imports) in /opt/venv ───────────
FROM base AS builder
# Reason: as in base; git fetches diffusers at the commit uv.lock pins.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PYTHON_DOWNLOADS=auto \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.13
# 1. the third-party dependencies alone: this layer is rebuilt only when uv.lock or a pyproject.toml changes
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/core/pyproject.toml,target=packages/core/pyproject.toml \
    --mount=type=bind,source=packages/flux3/pyproject.toml,target=packages/flux3/pyproject.toml \
    --mount=type=bind,source=packages/moss/pyproject.toml,target=packages/moss/pyproject.toml \
    --mount=type=bind,source=packages/page/pyproject.toml,target=packages/page/pyproject.toml \
    --mount=type=bind,source=packages/runner/pyproject.toml,target=packages/runner/pyproject.toml \
    --mount=type=bind,source=packages/studio/pyproject.toml,target=packages/studio/pyproject.toml \
    uv sync --frozen --no-install-workspace --package diffu-runner --no-default-groups --no-editable
# 2. the workspace's own packages that diffu-runner imports, built as wheels (no source tree in the image); studio and
# flux3 are workspace members uv must see, not installed
COPY pyproject.toml uv.lock ./
COPY packages/core packages/core
COPY packages/moss packages/moss
COPY packages/page packages/page
COPY packages/runner packages/runner
COPY packages/studio/pyproject.toml packages/studio/pyproject.toml
COPY packages/flux3/pyproject.toml packages/flux3/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --package diffu-runner --no-default-groups --no-editable

# ── runtime: the Python and the venv, a non-root user, no build tools ─────────────────────────────────────────────
FROM base AS runtime
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION
LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.title="diffu-inference" \
      org.opencontainers.image.description="diffu-run: synthetic archive pages with ALTO/PAGE labels, drawn by MOSS-Image on Ray" \
      org.opencontainers.image.licenses="Apache-2.0"
COPY --link --from=builder /opt/python /opt/python
COPY --link --from=builder /opt/venv /opt/venv
RUN groupadd -r --gid 10001 diffu \
    && useradd -r --no-create-home --shell /usr/sbin/nologin --uid 10001 --gid 10001 diffu \
    && { find / -xdev -perm /6000 -type f -exec chmod a-s {} + || true; }
# HOME and every cache under /tmp (an emptyDir under a read-only root); weights are read only from mounted volumes and
# never fetched (HF_HUB_OFFLINE), torch loads checkpoints weights-only, and nothing reports usage
ENV PATH="/opt/venv/bin:${PATH}" \
    HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    RAY_USAGE_STATS_ENABLED=0 \
    TORCH_FORCE_WEIGHTS_ONLY_LOAD=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1
USER 10001:10001
WORKDIR /tmp
# KubeRay replaces the entrypoint with its own `ray start` command; tini serves `docker run`
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["diffu-run", "--help"]
