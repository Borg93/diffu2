#!/bin/bash
# Run one GPU job on a DGX Spark (GB10) without risking the whole machine.
#
# The GB10's 128 GB is unified: GPU allocations come out of system RAM, cgroup memory limits don't see them,
# and running out doesn't raise a CUDA out-of-memory error, it hangs the box (swap thrashing, or a page-
# migration livelock the kernel OOM killer never sees). So every GPU job goes through this guard:
#
#   - one guarded job at a time (an flock): GPU jobs don't stack up;
#   - a pre-flight check: the job starts only if MemAvailable >= --need-gb + --floor-gb;
#   - a watchdog: if MemAvailable falls below --floor-gb, every process of the job is sent SIGTERM, then
#     SIGKILL 10 s later (the job runs in its own systemd user scope, so torchrun children and dataloader
#     workers go too);
#   - no swap for the job (MemorySwapMax=0: no swap death spiral) and oom_score_adj 500 (the kernel kills
#     the job before your services);
#   - the peak memory in use is logged, to size the next run.
#
#   scripts/spark_guard.sh --need-gb 40 --name canvas --log runs/canvas/guard.log -- uv run diffu-train ...
#
# Exit codes: the job's own; 3 if another guarded job holds the lock; 4 if too little memory is free;
# 137 if the watchdog killed it. Sources: natolambert/dgx-spark-setup (scripts/oom_protection.sh) and the
# NVIDIA developer forum thread on GB10 post-mortems and the cgroups gap.

set -u

need_gb=""
floor_gb=20
name="job"
log=""
lock="${XDG_RUNTIME_DIR:-/tmp}/spark_guard.lock"
while [ $# -gt 0 ]; do
    case "$1" in
        --need-gb) need_gb="$2"; shift 2 ;;
        --floor-gb) floor_gb="$2"; shift 2 ;;
        --name) name="$2"; shift 2 ;;
        --log) log="$2"; shift 2 ;;
        --lock) lock="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "spark_guard: unknown option $1" >&2; exit 2 ;;
    esac
done
if [ -z "$need_gb" ] || [ $# -eq 0 ]; then
    echo "usage: spark_guard.sh --need-gb N [--floor-gb 20] [--name x] [--log file] -- command ..." >&2
    exit 2
fi

say() {
    local line
    line="$(date '+%F %T') [spark_guard:$name] $*"
    echo "$line" >&2
    [ -n "$log" ] && echo "$line" >> "$log"
}
avail_gb() { awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo; }
used_gb() { awk '/^MemTotal:/ {t=$2} /^MemAvailable:/ {a=$2} END {printf "%d", (t - a) / 1048576}' /proc/meminfo; }

[ -n "$log" ] && mkdir -p "$(dirname "$log")"
exec 9>>"$lock"
if ! flock -n 9; then
    say "another guarded GPU job is running ($(cat "$lock.owner" 2>/dev/null || echo unknown)); not starting"
    exit 3
fi
echo "$name pid $$ since $(date '+%F %T')" > "$lock.owner"

avail=$(avail_gb)
if [ "$avail" -lt $((need_gb + floor_gb)) ]; then
    say "only ${avail} GB available; this job needs ~${need_gb} GB plus a ${floor_gb} GB floor; not starting"
    exit 4
fi

unit="spark-guard-${name//[^A-Za-z0-9_-]/_}-$(date +%s)"
say "start (${avail} GB available, need ~${need_gb} GB, floor ${floor_gb} GB) unit=${unit}.scope: $*"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
[ -x /usr/local/cuda/bin/ptxas ] && export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"
systemd-run --user --scope --quiet --unit="$unit" -p MemorySwapMax=0 \
    bash -c 'echo 500 > /proc/self/oom_score_adj; exec "$@"' _ "$@" 9>&- &
job=$!

peak=0
killed=0
ticks=0
while kill -0 "$job" 2>/dev/null; do
    avail=$(avail_gb)
    used=$(used_gb)
    [ "$used" -gt "$peak" ] && peak=$used
    if [ "$avail" -lt "$floor_gb" ] && [ "$killed" -eq 0 ]; then
        say "MemAvailable ${avail} GB < floor ${floor_gb} GB: stopping the job (SIGTERM, SIGKILL in 10 s)"
        systemctl --user kill --signal=SIGTERM "${unit}.scope" 2>/dev/null
        killed=1
        ( sleep 10; systemctl --user kill --signal=SIGKILL "${unit}.scope" 2>/dev/null ) &
    fi
    ticks=$((ticks + 1))
    [ $((ticks % 60)) -eq 0 ] && say "running: ${used} GB in use (peak ${peak}), ${avail} GB available"
    sleep 1
done
wait "$job"
status=$?
[ "$killed" -eq 1 ] && status=137
say "end: exit ${status}, peak ${peak} GB in use on the box"
rm -f "$lock.owner"
exit "$status"
