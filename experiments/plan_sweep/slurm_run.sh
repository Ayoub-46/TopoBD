#!/bin/bash
# =============================================================================
# SLURM job: FL plan sweep — §8 run matrix (fl_backdoor_run_plan.md)
#
# Parallelism (3 GPUs):
#   GPU 0  cifar10 (all α-levels, all attacks, all defenses)
#   GPU 1  gtsrb   (same)
#   GPU 2  Tier 2  (tiny_imagenet + femnist_leaf)
#
# Resume: already-complete runs are skipped automatically.
# Resubmit the same sbatch command to continue from where the job stopped.
#
# Usage
# -----
#   sbatch experiments/plan_sweep/slurm_run.sh
#
# Optional overrides via --export:
#   sbatch --export=ALL,TIER=1,DEFENSES="none toposentinel flame" \
#          experiments/plan_sweep/slurm_run.sh
# =============================================================================

#SBATCH --job-name=fl_plan_sweep
#SBATCH --partition=gpu-a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96GB
#SBATCH --gres=gpu:3
#SBATCH --time=100:00:00
#SBATCH --output=logs/plan_sweep_%j.out
#SBATCH --error=logs/plan_sweep_%j.err

# ---- Configuration ----------------------------------------------------------
ENVIRONMENT_NAME="${ENVIRONMENT_NAME:-toposentinel}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"
TIME_LIMIT_HOURS="${TIME_LIMIT_HOURS:-98}"
RESULTS_DIR="${RESULTS_DIR:-results}"
TIER="${TIER:-all}"

# Subset filters (space-separated; leave empty for full matrix)
DATASETS="${DATASETS:-}"
ALPHAS="${ALPHAS:-}"
ATTACKS="${ATTACKS:-}"
DEFENSES="${DEFENSES:-}"
SEEDS="${SEEDS:-}"

SUMMARIZE_ONLY="${SUMMARIZE_ONLY:-0}"

set -uo pipefail

# =============================================================================
# 1. ENVIRONMENT
# =============================================================================
echo "============================================================"
echo " Job ID   : $SLURM_JOB_ID"
echo " Node     : $(hostname)"
echo " Start    : $(date '+%Y-%m-%d %H:%M:%S')"
echo " Tier     : $TIER"
echo "============================================================"

mkdir -p logs "$RESULTS_DIR/plan_sweep"

module load Anaconda3 2>/dev/null || module load anaconda3 2>/dev/null || true

if ! conda info --envs | grep -qE "^${ENVIRONMENT_NAME}\s"; then
    conda create -n "${ENVIRONMENT_NAME}" python="${PYTHON_VERSION}" -y
fi

CONDA_RUN="conda run -n ${ENVIRONMENT_NAME} --no-capture-output"

$CONDA_RUN pip install \
    torch==2.5.1+${TORCH_CUDA} \
    torchvision==0.20.1+${TORCH_CUDA} \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" -q

$CONDA_RUN pip install -r requirements.txt -q
set -e

# =============================================================================
# 2. DATA: SCRATCH COPY
# =============================================================================
REPO_DATA="$(pwd)/data"
SCRATCH="${SLURM_TMPDIR:-/tmp/${USER:-user}/$SLURM_JOB_ID}"
SCRATCH_DATA="$SCRATCH/data"
mkdir -p "$SCRATCH_DATA"

_ORIGINAL_DATA_WAS_DIR=0
cleanup() {
    [ -L "data" ] && rm -f data
    if [ "$_ORIGINAL_DATA_WAS_DIR" -eq 1 ] && [ -d "${REPO_DATA}_bk" ]; then
        mv "${REPO_DATA}_bk" "$REPO_DATA"
    fi
    rm -rf "$SCRATCH"
}
trap cleanup EXIT

if [ -d "$REPO_DATA" ]; then
    T0=$(date +%s)
    cp -r "$REPO_DATA/." "$SCRATCH_DATA/"
    echo "[data] Copied in $(( $(date +%s) - T0 ))s"
    _ORIGINAL_DATA_WAS_DIR=1
    mv "$REPO_DATA" "${REPO_DATA}_bk"
    ln -s "$SCRATCH_DATA" "$REPO_DATA"
fi

# =============================================================================
# 3. COMMON ARGUMENTS
# =============================================================================
COMMON="--results-dir $RESULTS_DIR --time-limit-hours $TIME_LIMIT_HOURS --device cuda --tier $TIER"
[ -n "$DATASETS" ]  && COMMON="$COMMON --datasets $DATASETS"
[ -n "$ALPHAS" ]    && COMMON="$COMMON --alphas $ALPHAS"
[ -n "$ATTACKS" ]   && COMMON="$COMMON --attacks $ATTACKS"
[ -n "$DEFENSES" ]  && COMMON="$COMMON --defenses $DEFENSES"
[ -n "$SEEDS" ]     && COMMON="$COMMON --seeds $SEEDS"
[ "$SUMMARIZE_ONLY" -eq 1 ] && COMMON="$COMMON --summarize-only"

# =============================================================================
# 4. PARALLEL GPU WORKERS
# =============================================================================
_pids=""
_forward_sigterm() {
    # shellcheck disable=SC2086
    kill -TERM $_pids 2>/dev/null || true
}
trap '_forward_sigterm' TERM

T_START=$(date +%s)

# GPU 0 — cifar10 (largest Tier 1 dataset, heaviest)
echo "[gpu0] Starting cifar10 …"
CUDA_VISIBLE_DEVICES=0 $CONDA_RUN python -m experiments.plan_sweep.run_sweep \
    --datasets cifar10 $COMMON \
    > "logs/plan_sweep_gpu0_${SLURM_JOB_ID}.log" 2>&1 &
PID0=$!

# GPU 1 — gtsrb
echo "[gpu1] Starting gtsrb …"
CUDA_VISIBLE_DEVICES=1 $CONDA_RUN python -m experiments.plan_sweep.run_sweep \
    --datasets gtsrb $COMMON \
    > "logs/plan_sweep_gpu1_${SLURM_JOB_ID}.log" 2>&1 &
PID1=$!

# GPU 2 — Tier 2 breadth (tiny_imagenet + femnist_leaf)
echo "[gpu2] Starting tier-2 breadth …"
CUDA_VISIBLE_DEVICES=2 $CONDA_RUN python -m experiments.plan_sweep.run_sweep \
    --tier 2 $COMMON \
    > "logs/plan_sweep_gpu2_${SLURM_JOB_ID}.log" 2>&1 &
PID2=$!

_pids="$PID0 $PID1 $PID2"

EXIT0=0; EXIT1=0; EXIT2=0
wait $PID0 || EXIT0=$?
wait $PID1 || EXIT1=$?
wait $PID2 || EXIT2=$?

T_ELAPSED=$(( $(date +%s) - T_START ))
printf "[sweep] Workers done in %dh %dm %ds\n" \
    $((T_ELAPSED/3600)) $(((T_ELAPSED%3600)/60)) $((T_ELAPSED%60))

# =============================================================================
# 5. MERGED SUMMARY
# =============================================================================
$CONDA_RUN python -m experiments.plan_sweep.run_sweep \
    --results-dir "$RESULTS_DIR" \
    --summarize-only --tier all

echo "============================================================"
echo " Complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Summary : $RESULTS_DIR/plan_sweep_summary.csv"
NDIRS=$(ls -1 "$RESULTS_DIR/plan_sweep/" 2>/dev/null | wc -l || echo 0)
echo " Run dirs: $NDIRS"
echo "============================================================"

[ $EXIT0 -eq 0 ] && [ $EXIT1 -eq 0 ] && [ $EXIT2 -eq 0 ] || exit 1
