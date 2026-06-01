#!/bin/bash
# =============================================================================
# SLURM job: FL benchmark — full attack × defense × dataset × seed matrix.
#
# Parallelism
# -----------
# Three GPUs run in parallel:
#   GPU 0  mnist + femnist          (fast datasets — share one GPU)
#   GPU 1  gtsrb + cifar10          (medium datasets — share one GPU)
#   GPU 2  tiny_imagenet            (heaviest — alone on one GPU)
#
# Each GPU worker calls run_benchmark.py with its dataset subset and the
# shared time budget.  The main process waits for all three, then runs a
# final summarize-only pass that merges all four datasets into one CSV.
#
# Resume / graceful stop
# ----------------------
# Already-complete runs (final_model.pt present + full metrics.csv) are
# skipped automatically.  Resubmit the same sbatch command to continue from
# where the job left off.  The Python script also catches SIGTERM so the
# current run finishes cleanly before the job exits.
#
# Usage
# -----
#   sbatch experiments/benchmark/slurm_run.sh
#
# Optional overrides via --export:
#   sbatch --export=ALL,ATTACKS="patch neurotoxin",SEEDS="0 1 2" \
#          experiments/benchmark/slurm_run.sh
#   sbatch --export=ALL,SUMMARIZE_ONLY=1 \
#          experiments/benchmark/slurm_run.sh
# =============================================================================

# ---- SLURM directives -------------------------------------------------------
#SBATCH --job-name=fl_benchmark
#SBATCH --partition=gpu-a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24          # 8 per GPU worker + headroom
#SBATCH --mem=96GB                  # ~32 GB per worker
#SBATCH --gres=gpu:3
#SBATCH --time=100:00:00
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

# ---- Configuration (override via --export) ----------------------------------
ENVIRONMENT_NAME="${ENVIRONMENT_NAME:-toposentinel}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"

# Python-side time budget (hours).  Set below the SBATCH --time value so the
# workers stop voluntarily and the summary pass completes before the hard kill.
TIME_LIMIT_HOURS="${TIME_LIMIT_HOURS:-98}"

RESULTS_DIR="${RESULTS_DIR:-results}"

# Optional subsetting — leave empty to run the full matrix
ATTACKS="${ATTACKS:-}"
DEFENSES="${DEFENSES:-}"
SEEDS="${SEEDS:-}"

# Set to 1 to skip training and only (re-)generate the summary CSV
SUMMARIZE_ONLY="${SUMMARIZE_ONLY:-0}"

# ---- Strict error handling --------------------------------------------------
set -uo pipefail

# =============================================================================
# 1. ENVIRONMENT SETUP
# =============================================================================
echo "============================================================"
echo " Job ID   : $SLURM_JOB_ID"
echo " Node     : $(hostname)"
echo " Start    : $(date '+%Y-%m-%d %H:%M:%S')"
echo " Workdir  : $(pwd)"
echo " GPUs     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd ',' || echo 'n/a')"
echo "============================================================"

mkdir -p logs "$RESULTS_DIR/benchmark"

module load Anaconda3 2>/dev/null || module load anaconda3 2>/dev/null || true

if ! conda info --envs | grep -qE "^${ENVIRONMENT_NAME}\s"; then
    echo "[setup] Creating conda environment '${ENVIRONMENT_NAME}' …"
    conda create -n "${ENVIRONMENT_NAME}" python="${PYTHON_VERSION}" -y
fi

CONDA_RUN="conda run -n ${ENVIRONMENT_NAME} --no-capture-output"
TORCH_WHEEL_URL="https://download.pytorch.org/whl/${TORCH_CUDA}"

echo "[setup] Installing PyTorch (${TORCH_CUDA}) …"
$CONDA_RUN pip install \
    torch==2.5.1+${TORCH_CUDA} \
    torchvision==0.20.1+${TORCH_CUDA} \
    --index-url "${TORCH_WHEEL_URL}" -q

echo "[setup] Installing project dependencies …"
$CONDA_RUN pip install -r requirements.txt -q

set -e

echo "[setup] Python : $($CONDA_RUN python --version)"
echo "[setup] PyTorch: $($CONDA_RUN python -c 'import torch; print(torch.__version__)')"
echo "[setup] CUDA   : $($CONDA_RUN python -c 'import torch; print(torch.cuda.is_available())')"

# =============================================================================
# 2. DATA: COPY ALL DATASETS TO LOCAL SCRATCH
# =============================================================================
REPO_DATA="$(pwd)/data"

SCRATCH="${SLURM_TMPDIR:-/tmp/${USER:-user}/$SLURM_JOB_ID}"
SCRATCH_DATA="$SCRATCH/data"
mkdir -p "$SCRATCH_DATA"

_ORIGINAL_DATA_WAS_DIR=0
cleanup() {
    echo "[cleanup] Restoring data directory …"
    [ -L "data" ] && rm -f data
    if [ "$_ORIGINAL_DATA_WAS_DIR" -eq 1 ] && [ -d "${REPO_DATA}_network_backup" ]; then
        mv "${REPO_DATA}_network_backup" "$REPO_DATA"
    fi
    rm -rf "$SCRATCH"
    echo "[cleanup] Done."
}
trap cleanup EXIT

if [ -d "$REPO_DATA" ]; then
    echo "[data] Copying datasets to scratch …"
    T0=$(date +%s)
    cp -r "$REPO_DATA/." "$SCRATCH_DATA/"
    echo "[data] Copy complete in $(( $(date +%s) - T0 ))s"
    _ORIGINAL_DATA_WAS_DIR=1
    mv "$REPO_DATA" "${REPO_DATA}_network_backup"
    ln -s "$SCRATCH_DATA" "$REPO_DATA"
    echo "[data] data/ → scratch (fast path)"
else
    echo "[data] Warning: data/ not found — datasets will be downloaded on first run."
fi

# =============================================================================
# 3. BUILD COMMON ARGUMENTS
# =============================================================================
COMMON_ARGS="--results-dir $RESULTS_DIR --time-limit-hours $TIME_LIMIT_HOURS --device cuda"
[ -n "$ATTACKS" ]  && COMMON_ARGS="$COMMON_ARGS --attacks $ATTACKS"
[ -n "$DEFENSES" ] && COMMON_ARGS="$COMMON_ARGS --defenses $DEFENSES"
[ -n "$SEEDS" ]    && COMMON_ARGS="$COMMON_ARGS --seeds $SEEDS"
[ "$SUMMARIZE_ONLY" -eq 1 ] && COMMON_ARGS="$COMMON_ARGS --summarize-only"

# =============================================================================
# 4. PARALLEL GPU WORKERS
# =============================================================================
echo ""
echo "============================================================"
echo " FL Benchmark — parallel execution"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

T_START=$(date +%s)

# SIGTERM trap: forward signal to all child workers so they stop gracefully.
_pids=""
_forward_sigterm() {
    echo "[main] SIGTERM received — forwarding to workers …"
    # shellcheck disable=SC2086
    kill -TERM $_pids 2>/dev/null || true
}
trap '_forward_sigterm' TERM

# GPU 0 — mnist + femnist (fast; share one GPU)
echo "[gpu0] Starting mnist + femnist …"
CUDA_VISIBLE_DEVICES=0 $CONDA_RUN python -m experiments.benchmark.run_benchmark \
    --datasets mnist femnist \
    $COMMON_ARGS \
    > "logs/benchmark_gpu0_${SLURM_JOB_ID}.log" 2>&1 &
PID0=$!

# GPU 1 — gtsrb + cifar10
echo "[gpu1] Starting gtsrb + cifar10 …"
CUDA_VISIBLE_DEVICES=1 $CONDA_RUN python -m experiments.benchmark.run_benchmark \
    --datasets gtsrb cifar10 \
    $COMMON_ARGS \
    > "logs/benchmark_gpu1_${SLURM_JOB_ID}.log" 2>&1 &
PID1=$!

# GPU 2 — tiny_imagenet (heaviest — alone)
echo "[gpu2] Starting tiny_imagenet …"
CUDA_VISIBLE_DEVICES=2 $CONDA_RUN python -m experiments.benchmark.run_benchmark \
    --datasets tiny_imagenet \
    $COMMON_ARGS \
    > "logs/benchmark_gpu2_${SLURM_JOB_ID}.log" 2>&1 &
PID2=$!

_pids="$PID0 $PID1 $PID2"

# Wait for all workers; collect exit codes
EXIT0=0; EXIT1=0; EXIT2=0
wait $PID0 || EXIT0=$?
wait $PID1 || EXIT1=$?
wait $PID2 || EXIT2=$?

T_ELAPSED=$(( $(date +%s) - T_START ))
printf "[benchmark] All workers finished in %dh %dm %ds\n" \
    $((T_ELAPSED/3600)) $(((T_ELAPSED%3600)/60)) $((T_ELAPSED%60))

echo "[gpu0] exit=$EXIT0  [gpu1] exit=$EXIT1  [gpu2] exit=$EXIT2"

# =============================================================================
# 5. MERGE SUMMARY (all four datasets)
# =============================================================================
echo ""
echo "[summary] Generating merged benchmark_summary.csv …"
$CONDA_RUN python -m experiments.benchmark.run_benchmark \
    --results-dir "$RESULTS_DIR" \
    --summarize-only

# =============================================================================
# 6. JOB REPORT
# =============================================================================
echo ""
echo "============================================================"
echo " Job complete"
echo " End: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Summary: $RESULTS_DIR/benchmark_summary.csv"
NDIRS=$(ls -1 "$RESULTS_DIR/benchmark/" 2>/dev/null | wc -l || echo 0)
echo " Run directories: $NDIRS"
echo " Per-GPU logs:"
ls -lh "logs/benchmark_gpu"*"_${SLURM_JOB_ID}.log" 2>/dev/null || true
echo "============================================================"

# Fail the job if any worker failed
[ $EXIT0 -eq 0 ] && [ $EXIT1 -eq 0 ] && [ $EXIT2 -eq 0 ] || exit 1
