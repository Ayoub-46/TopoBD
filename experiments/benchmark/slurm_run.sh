#!/bin/bash

# ==========================================
# SLURM DIRECTIVES
# ==========================================
#SBATCH --partition=gpu-h100
#SBATCH --time=100:00:00
#SBATCH --job-name=fl_benchmark
#SBATCH --error=logs/benchmark_%j.err
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32          # 8 per GPU worker
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

# ==========================================
# CONFIGURATION
# ==========================================
PYTHON_VERSION=3.10
ENVIRONMENT_NAME="toposentinel"
TORCH_CUDA="${TORCH_CUDA:-cu124}"

# Python-side time budget — workers stop voluntarily before the hard SLURM
# limit so the summary pass completes cleanly.  Set ~2 h below --time.
TIME_LIMIT_HOURS="${TIME_LIMIT_HOURS:-98}"

RESULTS_DIR="${RESULTS_DIR:-results}"

# Optional subsetting — leave empty to run the full matrix for each dataset
ATTACKS="${ATTACKS:-}"
DEFENSES="${DEFENSES:-}"
SEEDS="${SEEDS:-}"

# Set to 1 to skip training and only (re-)generate the summary CSV
SUMMARIZE_ONLY="${SUMMARIZE_ONLY:-0}"

# ==========================================
# ENVIRONMENT SETUP
# ==========================================
module load Anaconda3

if ! conda info --envs | grep -q "^${ENVIRONMENT_NAME}"; then
    echo "Creating environment ${ENVIRONMENT_NAME}..."
    conda create -n ${ENVIRONMENT_NAME} python=${PYTHON_VERSION} -y
fi

source activate ${ENVIRONMENT_NAME}

echo "Installing PyTorch (${TORCH_CUDA})..."
pip install torch==2.5.1+${TORCH_CUDA} torchvision==0.20.1+${TORCH_CUDA} \
    --index-url https://download.pytorch.org/whl/${TORCH_CUDA} -q

echo "Installing project dependencies..."
pip install -r requirements.txt -q

mkdir -p logs "$RESULTS_DIR/benchmark"

echo "============================================================"
echo " Job ID   : $SLURM_JOB_ID"
echo " Node     : $(hostname)"
echo " Start    : $(date '+%Y-%m-%d %H:%M:%S')"
echo " Python   : $(python --version)"
echo " PyTorch  : $(python -c 'import torch; print(torch.__version__)')"
echo " CUDA     : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo " GPUs     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd ',' || echo n/a)"
echo "============================================================"

# ==========================================
# DATA: COPY TO LOCAL SCRATCH
# ==========================================
if [ -z "$SLURM_TMPDIR" ]; then
    SCRATCH_DIR="/tmp/$USER/$SLURM_JOB_ID"
else
    SCRATCH_DIR="$SLURM_TMPDIR"
fi

echo "Setting up scratch at: $SCRATCH_DIR"
mkdir -p $SCRATCH_DIR

if [ -d "data" ]; then
    echo "Copying data to local scratch..."
    cp -r data $SCRATCH_DIR/
    mv data data_network_backup
    ln -s $SCRATCH_DIR/data ./data
    echo "data/ -> scratch (fast path)"
else
    echo "Warning: 'data' folder not found — datasets will be downloaded on first run."
fi

# ==========================================
# BUILD COMMON ARGUMENTS
# ==========================================
COMMON="--results-dir $RESULTS_DIR --time-limit-hours $TIME_LIMIT_HOURS --device cuda"
[ -n "$ATTACKS" ]  && COMMON="$COMMON --attacks $ATTACKS"
[ -n "$DEFENSES" ] && COMMON="$COMMON --defenses $DEFENSES"
[ -n "$SEEDS" ]    && COMMON="$COMMON --seeds $SEEDS"
[ "$SUMMARIZE_ONLY" -eq 1 ] && COMMON="$COMMON --summarize-only"

# ==========================================
# PARALLEL EXECUTION — 4 GPUS
# ==========================================
# One GPU per dataset; each worker runs the full attack × defense × seed
# matrix for its dataset and stops gracefully when the time budget runs out.
# Already-complete runs are skipped automatically on resubmission.
#
#   GPU 0  femnist        (~50 min/run)
#   GPU 1  gtsrb          (~90 min/run)
#   GPU 2  cifar10        (~180 min/run)
#   GPU 3  tiny_imagenet  (~300 min/run)

echo ""
echo "Starting parallel benchmark on 4 GPUs..."
echo "$(date '+%Y-%m-%d %H:%M:%S')"

# # --- GPU 0: FEMNIST ---
# (
#     echo "[GPU 0] Starting femnist"
#     export CUDA_VISIBLE_DEVICES=0
#     python -m experiments.benchmark.run_benchmark \
#         --datasets femnist \
#         $COMMON \
#         > logs/benchmark_gpu0_${SLURM_JOB_ID}.log 2>&1
#     echo "[GPU 0] Done"
# ) &
# PID0=$!

# # --- GPU 1: GTSRB ---
# (
#     echo "[GPU 1] Starting gtsrb"
#     export CUDA_VISIBLE_DEVICES=1
#     python -m experiments.benchmark.run_benchmark \
#         --datasets gtsrb \
#         $COMMON \
#         > logs/benchmark_gpu1_${SLURM_JOB_ID}.log 2>&1
#     echo "[GPU 1] Done"
# ) &
# PID1=$!

# --- GPU 2: CIFAR-10 ---
(
    echo "[GPU 2] Starting cifar10"
    export CUDA_VISIBLE_DEVICES=0
    python -m experiments.benchmark.run_benchmark \
        --datasets cifar10 \
        $COMMON \
        > logs/benchmark_gpu2_${SLURM_JOB_ID}.log 2>&1
    echo "[GPU 2] Done"
) &
PID2=$!

# # --- GPU 3: Tiny-ImageNet ---
# (
#     echo "[GPU 3] Starting tiny_imagenet"
#     export CUDA_VISIBLE_DEVICES=3
#     python -m experiments.benchmark.run_benchmark \
#         --datasets tiny_imagenet \
#         $COMMON \
#         > logs/benchmark_gpu3_${SLURM_JOB_ID}.log 2>&1
#     echo "[GPU 3] Done"
# ) &
# PID3=$!

# Wait for all workers
wait $PID2 

echo ""
echo "All workers finished."
echo "$(date '+%Y-%m-%d %H:%M:%S')"

# Show last few lines of each GPU log for quick inspection
for gpu in 0 1 2 3; do
    logfile="logs/benchmark_gpu${gpu}_${SLURM_JOB_ID}.log"
    if [ -f "$logfile" ]; then
        echo ""
        echo "--- GPU ${gpu} last 5 lines ---"
        tail -5 "$logfile"
    fi
done

# ==========================================
# MERGE SUMMARY
# ==========================================
echo ""
echo "Generating merged benchmark_summary.csv..."
python -m experiments.benchmark.run_benchmark \
    --results-dir "$RESULTS_DIR" \
    --summarize-only

# ==========================================
# CLEANUP
# ==========================================
if [ -L "data" ]; then
    rm -f data
    [ -d "data_network_backup" ] && mv data_network_backup data
fi
rm -rf $SCRATCH_DIR

echo ""
echo "============================================================"
echo " Job complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Summary    : $RESULTS_DIR/benchmark_summary.csv"
echo " Run dirs   : $(ls -1 $RESULTS_DIR/benchmark/ 2>/dev/null | wc -l)"
echo " Per-GPU logs:"
ls -lh logs/benchmark_gpu*_${SLURM_JOB_ID}.log 2>/dev/null || true
echo "============================================================"
