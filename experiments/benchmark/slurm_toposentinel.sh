#!/bin/bash

# ==========================================
# SLURM DIRECTIVES
# ==========================================
#SBATCH --partition=gpu-a6000
#SBATCH --time=100:00:00
#SBATCH --job-name=topo_benchmark
#SBATCH --error=logs/topo_benchmark_%j.err
#SBATCH --output=logs/topo_benchmark_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # 8 per GPU worker
#SBATCH --mem=64GB
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

# Optional subsetting — leave empty to run all attacks / seeds
ATTACKS="${ATTACKS:-}"
SEEDS="${SEEDS:-}"

# DKW filter hyperparameters (leave empty to use script defaults)
TARGET_FPR="${TARGET_FPR:-}"        # default 0.05
DKW_CONFIDENCE="${DKW_CONFIDENCE:-}" # default 0.95

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

mkdir -p logs "$RESULTS_DIR/toposentinel_benchmark"

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
    if cp -r data $SCRATCH_DIR/ && [ -d "$SCRATCH_DIR/data" ]; then
        mv data data_network_backup
        ln -s $SCRATCH_DIR/data ./data
        echo "data/ -> scratch (fast path)"
    else
        echo "Warning: cp to scratch failed — keeping data/ on network filesystem."
        rm -rf $SCRATCH_DIR/data 2>/dev/null || true
    fi
else
    echo "Warning: 'data' folder not found — datasets will be downloaded on first run."
fi

# ==========================================
# BUILD COMMON ARGUMENTS
# ==========================================
COMMON="--results-dir $RESULTS_DIR --time-limit-hours $TIME_LIMIT_HOURS --device cuda"
[ -n "$ATTACKS" ]        && COMMON="$COMMON --attacks $ATTACKS"
[ -n "$SEEDS" ]          && COMMON="$COMMON --seeds $SEEDS"
[ -n "$TARGET_FPR" ]     && COMMON="$COMMON --target-fpr $TARGET_FPR"
[ -n "$DKW_CONFIDENCE" ] && COMMON="$COMMON --dkw-confidence $DKW_CONFIDENCE"
[ "$SUMMARIZE_ONLY" -eq 1 ] && COMMON="$COMMON --summarize-only"

# ==========================================
# PARALLEL EXECUTION — 4 GPUS
# ==========================================
# One GPU per dataset; each worker runs the full attack × seed matrix for its
# dataset against TopoSentinel (DKW filter) and stops gracefully when the time
# budget runs out.  Already-complete runs are skipped on resubmission.
#
# Estimated wall time per GPU (4 attacks × 10 seeds):
#   GPU 0  femnist   ~33 h  (~50 min/run)
#   GPU 1  gtsrb     ~60 h  (~90 min/run)
#   GPU 2  cifar10   ~120 h (~180 min/run)
#   GPU 3  cifar100  ~133 h (~200 min/run)

echo ""
echo "Starting TopoSentinel benchmark on 4 GPUs..."
echo "$(date '+%Y-%m-%d %H:%M:%S')"

# --- GPU 0: FEMNIST ---
(
    echo "[GPU 0] Starting femnist"
    export CUDA_VISIBLE_DEVICES=0
    python -m experiments.benchmark.run_toposentinel \
        --datasets femnist \
        $COMMON \
        > logs/topo_benchmark_gpu0_${SLURM_JOB_ID}.log 2>&1
    echo "[GPU 0] Done"
) &
PID0=$!

# # --- GPU 1: GTSRB ---
# (
#     echo "[GPU 1] Starting gtsrb"
#     export CUDA_VISIBLE_DEVICES=1
#     python -m experiments.benchmark.run_toposentinel \
#         --datasets gtsrb \
#         $COMMON \
#         > logs/topo_benchmark_gpu1_${SLURM_JOB_ID}.log 2>&1
#     echo "[GPU 1] Done"
# ) &
# PID1=$!

# # --- GPU 2: CIFAR-10 ---
# (
#     echo "[GPU 2] Starting cifar10"
#     export CUDA_VISIBLE_DEVICES=2
#     python -m experiments.benchmark.run_toposentinel \
#         --datasets cifar10 \
#         $COMMON \
#         > logs/topo_benchmark_gpu2_${SLURM_JOB_ID}.log 2>&1
#     echo "[GPU 2] Done"
# ) &
# PID2=$!

# --- GPU 3: CIFAR-100 ---
# (
#     echo "[GPU 3] Starting cifar100"
#     export CUDA_VISIBLE_DEVICES=3
#     python -m experiments.benchmark.run_toposentinel \
#         --datasets cifar100 \
#         $COMMON \
#         > logs/topo_benchmark_gpu3_${SLURM_JOB_ID}.log 2>&1
#     echo "[GPU 3] Done"
# ) &
# PID3=$!

# Wait for all workers
wait $PID0 
# $PID1 $PID2 
# $PID3

echo ""
echo "All workers finished."
echo "$(date '+%Y-%m-%d %H:%M:%S')"

# Show last few lines of each GPU log for quick inspection
for gpu in 0 1 2 3; do
    logfile="logs/topo_benchmark_gpu${gpu}_${SLURM_JOB_ID}.log"
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
echo "Generating merged toposentinel_summary.csv..."
python -m experiments.benchmark.run_toposentinel \
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
echo " Summary    : $RESULTS_DIR/toposentinel_summary.csv"
echo " Run dirs   : $(ls -1 $RESULTS_DIR/toposentinel_benchmark/ 2>/dev/null | wc -l)"
echo " Per-GPU logs:"
ls -lh logs/topo_benchmark_gpu*_${SLURM_JOB_ID}.log 2>/dev/null || true
echo "============================================================"
