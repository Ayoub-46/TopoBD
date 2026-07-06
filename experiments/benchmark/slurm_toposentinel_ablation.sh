#!/bin/bash

# ==========================================
# SLURM DIRECTIVES
# ==========================================
#SBATCH --partition=gpu-a6000
#SBATCH --time=140:00:00
#SBATCH --job-name=topo_ablation
#SBATCH --error=logs/topo_ablation_%j.err
#SBATCH --output=logs/topo_ablation_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64          # 8 per GPU worker
#SBATCH --mem=256GB
#SBATCH --gres=gpu:8
#
# NOTE: This script requires 8 GPUs (one per dataset × variant).  If your
# partition only has 4-GPU nodes, either:
#   a) Change --gres=gpu:4 and run two submissions with VARIANTS=dkw and
#      VARIANTS=fixed respectively, or
#   b) Remove 4 workers and let the remaining 4 each handle both variants
#      (double runtime per GPU; use resubmission to complete).

# ==========================================
# CONFIGURATION
# ==========================================
PYTHON_VERSION=3.10
ENVIRONMENT_NAME="toposentinel"
TORCH_CUDA="${TORCH_CUDA:-cu124}"

# Python-side time budget — workers stop voluntarily before the hard SLURM
# limit so the summary pass completes cleanly.  Set ~2 h below --time.
TIME_LIMIT_HOURS="${TIME_LIMIT_HOURS:-138}"

RESULTS_DIR="${RESULTS_DIR:-results}"

# Optional subsetting — leave empty to run all
ATTACKS="${ATTACKS:-}"
SEEDS="${SEEDS:-}"
# Override to run only one variant: VARIANTS="dkw" or VARIANTS="fixed"
VARIANTS="${VARIANTS:-}"

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

mkdir -p logs "$RESULTS_DIR/toposentinel_ablation"

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
[ -n "$ATTACKS"  ] && COMMON="$COMMON --attacks $ATTACKS"
[ -n "$SEEDS"    ] && COMMON="$COMMON --seeds $SEEDS"
[ -n "$VARIANTS" ] && COMMON="$COMMON --variants $VARIANTS"
[ "$SUMMARIZE_ONLY" -eq 1 ] && COMMON="$COMMON --summarize-only"

# ==========================================
# PARALLEL EXECUTION — 8 GPUS
# ==========================================
# Layout: one GPU per (dataset, variant) pair.
# Each worker runs the full attack × seed matrix for its slice and stops
# gracefully when the time budget runs out.  Already-complete runs are
# skipped on resubmission.
#
# Estimated wall time per GPU (4 attacks × 10 seeds):
#   GPU 0  femnist  / dkw    ~33 h
#   GPU 1  gtsrb    / dkw    ~60 h
#   GPU 2  cifar10  / dkw   ~120 h
#   GPU 3  cifar100 / dkw   ~133 h
#   GPU 4  femnist  / fixed  ~33 h
#   GPU 5  gtsrb    / fixed  ~60 h
#   GPU 6  cifar10  / fixed ~120 h
#   GPU 7  cifar100 / fixed ~133 h

echo ""
echo "Starting TopoSentinel ablation on 8 GPUs..."
echo "$(date '+%Y-%m-%d %H:%M:%S')"

# ---- DKW variant (GPUs 0-3) ------------------------------------------------

# --- GPU 0: FEMNIST / DKW ---
(
    echo "[GPU 0] Starting femnist / dkw"
    export CUDA_VISIBLE_DEVICES=0
    python -m experiments.benchmark.run_toposentinel_ablation \
        --datasets femnist \
        --variants dkw \
        $COMMON \
        > logs/topo_ablation_gpu0_${SLURM_JOB_ID}.log 2>&1
    echo "[GPU 0] Done"
) &
PID0=$!

# --- GPU 1: GTSRB / DKW ---
(
    echo "[GPU 1] Starting gtsrb / dkw"
    export CUDA_VISIBLE_DEVICES=1
    python -m experiments.benchmark.run_toposentinel_ablation \
        --datasets gtsrb \
        --variants dkw \
        $COMMON \
        > logs/topo_ablation_gpu1_${SLURM_JOB_ID}.log 2>&1
    echo "[GPU 1] Done"
) &
PID1=$!

# --- GPU 2: CIFAR-10 / DKW ---
(
    echo "[GPU 2] Starting cifar10 / dkw"
    export CUDA_VISIBLE_DEVICES=2
    python -m experiments.benchmark.run_toposentinel_ablation \
        --datasets cifar10 \
        --variants dkw \
        $COMMON \
        > logs/topo_ablation_gpu2_${SLURM_JOB_ID}.log 2>&1
    echo "[GPU 2] Done"
) &
PID2=$!

# --- GPU 3: CIFAR-100 / DKW ---
(
    echo "[GPU 3] Starting cifar100 / dkw"
    export CUDA_VISIBLE_DEVICES=3
    python -m experiments.benchmark.run_toposentinel_ablation \
        --datasets cifar100 \
        --variants dkw \
        $COMMON \
        > logs/topo_ablation_gpu3_${SLURM_JOB_ID}.log 2>&1
    echo "[GPU 3] Done"
) &
PID3=$!

# ---- Fixed variant (GPUs 4-7) ----------------------------------------------

# --- GPU 4: FEMNIST / FIXED ---
(
    echo "[GPU 4] Starting femnist / fixed"
    export CUDA_VISIBLE_DEVICES=4
    python -m experiments.benchmark.run_toposentinel_ablation \
        --datasets femnist \
        --variants fixed \
        $COMMON \
        > logs/topo_ablation_gpu4_${SLURM_JOB_ID}.log 2>&1
    echo "[GPU 4] Done"
) &
PID4=$!

# --- GPU 5: GTSRB / FIXED ---
(
    echo "[GPU 5] Starting gtsrb / fixed"
    export CUDA_VISIBLE_DEVICES=5
    python -m experiments.benchmark.run_toposentinel_ablation \
        --datasets gtsrb \
        --variants fixed \
        $COMMON \
        > logs/topo_ablation_gpu5_${SLURM_JOB_ID}.log 2>&1
    echo "[GPU 5] Done"
) &
PID5=$!

# --- GPU 6: CIFAR-10 / FIXED ---
(
    echo "[GPU 6] Starting cifar10 / fixed"
    export CUDA_VISIBLE_DEVICES=6
    python -m experiments.benchmark.run_toposentinel_ablation \
        --datasets cifar10 \
        --variants fixed \
        $COMMON \
        > logs/topo_ablation_gpu6_${SLURM_JOB_ID}.log 2>&1
    echo "[GPU 6] Done"
) &
PID6=$!

# --- GPU 7: CIFAR-100 / FIXED ---
(
    echo "[GPU 7] Starting cifar100 / fixed"
    export CUDA_VISIBLE_DEVICES=7
    python -m experiments.benchmark.run_toposentinel_ablation \
        --datasets cifar100 \
        --variants fixed \
        $COMMON \
        > logs/topo_ablation_gpu7_${SLURM_JOB_ID}.log 2>&1
    echo "[GPU 7] Done"
) &
PID7=$!

# Wait for all workers
wait $PID0 $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7

echo ""
echo "All workers finished."
echo "$(date '+%Y-%m-%d %H:%M:%S')"

# Show last few lines of each GPU log for quick inspection
for gpu in 0 1 2 3 4 5 6 7; do
    logfile="logs/topo_ablation_gpu${gpu}_${SLURM_JOB_ID}.log"
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
echo "Generating merged toposentinel_ablation_summary.csv..."
python -m experiments.benchmark.run_toposentinel_ablation \
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
echo " Summary    : $RESULTS_DIR/toposentinel_ablation_summary.csv"
echo " Run dirs   : $(ls -1 $RESULTS_DIR/toposentinel_ablation/ 2>/dev/null | wc -l)"
echo " Per-GPU logs:"
ls -lh logs/topo_ablation_gpu*_${SLURM_JOB_ID}.log 2>/dev/null || true
echo "============================================================"
