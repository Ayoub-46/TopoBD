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
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#
# One dataset per job on a single GPU (robust + easy to schedule). To run the
# full matrix, submit one job per dataset — they run in parallel across the
# cluster, no 4-GPU node required:
#
#   for d in femnist gtsrb cifar10 cifar100; do
#       sbatch --job-name=topo_$d --export=ALL,DATASET=$d slurm_toposentinel.sh
#   done
#
# Each job stops gracefully before the SLURM wall-clock limit and skips
# already-complete runs, so a dataset that needs more than one job just gets
# resubmitted and resumes where it left off.

# ==========================================
# CONFIGURATION
# ==========================================
PYTHON_VERSION=3.10
ENVIRONMENT_NAME="toposentinel"
TORCH_CUDA="${TORCH_CUDA:-cu124}"

# Python-side time budget — the runner stops voluntarily before the hard SLURM
# limit so the summary pass completes cleanly.  Set ~2 h below --time.
TIME_LIMIT_HOURS="${TIME_LIMIT_HOURS:-98}"

RESULTS_DIR="${RESULTS_DIR:-results}"

# Dataset for this job. Leave empty to run ALL datasets sequentially (resumable
# across resubmissions); set e.g. DATASET=gtsrb for one-dataset-per-job.
DATASET="${DATASET:-}"

# Optional subsetting — leave empty to run all attacks / seeds
ATTACKS="${ATTACKS:-}"
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
# BUILD ARGUMENTS
# ==========================================
COMMON="--results-dir $RESULTS_DIR --time-limit-hours $TIME_LIMIT_HOURS --device cuda"
[ -n "$DATASET" ]           && COMMON="$COMMON --datasets $DATASET"
[ -n "$ATTACKS" ]           && COMMON="$COMMON --attacks $ATTACKS"
[ -n "$SEEDS" ]             && COMMON="$COMMON --seeds $SEEDS"
[ "$SUMMARIZE_ONLY" -eq 1 ] && COMMON="$COMMON --summarize-only"

# ==========================================
# EXECUTION — single GPU, one dataset (or all if DATASET is empty)
# ==========================================
# The runner iterates the (dataset ×) attack × seed matrix shortest-first,
# skips already-complete runs, and stops gracefully before the time budget so
# the summary pass always completes.  Resubmit to resume.
#
# Per-dataset estimate (4 attacks × 10 seeds, ~min/run): femnist ~33 h,
# gtsrb ~60 h, cifar10 ~120 h, cifar100 ~133 h — hence one-job-per-dataset.

echo ""
echo "Starting TopoSentinel benchmark  (DATASET='${DATASET:-ALL}')"
echo "$(date '+%Y-%m-%d %H:%M:%S')"

RUN_LOG="logs/topo_benchmark_${SLURM_JOB_ID}.log"
python -m experiments.benchmark.run_toposentinel $COMMON 2>&1 | tee "$RUN_LOG"

echo ""
echo "Run finished. $(date '+%Y-%m-%d %H:%M:%S')"

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
echo " Dataset    : ${DATASET:-ALL}"
echo " Summary    : $RESULTS_DIR/toposentinel_summary.csv"
echo " Run dirs   : $(ls -1 $RESULTS_DIR/toposentinel_benchmark/ 2>/dev/null | wc -l)"
echo " Run log    : $RUN_LOG"
echo "============================================================"
