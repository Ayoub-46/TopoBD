#!/bin/bash
# =============================================================================
# SLURM job: train all prerequisite FL experiments then run gradient-alignment
# diagnostics.  Only GTSRB.  All five training runs are sequential on one GPU
# (each occupies the full device — parallelism would require multiple nodes).
#
# Usage
# -----
#   sbatch experiments/gradient_alignment/slurm_run.sh
#
# Optional overrides via --export:
#   sbatch --export=ALL,SKIP_TRAINING=1 \
#          experiments/gradient_alignment/slurm_run.sh
#   sbatch --export=ALL,FORCE_RETRAIN=1 \
#          experiments/gradient_alignment/slurm_run.sh
# =============================================================================

# ---- SLURM directives -------------------------------------------------------
#SBATCH --job-name=toposentinel_gtsrb
#SBATCH --partition=gpu-a6000          # adjust to your cluster's GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # 4 DataLoader workers per run + headroom
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1                   # all runs are sequential → one GPU
#SBATCH --time=48:00:00               # benign+neurotoxin skip; A3FL+IBA+Chameleon ≈ 10-24 h
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ---- Configuration (override via --export or edit here) ---------------------
ENVIRONMENT_NAME="${ENVIRONMENT_NAME:-toposentinel}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"      # set to 1 to jump straight to diagnostics
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"      # set to 1 to re-train already-trained runs
N_BATCHES="${N_BATCHES:-50}"             # diagnostic batches per attack
BATCH_SIZE="${BATCH_SIZE:-64}"           # diagnostic batch size
N_PER_CLASS="${N_PER_CLASS:-100}"        # samples per class for Experiment 2

# PyTorch CUDA wheel suffix — must match the CUDA version on the cluster nodes.
# Check with: nvidia-smi | grep "CUDA Version"
# Common values: cu118  cu121  cu124  cu126
TORCH_CUDA="${TORCH_CUDA:-cu124}"
TORCH_WHEEL_URL="https://download.pytorch.org/whl/${TORCH_CUDA}"

# ---- Strict error handling --------------------------------------------------
# Note: set -e is placed AFTER conda activation because conda's shell hooks
# return non-zero in some environments even on success.
set -uo pipefail

# =============================================================================
# 1. ENVIRONMENT SETUP
# =============================================================================
echo "============================================================"
echo " Job ID   : $SLURM_JOB_ID"
echo " Node     : $(hostname)"
echo " Start    : $(date '+%Y-%m-%d %H:%M:%S')"
echo " Workdir  : $(pwd)"
echo "============================================================"

mkdir -p logs results experiments/gradient_alignment/outputs

# Load Anaconda module (adjust the module name to match your cluster)
module load Anaconda3 2>/dev/null || module load anaconda3 2>/dev/null || true

# Create conda environment if it doesn't exist
if ! conda info --envs | grep -qE "^${ENVIRONMENT_NAME}\s"; then
    echo "[setup] Creating conda environment '${ENVIRONMENT_NAME}' (Python ${PYTHON_VERSION})..."
    conda create -n "${ENVIRONMENT_NAME}" python="${PYTHON_VERSION}" -y
fi

# Activate via 'conda run' to avoid sourcing conda's init script, which is
# fragile in non-interactive SLURM shells.  All python commands below are
# wrapped in 'conda run'.
CONDA_RUN="conda run -n ${ENVIRONMENT_NAME} --no-capture-output"

# Install PyTorch with the correct CUDA build first.
# Plain PyPI only serves CPU wheels, so we must point pip at the PyTorch index.
# This is a no-op if the right version is already installed.
echo "[setup] Installing PyTorch (${TORCH_CUDA})..."
$CONDA_RUN pip install \
    torch==2.5.1+${TORCH_CUDA} \
    torchvision==0.20.1+${TORCH_CUDA} \
    --index-url "${TORCH_WHEEL_URL}"

# Install the remaining dependencies from the flat requirements.txt.
# Using requirements.txt avoids invoking the pyproject.toml build backend
# (which requires a specific setuptools version not present on all clusters).
echo "[setup] Installing project dependencies (requirements.txt)..."
$CONDA_RUN pip install -r requirements.txt

# Enable strict errors now that installation is done
set -e

echo "[setup] Python: $($CONDA_RUN python --version)"
echo "[setup] PyTorch: $($CONDA_RUN python -c 'import torch; print(torch.__version__)')"
echo "[setup] CUDA available: $($CONDA_RUN python -c 'import torch; print(torch.cuda.is_available())')"
echo "[setup] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not found')"

# =============================================================================
# 2. DATA: COPY GTSRB TO LOCAL SCRATCH FOR FAST I/O
# =============================================================================
# Network filesystems (NFS/Lustre) are the #1 bottleneck for small-file I/O
# during DataLoader workers.  Copying to local NVMe before training is
# consistently 2-5× faster on GTSRB.

REPO_DATA="$(pwd)/data"

# Prefer SLURM's designated scratch ($SLURM_TMPDIR on most clusters).
# Fall back to /tmp when not set (e.g. interactive nodes).
if [ -n "${SLURM_TMPDIR:-}" ]; then
    SCRATCH="$SLURM_TMPDIR"
else
    SCRATCH="/tmp/${USER:-user}/$SLURM_JOB_ID"
fi

SCRATCH_DATA="$SCRATCH/data"
mkdir -p "$SCRATCH_DATA"

# Register a cleanup trap: restore the original data symlink/directory and
# wipe scratch on EXIT (whether success, error, or SIGTERM from SLURM).
_ORIGINAL_DATA_WAS_DIR=0
cleanup() {
    echo "[cleanup] Restoring data directory and removing scratch..."
    # Remove the symlink we created
    if [ -L "data" ]; then
        rm -f data
    fi
    # Restore the original directory if we moved it
    if [ "$_ORIGINAL_DATA_WAS_DIR" -eq 1 ] && [ -d "${REPO_DATA}_network_backup" ]; then
        mv "${REPO_DATA}_network_backup" "$REPO_DATA"
    fi
    rm -rf "$SCRATCH"
    echo "[cleanup] Done."
}
trap cleanup EXIT

# Copy only GTSRB (≈550 MB) — skip CIFAR/MNIST which we don't use here.
if [ -d "$REPO_DATA/gtsrb" ]; then
    echo "[data] Copying GTSRB to scratch: $SCRATCH_DATA/gtsrb ..."
    T_COPY_START=$(date +%s)
    cp -r "$REPO_DATA/gtsrb" "$SCRATCH_DATA/"
    T_COPY_END=$(date +%s)
    echo "[data] Copy complete in $((T_COPY_END - T_COPY_START))s"

    # Replace ./data with a symlink to scratch so all code sees the fast copy.
    # Preserve other subdirectories (CIFAR, MNIST) via a partial approach:
    # move the original data dir aside and create a new symlink.
    _ORIGINAL_DATA_WAS_DIR=1
    mv "$REPO_DATA" "${REPO_DATA}_network_backup"
    ln -s "$SCRATCH_DATA" "$REPO_DATA"

    # Copy non-GTSRB subdirs back into scratch so they remain accessible
    for d in "${REPO_DATA}_network_backup"/*/; do
        subdir=$(basename "$d")
        if [ "$subdir" != "gtsrb" ] && [ ! -e "$SCRATCH_DATA/$subdir" ]; then
            cp -r "$d" "$SCRATCH_DATA/$subdir"
        fi
    done
    echo "[data] data/ now points to scratch (fast path)"
else
    echo "[data] Warning: data/gtsrb not found — data will be downloaded on first run."
fi

# =============================================================================
# 3. TRAINING PHASE
# =============================================================================
echo ""
echo "============================================================"
echo " Phase 1 — FL Training"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

if [ "$SKIP_TRAINING" -eq 1 ]; then
    echo "[training] SKIP_TRAINING=1 — skipping to diagnostics."
else
    # Build the argument list for train_prerequisites.py
    TRAIN_ARGS="experiments/gradient_alignment/train_prerequisites.py"
    if [ "$FORCE_RETRAIN" -eq 1 ]; then
        TRAIN_ARGS="$TRAIN_ARGS --force"
    fi

    # Run sequentially — each FL experiment uses the whole GPU.
    # Logs are emitted to the SLURM output file in real time.
    T_TRAIN_START=$(date +%s)

    # Temporarily disable set -e so a training failure doesn't kill the whole
    # job — the diagnostic phase should still run on whatever checkpoints exist.
    set +e
    $CONDA_RUN python $TRAIN_ARGS
    TRAIN_EXIT=$?
    set -e

    T_TRAIN_END=$(date +%s)
    TRAIN_ELAPSED=$((T_TRAIN_END - T_TRAIN_START))
    echo "[training] Finished in $(printf '%dh %dm %ds' \
        $((TRAIN_ELAPSED/3600)) $(((TRAIN_ELAPSED%3600)/60)) $((TRAIN_ELAPSED%60)))"

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "[training] ERROR: train_prerequisites.py exited with code $TRAIN_EXIT"
        echo "[training] Diagnostics will proceed with whatever checkpoints exist."
    fi
fi

# =============================================================================
# 4. DIAGNOSTIC PHASE
# =============================================================================
echo ""
echo "============================================================"
echo " Phase 2 — Gradient Alignment Diagnostics"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

T_DIAG_START=$(date +%s)

$CONDA_RUN python -m experiments.gradient_alignment.run_all \
    --config          configs/gtsrb_fedavg_iba.yaml \
    --benign-checkpoint results/gtsrb_benign_iid/final_model.pt \
    --attacks         neurotoxin a3fl iba chameleon \
    --n-batches       "$N_BATCHES" \
    --batch-size      "$BATCH_SIZE" \
    --n-per-class     "$N_PER_CLASS" \
    --output-dir      experiments/gradient_alignment/outputs \
    --results-dir     results \
    --device          cuda

T_DIAG_END=$(date +%s)
DIAG_ELAPSED=$((T_DIAG_END - T_DIAG_START))
echo "[diagnostics] Finished in $(printf '%dh %dm %ds' \
    $((DIAG_ELAPSED/3600)) $(((DIAG_ELAPSED%3600)/60)) $((DIAG_ELAPSED%60)))"

# =============================================================================
# 5. SUMMARY
# =============================================================================
echo ""
echo "============================================================"
echo " Job complete"
echo " End : $(date '+%Y-%m-%d %H:%M:%S')"
echo " Output files:"
ls -lh experiments/gradient_alignment/outputs/ 2>/dev/null | grep -v '^total' || true
echo "============================================================"
