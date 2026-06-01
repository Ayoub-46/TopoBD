#!/bin/bash
# =============================================================================
# SLURM job: train all CIFAR-10 prerequisite FL experiments then (optionally)
# run gradient-alignment diagnostics.
#
# Parallelism — 4 A6000 GPUs:
#   GPU 0  cifar10_benign_iid  +  cifar10_fedavg_neurotoxin   (≈3 h + ≈3 h = ≈6 h)
#   GPU 1  cifar10_fedavg_a3fl                                 (≈4 h)
#   GPU 2  cifar10_fedavg_iba                                  (≈4 h)
#   GPU 3  cifar10_fedavg_chameleon                            (≈4 h)
#
# The benign and neurotoxin runs are placed together on GPU 0 because they are
# the two lightest runs (~3 h each); all other GPUs carry a single heavier run.
# Expected wall-clock: ≈6 h (bottleneck is GPU 0).
#
# Resubmit the same sbatch command to resume — already-present checkpoints are
# skipped automatically (final_model.pt existence check in train_prerequisites.py).
#
# Usage
# -----
#   sbatch experiments/gradient_alignment/slurm_cifar10_prerequisites.sh
#
# Optional overrides via --export:
#   sbatch --export=ALL,FORCE_RETRAIN=1 \
#          experiments/gradient_alignment/slurm_cifar10_prerequisites.sh
#
#   sbatch --export=ALL,RUN_DIAGNOSTICS=1,N_BATCHES=200,N_PER_CLASS=200 \
#          experiments/gradient_alignment/slurm_cifar10_prerequisites.sh
# =============================================================================

# ---- SLURM directives -------------------------------------------------------
#SBATCH --job-name=cifar10_prerequisites
#SBATCH --partition=gpu-a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32          # 8 per GPU worker + headroom for DataLoader
#SBATCH --mem=128GB                  # ~32 GB per worker
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00             # bottleneck ≈6 h; 6 h margin for slow I/O
#SBATCH --output=logs/cifar10_prereq_%j.out
#SBATCH --error=logs/cifar10_prereq_%j.err

# ---- Configuration (override via --export or edit here) ---------------------
ENVIRONMENT_NAME="${ENVIRONMENT_NAME:-toposentinel}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"

FORCE_RETRAIN="${FORCE_RETRAIN:-0}"      # 1 = re-train even if checkpoint exists

# Set to 1 to run gradient-alignment diagnostics immediately after training.
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-0}"
N_BATCHES="${N_BATCHES:-200}"            # diagnostic batches per attack
BATCH_SIZE="${BATCH_SIZE:-64}"
N_PER_CLASS="${N_PER_CLASS:-200}"        # samples per class for Experiment 2
EXTENDED_LEMMA2="${EXTENDED_LEMMA2:-1}"  # 1 = also run the rank-comparison experiment

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
echo " GPUs     : $(nvidia-smi --query-gpu=name --format=csv,noheader \
                    2>/dev/null | paste -sd ',' || echo 'n/a')"
echo "============================================================"

mkdir -p logs results experiments/gradient_alignment/outputs_cifar10

module load Anaconda3 2>/dev/null || module load anaconda3 2>/dev/null || true

if ! conda info --envs | grep -qE "^${ENVIRONMENT_NAME}\s"; then
    echo "[setup] Creating conda environment '${ENVIRONMENT_NAME}' (Python ${PYTHON_VERSION})..."
    conda create -n "${ENVIRONMENT_NAME}" python="${PYTHON_VERSION}" -y
fi

CONDA_RUN="conda run -n ${ENVIRONMENT_NAME} --no-capture-output"
TORCH_WHEEL_URL="https://download.pytorch.org/whl/${TORCH_CUDA}"

echo "[setup] Installing PyTorch (${TORCH_CUDA})..."
$CONDA_RUN pip install \
    torch==2.5.1+${TORCH_CUDA} \
    torchvision==0.20.1+${TORCH_CUDA} \
    --index-url "${TORCH_WHEEL_URL}" -q

echo "[setup] Installing project dependencies..."
$CONDA_RUN pip install -r requirements.txt -q

set -e

echo "[setup] Python : $($CONDA_RUN python --version)"
echo "[setup] PyTorch: $($CONDA_RUN python -c 'import torch; print(torch.__version__)')"
echo "[setup] CUDA   : $($CONDA_RUN python -c 'import torch; print(torch.cuda.is_available())')"

# =============================================================================
# 2. DATA: COPY CIFAR-10 TO LOCAL SCRATCH FOR FAST I/O
# =============================================================================
# Each of the 4 workers reads the same dataset.  Copying to local NVMe once
# avoids contention on the network filesystem.

REPO_DATA="$(pwd)/data"
SCRATCH="${SLURM_TMPDIR:-/tmp/${USER:-user}/$SLURM_JOB_ID}"
SCRATCH_DATA="$SCRATCH/data"
mkdir -p "$SCRATCH_DATA"

_ORIGINAL_DATA_WAS_DIR=0
cleanup() {
    echo "[cleanup] Restoring data directory and removing scratch..."
    [ -L "data" ] && rm -f data
    if [ "$_ORIGINAL_DATA_WAS_DIR" -eq 1 ] && [ -d "${REPO_DATA}_network_backup" ]; then
        mv "${REPO_DATA}_network_backup" "$REPO_DATA"
    fi
    rm -rf "$SCRATCH"
    echo "[cleanup] Done."
}
trap cleanup EXIT

# CIFAR-10 is stored in data/cifar-10-batches-py/ after torchvision extraction.
# Copy the whole data/ tree — workers may also download GTSRB or other datasets
# if not yet present, and they need a writable scratch location for that.
if [ -d "$REPO_DATA" ]; then
    echo "[data] Copying data/ to scratch ($(du -sh "$REPO_DATA" | cut -f1))..."
    T0=$(date +%s)
    cp -r "$REPO_DATA/." "$SCRATCH_DATA/"
    echo "[data] Copy complete in $(( $(date +%s) - T0 ))s"
    _ORIGINAL_DATA_WAS_DIR=1
    mv "$REPO_DATA" "${REPO_DATA}_network_backup"
    ln -s "$SCRATCH_DATA" "$REPO_DATA"
    echo "[data] data/ → scratch (fast path)"
else
    echo "[data] Warning: data/ not found — CIFAR-10 will be downloaded on first run."
fi

# =============================================================================
# 3. BUILD COMMON TRAINING ARGUMENTS
# =============================================================================
TRAIN_BASE="experiments/gradient_alignment/train_prerequisites.py --dataset cifar10"
[ "$FORCE_RETRAIN" -eq 1 ] && TRAIN_BASE="$TRAIN_BASE --force"

# =============================================================================
# 4. PARALLEL TRAINING — 4 GPUS
# =============================================================================
echo ""
echo "============================================================"
echo " Training Phase — 4 parallel GPU workers"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Forward SIGTERM from SLURM to all child workers so they stop cleanly.
_pids=""
_forward_sigterm() {
    echo "[main] SIGTERM received — forwarding to training workers..."
    # shellcheck disable=SC2086
    kill -TERM $_pids 2>/dev/null || true
}
trap '_forward_sigterm' TERM

T_TRAIN_START=$(date +%s)

# GPU 0 — benign baseline then neurotoxin (lightest pair, run sequentially)
echo "[gpu0] Starting cifar10_benign_iid then cifar10_fedavg_neurotoxin..."
CUDA_VISIBLE_DEVICES=0 $CONDA_RUN python $TRAIN_BASE \
    --runs cifar10_benign_iid cifar10_fedavg_neurotoxin \
    > "logs/cifar10_prereq_gpu0_${SLURM_JOB_ID}.log" 2>&1 &
PID0=$!

# GPU 1 — A3FL (learnable trigger, trigger optimisation overhead)
echo "[gpu1] Starting cifar10_fedavg_a3fl..."
CUDA_VISIBLE_DEVICES=1 $CONDA_RUN python $TRAIN_BASE \
    --runs cifar10_fedavg_a3fl \
    > "logs/cifar10_prereq_gpu1_${SLURM_JOB_ID}.log" 2>&1 &
PID1=$!

# GPU 2 — IBA (U-Net generator training overhead)
echo "[gpu2] Starting cifar10_fedavg_iba..."
CUDA_VISIBLE_DEVICES=2 $CONDA_RUN python $TRAIN_BASE \
    --runs cifar10_fedavg_iba \
    > "logs/cifar10_prereq_gpu2_${SLURM_JOB_ID}.log" 2>&1 &
PID2=$!

# GPU 3 — Chameleon (per-round PGD overhead)
echo "[gpu3] Starting cifar10_fedavg_chameleon..."
CUDA_VISIBLE_DEVICES=3 $CONDA_RUN python $TRAIN_BASE \
    --runs cifar10_fedavg_chameleon \
    > "logs/cifar10_prereq_gpu3_${SLURM_JOB_ID}.log" 2>&1 &
PID3=$!

_pids="$PID0 $PID1 $PID2 $PID3"

# Collect exit codes
EXIT0=0; EXIT1=0; EXIT2=0; EXIT3=0
wait $PID0 || EXIT0=$?
wait $PID1 || EXIT1=$?
wait $PID2 || EXIT2=$?
wait $PID3 || EXIT3=$?

T_TRAIN_ELAPSED=$(( $(date +%s) - T_TRAIN_START ))
printf "[training] All workers finished in %dh %dm %ds\n" \
    $((T_TRAIN_ELAPSED/3600)) $(((T_TRAIN_ELAPSED%3600)/60)) $((T_TRAIN_ELAPSED%60))
echo "[training] Exit codes — gpu0=$EXIT0  gpu1=$EXIT1  gpu2=$EXIT2  gpu3=$EXIT3"

# Show per-GPU tail in the main log for quick debugging
for gpu in 0 1 2 3; do
    logfile="logs/cifar10_prereq_gpu${gpu}_${SLURM_JOB_ID}.log"
    if [ -f "$logfile" ]; then
        echo ""
        echo "--- GPU ${gpu} last 5 lines ---"
        tail -5 "$logfile"
    fi
done

# Report which checkpoints are now present
echo ""
echo "[checkpoints] Status after training:"
for run in cifar10_benign_iid cifar10_fedavg_neurotoxin \
           cifar10_fedavg_a3fl cifar10_fedavg_iba cifar10_fedavg_chameleon; do
    ckpt="results/${run}/final_model.pt"
    if [ -f "$ckpt" ]; then
        echo "  [OK]     $ckpt"
    else
        echo "  [MISSING] $ckpt"
    fi
done

# =============================================================================
# 5. OPTIONAL DIAGNOSTICS
# =============================================================================
if [ "$RUN_DIAGNOSTICS" -eq 1 ]; then
    echo ""
    echo "============================================================"
    echo " Diagnostics Phase — gradient alignment on CIFAR-10"
    echo " $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    DIAG_ARGS="--config          configs/cifar10_fedavg_iba.yaml
               --benign-checkpoint results/cifar10_benign_iid/final_model.pt
               --attacks         neurotoxin a3fl iba chameleon
               --n-batches       ${N_BATCHES}
               --batch-size      ${BATCH_SIZE}
               --n-per-class     ${N_PER_CLASS}
               --output-dir      experiments/gradient_alignment/outputs_cifar10
               --results-dir     results
               --device          cuda"

    [ "$EXTENDED_LEMMA2" -eq 1 ] && DIAG_ARGS="$DIAG_ARGS --extended-lemma2"

    T_DIAG_START=$(date +%s)

    # Run diagnostics on a single GPU (GPU 0 is free — its training finished last)
    set +e
    CUDA_VISIBLE_DEVICES=0 $CONDA_RUN python -m experiments.gradient_alignment.run_all \
        $DIAG_ARGS
    DIAG_EXIT=$?
    set -e

    T_DIAG_ELAPSED=$(( $(date +%s) - T_DIAG_START ))
    printf "[diagnostics] Finished in %dh %dm %ds  (exit=%d)\n" \
        $((T_DIAG_ELAPSED/3600)) $(((T_DIAG_ELAPSED%3600)/60)) \
        $((T_DIAG_ELAPSED%60)) "$DIAG_EXIT"

    echo ""
    echo "[diagnostics] Output files:"
    ls -lh experiments/gradient_alignment/outputs_cifar10/ 2>/dev/null \
        | grep -v '^total' || echo "  (none)"
else
    echo ""
    echo "[diagnostics] Skipped (set RUN_DIAGNOSTICS=1 to enable)."
    echo "[diagnostics] Run manually after training:"
    echo ""
    echo "  python -m experiments.gradient_alignment.run_all \\"
    echo "      --config        configs/cifar10_fedavg_iba.yaml \\"
    echo "      --benign-checkpoint results/cifar10_benign_iid/final_model.pt \\"
    echo "      --attacks       neurotoxin a3fl iba chameleon \\"
    echo "      --n-batches     ${N_BATCHES} --batch-size ${BATCH_SIZE} \\"
    echo "      --n-per-class   ${N_PER_CLASS} --extended-lemma2 \\"
    echo "      --output-dir    experiments/gradient_alignment/outputs_cifar10 \\"
    echo "      --device        cuda"
fi

# =============================================================================
# 6. JOB SUMMARY
# =============================================================================
echo ""
echo "============================================================"
echo " Job complete"
echo " End : $(date '+%Y-%m-%d %H:%M:%S')"
echo " Per-GPU logs:"
ls -lh logs/cifar10_prereq_gpu*_"${SLURM_JOB_ID}".log 2>/dev/null || true
echo "============================================================"

# Fail the job if any training worker failed
[ $EXIT0 -eq 0 ] && [ $EXIT1 -eq 0 ] && [ $EXIT2 -eq 0 ] && [ $EXIT3 -eq 0 ] || exit 1
