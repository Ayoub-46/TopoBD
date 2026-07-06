#!/bin/bash

# ==========================================
# SLURM DIRECTIVES
# ==========================================
#SBATCH --partition=gpu-a6000
#SBATCH --time=16:00:00              # bottleneck ≈2.5 h (GPU 0); 3.5 h margin
#SBATCH --job-name=femnist_prereqs
#SBATCH --error=logs/femnist_prereq_%j.err
#SBATCH --output=logs/femnist_prereq_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32          # 8 per GPU worker
#SBATCH --mem=64GB                   # LeNet5 is lightweight; 16 GB per worker is enough
#SBATCH --gres=gpu:4

# ==========================================
# CONFIGURATION
# ==========================================
PYTHON_VERSION=3.10
ENVIRONMENT_NAME="toposentinel"
TORCH_CUDA="${TORCH_CUDA:-cu124}"

FORCE_RETRAIN="${FORCE_RETRAIN:-0}"

# Set to 1 to run gradient-alignment diagnostics immediately after training
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"
N_BATCHES="${N_BATCHES:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"
N_PER_CLASS="${N_PER_CLASS:-200}"
EXTENDED_LEMMA2="${EXTENDED_LEMMA2:-1}"

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

mkdir -p logs results experiments/gradient_alignment/outputs_femnist

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
# FEMNIST uses EMNIST 'byclass' (~560 MB); copying avoids NFS contention
# across the 4 concurrent DataLoader workers.

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
    echo "Warning: 'data' folder not found — EMNIST/FEMNIST will be downloaded on first run."
fi

# ==========================================
# BUILD COMMON TRAINING ARGUMENTS
# ==========================================
TRAIN_BASE="experiments/gradient_alignment/train_prerequisites.py --dataset femnist"
[ "$FORCE_RETRAIN" -eq 1 ] && TRAIN_BASE="$TRAIN_BASE --force"

# ==========================================
# PARALLEL TRAINING — 4 GPUS
# ==========================================
# GPU assignment (LeNet5 is fast; all runs finish well under 3 h):
#   GPU 0 — benign_iid + fedavg_neurotoxin   ≈1 h + ≈1.5 h = ≈2.5 h  (bottleneck)
#   GPU 1 — fedavg_a3fl                       ≈2.5 h
#   GPU 2 — fedavg_iba                        ≈2.5 h
#   GPU 3 — fedavg_chameleon                  ≈2.5 h

echo ""
echo "Starting parallel training on 4 GPUs..."
echo "$(date '+%Y-%m-%d %H:%M:%S')"

# --- GPU 0: benign baseline then neurotoxin ---
(
    echo "[GPU 0] Starting femnist_benign_iid then femnist_fedavg_neurotoxin"
    export CUDA_VISIBLE_DEVICES=0
    python $TRAIN_BASE \
        --runs femnist_benign_iid femnist_fedavg_neurotoxin \
        > logs/femnist_prereq_gpu0_${SLURM_JOB_ID}.log 2>&1
    echo "[GPU 0] Done"
) &
PID0=$!

# --- GPU 1: A3FL ---
(
    echo "[GPU 1] Starting femnist_fedavg_a3fl"
    export CUDA_VISIBLE_DEVICES=1
    python $TRAIN_BASE \
        --runs femnist_fedavg_a3fl \
        > logs/femnist_prereq_gpu1_${SLURM_JOB_ID}.log 2>&1
    echo "[GPU 1] Done"
) &
PID1=$!

# --- GPU 2: IBA ---
(
    echo "[GPU 2] Starting femnist_fedavg_iba"
    export CUDA_VISIBLE_DEVICES=2
    python $TRAIN_BASE \
        --runs femnist_fedavg_iba \
        > logs/femnist_prereq_gpu2_${SLURM_JOB_ID}.log 2>&1
    echo "[GPU 2] Done"
) &
PID2=$!

# --- GPU 3: Chameleon ---
(
    echo "[GPU 3] Starting femnist_fedavg_chameleon"
    export CUDA_VISIBLE_DEVICES=3
    python $TRAIN_BASE \
        --runs femnist_fedavg_chameleon \
        > logs/femnist_prereq_gpu3_${SLURM_JOB_ID}.log 2>&1
    echo "[GPU 3] Done"
) &
PID3=$!

# Wait for all workers
wait $PID0 $PID1 $PID2 $PID3

echo ""
echo "All training workers finished."
echo "$(date '+%Y-%m-%d %H:%M:%S')"

# Show last few lines of each GPU log for quick inspection
for gpu in 0 1 2 3; do
    logfile="logs/femnist_prereq_gpu${gpu}_${SLURM_JOB_ID}.log"
    if [ -f "$logfile" ]; then
        echo ""
        echo "--- GPU ${gpu} last 5 lines ---"
        tail -5 "$logfile"
    fi
done

# Report checkpoint status
echo ""
echo "Checkpoint status:"
for run in femnist_benign_iid femnist_fedavg_neurotoxin \
           femnist_fedavg_a3fl femnist_fedavg_iba femnist_fedavg_chameleon; do
    ckpt="results/${run}/final_model.pt"
    [ -f "$ckpt" ] && echo "  [OK]      $ckpt" || echo "  [MISSING] $ckpt"
done

# ==========================================
# OPTIONAL DIAGNOSTICS
# ==========================================
if [ "$RUN_DIAGNOSTICS" -eq 1 ]; then
    echo ""
    echo "============================================================"
    echo " Diagnostics Phase"
    echo " $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    DIAG_FLAGS="--extended-lemma2"
    [ "$EXTENDED_LEMMA2" -eq 0 ] && DIAG_FLAGS=""

    export CUDA_VISIBLE_DEVICES=0
    python -m experiments.gradient_alignment.run_all \
        --config            configs/femnist_fedavg_iba.yaml \
        --benign-checkpoint results/femnist_benign_iid/final_model.pt \
        --attacks           neurotoxin a3fl iba chameleon \
        --n-batches         $N_BATCHES \
        --batch-size        $BATCH_SIZE \
        --n-per-class       $N_PER_CLASS \
        --output-dir        experiments/gradient_alignment/outputs_femnist \
        --results-dir       results \
        --device            cuda \
        $DIAG_FLAGS

    echo ""
    echo "Diagnostic outputs:"
    ls -lh experiments/gradient_alignment/outputs_femnist/ 2>/dev/null || echo "  (none)"
else
    echo ""
    echo "Diagnostics skipped (RUN_DIAGNOSTICS=0)."
    echo "Run manually after training:"
    echo ""
    echo "  python -m experiments.gradient_alignment.run_all \\"
    echo "      --config        configs/femnist_fedavg_iba.yaml \\"
    echo "      --benign-checkpoint results/femnist_benign_iid/final_model.pt \\"
    echo "      --attacks       neurotoxin a3fl iba chameleon \\"
    echo "      --n-batches     200 --batch-size 64 --n-per-class 200 \\"
    echo "      --extended-lemma2 \\"
    echo "      --output-dir    experiments/gradient_alignment/outputs_femnist \\"
    echo "      --device        cuda"
fi

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
echo "============================================================"
