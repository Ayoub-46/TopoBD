#!/bin/bash

# ==========================================
# SLURM DIRECTIVES
# ==========================================
#SBATCH --partition=gpu-a6000
#SBATCH --time=12:00:00             # bottleneck ≈6 h (GPU 0); 6 h margin
#SBATCH --job-name=cifar10_prereqs
#SBATCH --error=logs/cifar10_prereq_%j.err
#SBATCH --output=logs/cifar10_prereq_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32          # 8 per GPU worker
#SBATCH --mem=128GB
#SBATCH --gres=gpu:4

# ==========================================
# CONFIGURATION
# ==========================================
PYTHON_VERSION=3.10
ENVIRONMENT_NAME="toposentinel"
TORCH_CUDA="${TORCH_CUDA:-cu124}"   # override via --export if your cluster uses a different CUDA

FORCE_RETRAIN="${FORCE_RETRAIN:-0}"      # 1 = retrain even if checkpoint exists

# Set to 1 to run gradient-alignment diagnostics immediately after training
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"
N_BATCHES="${N_BATCHES:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"
N_PER_CLASS="${N_PER_CLASS:-200}"
EXTENDED_LEMMA2="${EXTENDED_LEMMA2:-1}"  # 1 = also run the rank-comparison experiment

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

mkdir -p logs results experiments/gradient_alignment/outputs_cifar10

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
    echo "Warning: 'data' folder not found — CIFAR-10 will be downloaded on first run."
fi

# ==========================================
# BUILD COMMON TRAINING ARGUMENTS
# ==========================================
TRAIN_BASE="experiments/gradient_alignment/train_prerequisites.py --dataset cifar10"
[ "$FORCE_RETRAIN" -eq 1 ] && TRAIN_BASE="$TRAIN_BASE --force"

# ==========================================
# PARALLEL TRAINING — 4 GPUS
# ==========================================
# GPU assignment (lightest pair on GPU 0 so all finish at roughly the same time):
#   GPU 0 — benign_iid + fedavg_neurotoxin   ≈3 h + ≈3 h = ≈6 h  (bottleneck)
#   GPU 1 — fedavg_a3fl                       ≈4 h
#   GPU 2 — fedavg_iba                        ≈4 h
#   GPU 3 — fedavg_chameleon                  ≈4 h

echo ""
echo "Starting parallel training on 4 GPUs..."
echo "$(date '+%Y-%m-%d %H:%M:%S')"

# --- GPU 0: benign baseline then neurotoxin ---
(
    echo "[GPU 0] Starting cifar10_benign_iid then cifar10_fedavg_neurotoxin"
    export CUDA_VISIBLE_DEVICES=0
    python $TRAIN_BASE \
        --runs cifar10_benign_iid cifar10_fedavg_neurotoxin \
        > logs/cifar10_prereq_gpu0_${SLURM_JOB_ID}.log 2>&1
    echo "[GPU 0] Done"
) &
PID0=$!

# --- GPU 1: A3FL ---
(
    echo "[GPU 1] Starting cifar10_fedavg_a3fl"
    export CUDA_VISIBLE_DEVICES=1
    python $TRAIN_BASE \
        --runs cifar10_fedavg_a3fl \
        > logs/cifar10_prereq_gpu1_${SLURM_JOB_ID}.log 2>&1
    echo "[GPU 1] Done"
) &
PID1=$!

# --- GPU 2: IBA ---
(
    echo "[GPU 2] Starting cifar10_fedavg_iba"
    export CUDA_VISIBLE_DEVICES=2
    python $TRAIN_BASE \
        --runs cifar10_fedavg_iba \
        > logs/cifar10_prereq_gpu2_${SLURM_JOB_ID}.log 2>&1
    echo "[GPU 2] Done"
) &
PID2=$!

# --- GPU 3: Chameleon ---
(
    echo "[GPU 3] Starting cifar10_fedavg_chameleon"
    export CUDA_VISIBLE_DEVICES=3
    python $TRAIN_BASE \
        --runs cifar10_fedavg_chameleon \
        > logs/cifar10_prereq_gpu3_${SLURM_JOB_ID}.log 2>&1
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
    logfile="logs/cifar10_prereq_gpu${gpu}_${SLURM_JOB_ID}.log"
    if [ -f "$logfile" ]; then
        echo ""
        echo "--- GPU ${gpu} last 5 lines ---"
        tail -5 "$logfile"
    fi
done

# Report checkpoint status
echo ""
echo "Checkpoint status:"
for run in cifar10_benign_iid cifar10_fedavg_neurotoxin \
           cifar10_fedavg_a3fl cifar10_fedavg_iba cifar10_fedavg_chameleon; do
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
        --config            configs/cifar10_fedavg_iba.yaml \
        --benign-checkpoint results/cifar10_benign_iid/final_model.pt \
        --attacks           neurotoxin a3fl iba chameleon \
        --n-batches         $N_BATCHES \
        --batch-size        $BATCH_SIZE \
        --n-per-class       $N_PER_CLASS \
        --output-dir        experiments/gradient_alignment/outputs_cifar10 \
        --results-dir       results \
        --device            cuda \
        $DIAG_FLAGS

    echo ""
    echo "Diagnostic outputs:"
    ls -lh experiments/gradient_alignment/outputs_cifar10/ 2>/dev/null || echo "  (none)"
else
    echo ""
    echo "Diagnostics skipped (RUN_DIAGNOSTICS=0)."
    echo "Run manually after training:"
    echo ""
    echo "  python -m experiments.gradient_alignment.run_all \\"
    echo "      --config        configs/cifar10_fedavg_iba.yaml \\"
    echo "      --benign-checkpoint results/cifar10_benign_iid/final_model.pt \\"
    echo "      --attacks       neurotoxin a3fl iba chameleon \\"
    echo "      --n-batches     200 --batch-size 64 --n-per-class 200 \\"
    echo "      --extended-lemma2 \\"
    echo "      --output-dir    experiments/gradient_alignment/outputs_cifar10 \\"
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
