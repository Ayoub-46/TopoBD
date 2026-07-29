#!/bin/bash

# ==========================================
# SLURM DIRECTIVES
# ==========================================
#SBATCH --partition=gpu-a6000
#SBATCH --time=12:00:00             # bottleneck ≈6 h (GPU 0); 6 h margin
#SBATCH --job-name=cifar10_vgg_prereqs
#SBATCH --error=logs/cifar10_vgg_prereq_%j.err
#SBATCH --output=logs/cifar10_vgg_prereq_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32          # 8 per GPU worker
#SBATCH --mem=128GB
#SBATCH --gres=gpu:4
#
# Full-scale CIFAR-10 / VGG-13-noBN gradient-alignment prerequisites.
# Trains benign + 4 attacks (200 rounds / 100 clients) from the vgg13_nobn
# configs into results/vgg13_nobn/, then runs the gradient-alignment
# diagnostics against them.  Distinct from the resnet18 prereqs (results/).

# ==========================================
# CONFIGURATION
# ==========================================
PYTHON_VERSION=3.10
ENVIRONMENT_NAME="toposentinel"
TORCH_CUDA="${TORCH_CUDA:-cu124}"   # override via --export if your cluster uses a different CUDA

VGG_RESULTS="results/vgg13_nobn"    # matches output_dir in the vgg13_nobn configs

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

mkdir -p logs "$VGG_RESULTS" experiments/gradient_alignment/outputs/outputs_cifar10_vgg13_nobn

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
    echo "Warning: 'data' folder not found — CIFAR-10 will be downloaded on first run."
fi

# ==========================================
# PARALLEL TRAINING — 4 GPUS (run_experiment.py on the vgg13_nobn configs)
# ==========================================
# Checkpoints land in results/vgg13_nobn/<name>/final_model.pt (config output_dir).
#   GPU 0 — benign_iid + neurotoxin   (lightest pair; bottleneck ≈6 h)
#   GPU 1 — a3fl
#   GPU 2 — iba
#   GPU 3 — chameleon

train() {  # $1 = CUDA device, $2.. = config basenames (without .yaml)
    local dev="$1"; shift
    export CUDA_VISIBLE_DEVICES="$dev"
    for cfg in "$@"; do
        echo "[GPU ${dev}] training ${cfg}"
        python run_experiment.py --config "configs/${cfg}.yaml" --device cuda
    done
    echo "[GPU ${dev}] Done"
}

echo ""
echo "Starting parallel VGG-13-noBN training on 4 GPUs..."
echo "$(date '+%Y-%m-%d %H:%M:%S')"

train 0 cifar10_benign_iid_vgg13_nobn cifar10_fedavg_neurotoxin_vgg13_nobn \
    > logs/cifar10_vgg_prereq_gpu0_${SLURM_JOB_ID}.log 2>&1 &
PID0=$!
train 1 cifar10_fedavg_a3fl_vgg13_nobn \
    > logs/cifar10_vgg_prereq_gpu1_${SLURM_JOB_ID}.log 2>&1 &
PID1=$!
train 2 cifar10_fedavg_iba_vgg13_nobn \
    > logs/cifar10_vgg_prereq_gpu2_${SLURM_JOB_ID}.log 2>&1 &
PID2=$!
train 3 cifar10_fedavg_chameleon_vgg13_nobn \
    > logs/cifar10_vgg_prereq_gpu3_${SLURM_JOB_ID}.log 2>&1 &
PID3=$!

wait $PID0 $PID1 $PID2 $PID3

echo ""
echo "All training workers finished. $(date '+%Y-%m-%d %H:%M:%S')"
for gpu in 0 1 2 3; do
    logfile="logs/cifar10_vgg_prereq_gpu${gpu}_${SLURM_JOB_ID}.log"
    [ -f "$logfile" ] && { echo ""; echo "--- GPU ${gpu} last 5 lines ---"; tail -5 "$logfile"; }
done

# ==========================================
# GRADIENT-ALIGNMENT DIAGNOSTICS
# ==========================================
if [ "$RUN_DIAGNOSTICS" -eq 1 ]; then
    EXTRA=""
    [ "$EXTENDED_LEMMA2" -eq 1 ] && EXTRA="--extended-lemma2"
    echo ""
    echo "Running gradient-alignment diagnostics (vgg13_nobn)..."
    python -m experiments.gradient_alignment.run_all \
        --config          configs/cifar10_fedavg_iba_vgg13_nobn.yaml \
        --benign-checkpoint "$VGG_RESULTS/cifar10_benign_iid/final_model.pt" \
        --attacks         neurotoxin a3fl iba chameleon \
        --results-dir     "$VGG_RESULTS" \
        --n-batches       "$N_BATCHES" \
        --batch-size      "$BATCH_SIZE" \
        --n-per-class     "$N_PER_CLASS" \
        $EXTRA \
        --output-dir      experiments/gradient_alignment/outputs/outputs_cifar10_vgg13_nobn \
        --device          cuda
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
echo " Checkpoints : $VGG_RESULTS/cifar10_*/final_model.pt"
echo " Diagnostics : experiments/gradient_alignment/outputs/outputs_cifar10_vgg13_nobn"
echo "============================================================"
