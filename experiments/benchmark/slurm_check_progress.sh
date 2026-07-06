#!/bin/bash

# ==========================================
# SLURM DIRECTIVES
# ==========================================
#SBATCH --partition=cpu
#SBATCH --time=00:10:00
#SBATCH --job-name=check_progress
#SBATCH --error=logs/check_progress_%j.err
#SBATCH --output=logs/check_progress_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB

# ==========================================
# CONFIGURATION
# ==========================================
ENVIRONMENT_NAME="toposentinel"
RESULTS_DIR="${RESULTS_DIR:-results}"
SUITE="${SUITE:-all}"              # all | benchmark | toposentinel | ablation
VERBOSE="${VERBOSE:-0}"            # 1 to show per-dataset seed grids
MISSING_ONLY="${MISSING_ONLY:-0}"  # 1 to list only incomplete run names

# ==========================================
# ENVIRONMENT SETUP
# ==========================================
module load Anaconda3
source activate ${ENVIRONMENT_NAME}

mkdir -p logs

# ==========================================
# BUILD ARGUMENTS
# ==========================================
ARGS="--results-dir $RESULTS_DIR --suite $SUITE"
[ "$VERBOSE"      -eq 1 ] && ARGS="$ARGS --verbose"
[ "$MISSING_ONLY" -eq 1 ] && ARGS="$ARGS --missing-only"

# ==========================================
# RUN
# ==========================================
echo "Progress report — $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results dir : $RESULTS_DIR"
echo "Suite       : $SUITE"
echo ""

python experiments/benchmark/check_progress.py $ARGS
