#!/bin/bash
# ------------------------------------------------------------------
# Run model_2_t_lag_ensemble.py for several radii in parallel.
# Model 2 (hour t + hour t-1): reg_lin_model, log_trans_lin_model, MLP without g and MLP with g
# (two-stage), both MLPs as ensembles of 5 seeds,
# all scored on log1p PM2.5. Train: Jan-Mar 2025, test: Jan 2026, site 1103 (downtown LA).
#
# Results: comparison/results_model_2_t_lag_ensemble/
# After all distances finish, plot them:
#   python3 comparison/plot_model_2_t_lag_ensemble_same_scale.py
#
# comparison sits inside the training data folder (code_full), next to jan2026_test and
# mar2025_train. Submit from either folder:
#   from code_full:             qsub comparison/run_model_2_t_lag_ensemble.sh
#   from inside comparison:   qsub run_model_2_t_lag_ensemble.sh
#
# One array task = one distance:
#   SGE / UGE (Notre Dame CRC):  qsub run_model_2_t_lag_ensemble.sh
#   SLURM:                       sbatch run_model_2_t_lag_ensemble.sh
#   No scheduler (one machine):  bash run_model_2_t_lag_ensemble.sh   (all distances at once)
#
# The logs/ folder must exist in the folder you submit from BEFORE qsub:  mkdir -p logs
# To change the distances, edit DISTANCES AND make the array range (-t / --array) match.
# ------------------------------------------------------------------

# ---- SGE / UGE ---------------------------------------------------
#$ -N pa_model_2_t_lag
#$ -t 1-8
#$ -pe smp 1
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -q long

# ---- SLURM -------------------------------------------------------
#SBATCH --job-name=pa_model_2_t_lag
#SBATCH --array=1-8
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=logs/pa_model_2_t_lag_%A_%a.out
#SBATCH --error=logs/pa_model_2_t_lag_%A_%a.err

DISTANCES=(2.5 3.5 5 7 9 11 13 15)

EPOCHS=150
HIDDEN=16,32,16
SEED=42          # first seed of the ensemble
N_SEEDS=5        # seeds 42, 43, 44, 45, 46

NAME=model_2_t_lag_ensemble

# Where the job was submitted from: code_full, or comparison inside it
if [[ -f comparison/$NAME.py ]]; then
    CODE_DIR=comparison
    DATA_DIR=.
else
    CODE_DIR=.
    DATA_DIR=..
fi

TRAIN_DIR="${TRAIN_DIR:-$DATA_DIR}"                              # Jan-Feb 2025 training data
TRAIN_EXTRA_DIR="${TRAIN_EXTRA_DIR:-$DATA_DIR/mar2025_train}"    # March 2025 training data
TEST_DIR="${TEST_DIR:-$DATA_DIR/jan2026_test}"                   # January 2026 test data
OUT_DIR="${OUT_DIR:-$CODE_DIR/results_$NAME}"
SCRIPT="${SCRIPT:-$CODE_DIR/$NAME.py}"

for folder in "$TRAIN_EXTRA_DIR" "$TEST_DIR"; do
    if [[ ! -d "$folder" ]]; then
        echo "Cannot find $folder. comparison must sit next to mar2025_train and jan2026_test." >&2
        exit 1
    fi
done

mkdir -p logs "$OUT_DIR"

# ---- Python environment ------------------------------------------
VENV="${VENV:-$HOME/purpleair_env}"

if ! command -v module > /dev/null 2>&1; then
    [[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh
fi

module load python/3.12

if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "Virtual environment not found at $VENV. Run 'bash setup_env.sh' first." >&2
    exit 1
fi

source "$VENV/bin/activate"
echo "Using $(which python3) ($(python3 --version))"

run_one () {
    local d=$1
    echo "[$(date)] Starting distance ${d} km on $(hostname)"
    python3 "$SCRIPT" \
        --distance "$d" \
        --epochs "$EPOCHS" \
        --hidden "$HIDDEN" \
        --seed "$SEED" \
        --n-seeds "$N_SEEDS" \
        --train-dir "$TRAIN_DIR" \
        --train-extra-dir "$TRAIN_EXTRA_DIR" \
        --test-dir "$TEST_DIR" \
        --out-dir "$OUT_DIR"
    echo "[$(date)] Finished distance ${d} km (exit $?)"
}

TASK_ID="${SGE_TASK_ID:-${SLURM_ARRAY_TASK_ID:-}}"

if [[ -n "$TASK_ID" && "$TASK_ID" != "undefined" ]]; then
    export OMP_NUM_THREADS="${NSLOTS:-${SLURM_CPUS_PER_TASK:-1}}"
    run_one "${DISTANCES[$((TASK_ID - 1))]}"
else
    for d in "${DISTANCES[@]}"; do
        run_one "$d" > "logs/${NAME}_${d}km.log" 2>&1 &
    done
    wait
    echo "All distances finished."
fi
