#!/bin/bash
# ------------------------------------------------------------------
# Run model_2_t_lag_ensemble_reno.py for several radii in parallel.
# Model 2 (hour t + hour t-1): reg_lin_model, log_trans_lin_model, MLP without g and MLP with g
# (two-stage), both MLPs as ensembles of 5 seeds,
# all scored on log1p PM2.5. Train: Jul-Sep 2021, test: Sep 2022, Reno4 (site 32-031-0031, Reno, Nevada).
#
# Results: comparison_reno/results_model_2_t_lag_ensemble_reno/
# After all distances finish, plot them:
#   python3 comparison_reno/plot_model_2_t_lag_ensemble_reno_same_scale.py
#
# comparison_reno sits inside the reno folder, next to train_2021 and test_2022.
# Submit from either folder:
#   from reno:                     qsub comparison_reno/run_model_2_t_lag_ensemble_reno.sh
#   from inside comparison_reno:   qsub run_model_2_t_lag_ensemble_reno.sh
#
# One array task = one distance:
#   SGE / UGE (Notre Dame CRC):  qsub run_model_2_t_lag_ensemble_reno.sh
#   SLURM:                       sbatch run_model_2_t_lag_ensemble_reno.sh
#   No scheduler (one machine):  bash run_model_2_t_lag_ensemble_reno.sh   (all distances at once)
#
# The logs/ folder must exist in the folder you submit from BEFORE qsub:  mkdir -p logs
# To change the distances, edit DISTANCES AND make the array range (-t / --array) match.
# ------------------------------------------------------------------

# ---- SGE / UGE ---------------------------------------------------
#$ -N pa_model_2_t_lag_reno
#$ -t 1-9
#$ -pe smp 1
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -q long

# ---- SLURM -------------------------------------------------------
#SBATCH --job-name=pa_model_2_t_lag_reno
#SBATCH --array=1-9
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=logs/pa_model_2_t_lag_reno_%A_%a.out
#SBATCH --error=logs/pa_model_2_t_lag_reno_%A_%a.err

# 9 distances: every Reno sensor is within 9.91 km of Reno4 (closest 1.62 km)
DISTANCES=(2 3 4 5 6 7 8 9 10)

EPOCHS=150
HIDDEN=16,32,16
SEED=42          # first seed of the ensemble
N_SEEDS=5        # seeds 42, 43, 44, 45, 46

NAME=model_2_t_lag_ensemble_reno

# Where the job was submitted from: reno, or comparison_reno inside it
if [[ -f comparison_reno/$NAME.py ]]; then
    CODE_DIR=comparison_reno
    DATA_DIR=.
else
    CODE_DIR=.
    DATA_DIR=..
fi

TRAIN_DIR="${TRAIN_DIR:-$DATA_DIR/train_2021}"   # July-September 2021 training data
TEST_DIR="${TEST_DIR:-$DATA_DIR/test_2022}"      # September 2022 test data
OUT_DIR="${OUT_DIR:-$CODE_DIR/results_$NAME}"
SCRIPT="${SCRIPT:-$CODE_DIR/$NAME.py}"

for folder in "$TRAIN_DIR" "$TEST_DIR"; do
    if [[ ! -d "$folder" ]]; then
        echo "Cannot find $folder. comparison_reno must sit inside reno, next to train_2021 and test_2022." >&2
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
