#!/bin/bash
# ------------------------------------------------------------------
# Run model_2_reno_train_jul-sep21_test_sep22_ensemble.py for several radii in parallel.
# ENSEMBLE version: the MLP is an ensemble of N_SEEDS MLPs (average prediction), and the faulty
# sensors 70519 and 115819 are excluded in the code. Results go to a new folder, so the
# single-seed results stay untouched.
# Smoke-to-smoke test in Reno, Nevada: train on Jul-Sep 2021, test on Sep 2022,
# the same sensors around the Reno4 EPA monitor (site 32-031-0031). Both periods had heavy
# wildfire smoke.
# log_trans_lin_model, reg_lin_model and the two-stage MLP Model 2, f with t-1 weather
# (hidden layers 16-32-16, plain two-stage training: g on all training hours, then f).
# Every model reports R² and adjusted R² on train and test, on the original scales and with
# reg_lin_model also on the log1p scale.
#
# Folder layout (upload the whole reno folder to CRC):
#   reno/
#     model_2_reno_train_jul-sep21_test_sep22_ensemble.py
#     run_model_2_reno_train_jul-sep21_test_sep22_ensemble.sh      (this file)
#     plot_model_2_reno_train_jul-sep21_test_sep22_ensemble_log1p_scale.py
#     train_2021/   Reno4_Site_0031_JulSep2021.csv, Sensors_with_coordinates.csv, sensor_*_history.csv
#     test_2022/    Reno4_Site_0031_Sep2022.csv,    Sensors_with_coordinates.csv, sensor_*_history.csv
#
# Submit from inside the reno folder:
#   cd reno
#   mkdir -p logs
#   qsub run_model_2_reno_train_jul-sep21_test_sep22_ensemble.sh
#
# Results: results_model_2_reno_train_jul-sep21_test_sep22_ensemble/ (inside reno)
# After all distances finish, plot them:
#   python3 plot_model_2_reno_train_jul-sep21_test_sep22_ensemble_log1p_scale.py
#
# One array task = one distance. Works with either scheduler:
#   SGE / UGE (Notre Dame CRC):  qsub run_model_2_reno_train_jul-sep21_test_sep22_ensemble.sh
#   SLURM:                       sbatch run_model_2_reno_train_jul-sep21_test_sep22_ensemble.sh
#   No scheduler (one machine):  bash run_model_2_reno_train_jul-sep21_test_sep22_ensemble.sh   (runs all distances at once)
#
# Before the first run, create the Python environment once:  bash setup_env.sh
# (it needs matplotlib for the graphs)
#
# The logs/ folder must exist in the folder you submit from BEFORE qsub
# (Grid Engine opens the log files before this script runs):  mkdir -p logs
#
# To change the distances, edit DISTANCES below AND make the array
# range (-t / --array) match the number of entries.
# ------------------------------------------------------------------

# ---- SGE / UGE ---------------------------------------------------
#$ -N pa_model_2_reno_ens
#$ -t 1-9
#$ -pe smp 1
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -q long

# ---- SLURM -------------------------------------------------------
#SBATCH --job-name=pa_model_2_reno_ens
#SBATCH --array=1-9
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=logs/pa_model_2_reno_ens_%A_%a.out
#SBATCH --error=logs/pa_model_2_reno_ens_%A_%a.err

# 9 distances: the closest sensor is 1.62 km from Reno4; 2.5 km has the same 3 sensors as 3 km
DISTANCES=(2 3 4 5 6 7 8 9 10)

EPOCHS=150
HIDDEN=16,32,16                       # hidden layers of g and f (the same as the LA tests)
SEED=42                               # first MLP seed: the ensemble uses 42, 43, 44, 45, 46
N_SEEDS=5                             # MLPs in the ensemble (plain two-stage, no cross-fitting)

NAME=model_2_reno_train_jul-sep21_test_sep22_ensemble

TRAIN_DIR="${TRAIN_DIR:-train_2021}"   # Jul-Sep 2021 training data
TEST_DIR="${TEST_DIR:-test_2022}"      # Sep 2022 test data
OUT_DIR="${OUT_DIR:-results_$NAME}"
SCRIPT="${SCRIPT:-$NAME.py}"

if [[ ! -f "$SCRIPT" ]]; then
    echo "Cannot find $SCRIPT. Submit from inside the reno folder." >&2
    exit 1
fi

if [[ ! -f "$TRAIN_DIR/Reno4_Site_0031_JulSep2021.csv" ]]; then
    echo "Cannot find $TRAIN_DIR/Reno4_Site_0031_JulSep2021.csv (Jul-Sep 2021 EPA data)." >&2
    exit 1
fi

if [[ ! -f "$TEST_DIR/Reno4_Site_0031_Sep2022.csv" ]]; then
    echo "Cannot find $TEST_DIR/Reno4_Site_0031_Sep2022.csv (Sep 2022 EPA data)." >&2
    exit 1
fi

mkdir -p logs "$OUT_DIR"

# ---- Python environment ------------------------------------------
# One-time setup (run once on a login node, NOT inside this job):
#   bash setup_env.sh
VENV="${VENV:-$HOME/purpleair_env}"

# Make the "module" command available in batch shells if it isn't already
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
    # Scheduler array job: run only this task's distance
    export OMP_NUM_THREADS="${NSLOTS:-${SLURM_CPUS_PER_TASK:-1}}"
    run_one "${DISTANCES[$((TASK_ID - 1))]}"
else
    # No scheduler: run every distance in the background on this machine
    for d in "${DISTANCES[@]}"; do
        run_one "$d" > "logs/${NAME}_${d}km.log" 2>&1 &
    done
    wait
    echo "All distances finished."
fi
