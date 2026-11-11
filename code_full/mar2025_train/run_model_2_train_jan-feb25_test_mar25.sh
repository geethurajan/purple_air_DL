#!/bin/bash
# ------------------------------------------------------------------
# Run model_2_train_jan-feb25_test_mar25.py for several radii in parallel.
# Next-month test: train on Jan-Feb 2025, test on March 2025, sensors around site 1103
# (downtown LA).
# log_trans_lin_model, reg_lin_model and the two-stage MLP Model 2, f with t-1 weather
# (hidden layers 16-32-16, plain two-stage training: g on all training hours, then f).
# Every model reports R² and adjusted R² on train and test, on the original scales and with
# reg_lin_model also on the log1p scale.
#
# Results: results_model_2_train_jan-feb25_test_mar25/ (inside mar2025_train)
# After all distances finish, plot them:
#   python3 mar2025_train/plot_model_2_train_jan-feb25_test_mar25.py
#
# mar2025_train sits inside the folder with the training data
# (LA_Site_1103.csv, Sensors_with_coordinates.csv, sensor_*_history.csv for Jan-Feb 2025).
# Submit from either folder:
#   from that folder:            qsub mar2025_train/run_model_2_train_jan-feb25_test_mar25.sh
#   from inside mar2025_train:   qsub run_model_2_train_jan-feb25_test_mar25.sh
#
# One array task = one distance. Works with either scheduler:
#   SGE / UGE (Notre Dame CRC):  qsub run_model_2_train_jan-feb25_test_mar25.sh
#   SLURM:                       sbatch run_model_2_train_jan-feb25_test_mar25.sh
#   No scheduler (one machine):  bash run_model_2_train_jan-feb25_test_mar25.sh   (runs all distances at once)
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
#$ -N pa_model_2_janfeb_mar25
#$ -t 1-8
#$ -pe smp 1
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -q long

# ---- SLURM -------------------------------------------------------
#SBATCH --job-name=pa_model_2_janfeb_mar25
#SBATCH --array=1-8
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pa_model_2_janfeb_mar25_%A_%a.out
#SBATCH --error=logs/pa_model_2_janfeb_mar25_%A_%a.err

DISTANCES=(2.5 3 5 7 9 11 13 15)

EPOCHS=150
HIDDEN=16,32,16                       # hidden layers of g and f (the same as the Jan 2026 test)
SEED=42                               # single MLP run (plain two-stage, no cross-fitting)

NAME=model_2_train_jan-feb25_test_mar25

# Where the job was submitted from: the training data folder, or mar2025_train inside it
if [[ -f mar2025_train/$NAME.py ]]; then
    CODE_DIR=mar2025_train
    DEFAULT_TRAIN_DIR=.
else
    CODE_DIR=.
    DEFAULT_TRAIN_DIR=..
fi

TRAIN_DIR="${TRAIN_DIR:-$DEFAULT_TRAIN_DIR}"   # folder with the Jan-Feb 2025 data and LA_Site_1103.csv
TEST_DIR="${TEST_DIR:-$CODE_DIR}"              # mar2025_train, with the March 2025 sensor files
OUT_DIR="${OUT_DIR:-$CODE_DIR/results_$NAME}"
SCRIPT="${SCRIPT:-$CODE_DIR/$NAME.py}"

if [[ ! -f "$SCRIPT" ]]; then
    echo "Cannot find $SCRIPT. Submit from the training data folder or from mar2025_train." >&2
    exit 1
fi

if [[ ! -f "$TRAIN_DIR/LA_Site_1103.csv" ]]; then
    echo "Cannot find $TRAIN_DIR/LA_Site_1103.csv (EPA data). Check the folder layout." >&2
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
