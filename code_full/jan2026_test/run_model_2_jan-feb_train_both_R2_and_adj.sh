#!/bin/bash
# ------------------------------------------------------------------
# Run model_2_jan-feb_train_both_R2_and_adj.py for several radii in parallel.
# Later-year test: train on Jan-Feb 2025, test on Jan 2026,
# same sensors around site 1103 (downtown LA).
# log_trans_lin_model, reg_lin_model and the two-stage MLP Model 2, f with t-1 weather
# (hidden layers 16-32-16, plain two-stage training: g on all training hours, then f).
# Every model reports R² and adjusted R² on train and test.
#
# Results: results_model_2_jan-feb_train_both_R2_and_adj/ (inside jan2026_test)
# After all distances finish, plot them:
#   python3 jan2026_test/plot_model_2_jan-feb_train_both_R2_and_adj.py
#
# jan2026_test sits inside the folder with the training data
# (LA_Site_1103.csv, sensor_*_history.csv, ...). Submit from either folder:
#   from that folder:            qsub jan2026_test/run_model_2_jan-feb_train_both_R2_and_adj.sh
#   from inside jan2026_test:    qsub run_model_2_jan-feb_train_both_R2_and_adj.sh
#
# One array task = one distance. Works with either scheduler:
#   SGE / UGE (Notre Dame CRC):  qsub run_model_2_jan-feb_train_both_R2_and_adj.sh
#   SLURM:                       sbatch run_model_2_jan-feb_train_both_R2_and_adj.sh
#   No scheduler (one machine):  bash run_model_2_jan-feb_train_both_R2_and_adj.sh   (runs all distances at once)
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
#$ -N pa_model_2_janfeb_R2_adj
#$ -t 1-8
#$ -pe smp 1
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -q long

# ---- SLURM -------------------------------------------------------
#SBATCH --job-name=pa_model_2_janfeb_R2_adj
#SBATCH --array=1-8
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pa_model_2_janfeb_R2_adj_%A_%a.out
#SBATCH --error=logs/pa_model_2_janfeb_R2_adj_%A_%a.err

DISTANCES=(2.5 3 5 7 9 11 13 15)

EPOCHS=150
HIDDEN=16,32,16                       # hidden layers of g and f (chosen on the 2025 validation days)
CROSSFIT_FOLDS=1                      # 1 = plain two-stage (g on all training hours, then f)

NAME=model_2_jan-feb_train_both_R2_and_adj

# Where the job was submitted from: the training data folder, or jan2026_test inside it
if [[ -f jan2026_test/$NAME.py ]]; then
    CODE_DIR=jan2026_test
    DEFAULT_TRAIN_DIR=.
else
    CODE_DIR=.
    DEFAULT_TRAIN_DIR=..
fi

TRAIN_DIR="${TRAIN_DIR:-$DEFAULT_TRAIN_DIR}"   # folder with the Jan-Feb 2025 training data
TEST_DIR="${TEST_DIR:-$CODE_DIR}"              # jan2026_test, with the January 2026 data
OUT_DIR="${OUT_DIR:-$CODE_DIR/results_$NAME}"
SCRIPT="${SCRIPT:-$CODE_DIR/$NAME.py}"

if [[ ! -f "$SCRIPT" ]]; then
    echo "Cannot find $SCRIPT. Submit from the training data folder or from jan2026_test." >&2
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
        --crossfit-folds "$CROSSFIT_FOLDS" \
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
