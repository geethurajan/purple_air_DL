#!/bin/bash
# ------------------------------------------------------------------
# Run hpc_two_stage_jan2026_test_janmaraverage.py for several radii in parallel.
# Later-year test: train on Jan-Mar 2025, test on Jan 2026,
# same sensors around site 1103 (downtown LA).
# log_trans_lin_model, reg_lin_model and the two-stage MLP (hidden layers 16-32-16),
# as an ensemble: 6 runs (seeds) per distance, predictions averaged.
#
# jan2026_test and mar2025_train sit inside the folder with the training data
# (LA_Site_1103.csv, sensor_*_history.csv, ...). Submit from either folder:
#   from that folder:            qsub jan2026_test/run_two_stage_jan2026_test_janmaraverage.sh
#   from inside jan2026_test:    qsub run_two_stage_jan2026_test_janmaraverage.sh
#
# One array task = one distance. Works with either scheduler:
#   SGE / UGE (Notre Dame CRC):  qsub run_two_stage_jan2026_test_janmaraverage.sh
#   SLURM:                       sbatch run_two_stage_jan2026_test_janmaraverage.sh
#   No scheduler (one machine):  bash run_two_stage_jan2026_test_janmaraverage.sh   (runs all distances at once)
#
# Before the first run, create the Python environment once:  bash setup_env.sh
#
# To change the distances, edit DISTANCES below AND make the array
# range (-t / --array) match the number of entries.
# ------------------------------------------------------------------

# ---- SGE / UGE ---------------------------------------------------
#$ -N pa_jan2026_janmaravg
#$ -t 1-8
#$ -pe smp 1
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -q long

# ---- SLURM -------------------------------------------------------
#SBATCH --job-name=pa_jan2026_janmaravg
#SBATCH --array=1-8
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pa_jan2026_janmaravg_%A_%a.out
#SBATCH --error=logs/pa_jan2026_janmaravg_%A_%a.err

DISTANCES=(2.5 3 5 7 9 11 13 15)

EPOCHS=150
HIDDEN=16,32,16                       # hidden layers of g and f (chosen on the 2025 validation days)
SEEDS=42,1,2,3,4,5                    # MLP runs whose predictions are averaged (the ensemble)

# Where the job was submitted from: the training data folder, or jan2026_test inside it
if [[ -f jan2026_test/hpc_two_stage_jan2026_test_janmaraverage.py ]]; then
    CODE_DIR=jan2026_test
    DEFAULT_TRAIN_DIR=.
    DEFAULT_TRAIN_EXTRA_DIR=mar2025_train
else
    CODE_DIR=.
    DEFAULT_TRAIN_DIR=..
    DEFAULT_TRAIN_EXTRA_DIR=../mar2025_train
fi

TRAIN_DIR="${TRAIN_DIR:-$DEFAULT_TRAIN_DIR}"                   # Jan-Feb 2025 training data
TRAIN_EXTRA_DIR="${TRAIN_EXTRA_DIR:-$DEFAULT_TRAIN_EXTRA_DIR}" # March 2025 training data
TEST_DIR="${TEST_DIR:-$CODE_DIR}"                              # jan2026_test, with the January 2026 data
OUT_DIR="${OUT_DIR:-$CODE_DIR/results_jan2026_test_janmaraverage}"
SCRIPT="${SCRIPT:-$CODE_DIR/hpc_two_stage_jan2026_test_janmaraverage.py}"

if [[ ! -f "$SCRIPT" ]]; then
    echo "Cannot find $SCRIPT. Submit from the training data folder or from jan2026_test." >&2
    exit 1
fi

if [[ ! -d "$TRAIN_EXTRA_DIR" ]]; then
    echo "Cannot find the March 2025 folder $TRAIN_EXTRA_DIR. Upload mar2025_train next to jan2026_test." >&2
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
        --seeds "$SEEDS" \
        --train-dir "$TRAIN_DIR" \
        --train-extra-dir "$TRAIN_EXTRA_DIR" \
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
        run_one "$d" > "logs/jan2026_janmaravg_${d}km.log" 2>&1 &
    done
    wait
    echo "All distances finished."
fi
