#!/bin/bash
# ------------------------------------------------------------------
# Run hpc_two_stage_jan2026_different_tries.py: ONE job for one distance (default 2.5 km).
# Step 1 chooses the MLP version on Jan-Feb 2025 validation days (20 versions x 5 seeds),
# step 2 trains the chosen version on all of Jan-Feb 2025 and tests it on January 2026.
#
# jan2026_test sits inside the folder with the training data
# (LA_Site_1103.csv, sensor_*_history.csv, ...). Submit from either folder:
#   from that folder:            qsub jan2026_test/run_two_stage_jan2026_different_tries.sh
#   from inside jan2026_test:    qsub run_two_stage_jan2026_different_tries.sh
#
# Works with either scheduler:
#   SGE / UGE (Notre Dame CRC):  qsub run_two_stage_jan2026_different_tries.sh
#   SLURM:                       sbatch run_two_stage_jan2026_different_tries.sh
#   No scheduler (one machine):  bash run_two_stage_jan2026_different_tries.sh
#
# Before the first run, create the Python environment once:  bash setup_env.sh
# ------------------------------------------------------------------

# ---- SGE / UGE ---------------------------------------------------
#$ -N pa_jan2026_tries
#$ -pe smp 1
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -q long

# ---- SLURM -------------------------------------------------------
#SBATCH --job-name=pa_jan2026_tries
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pa_jan2026_tries_%j.out
#SBATCH --error=logs/pa_jan2026_tries_%j.err

DISTANCE=2.5

# Where the job was submitted from: the training data folder, or jan2026_test inside it
if [[ -f jan2026_test/hpc_two_stage_jan2026_different_tries.py ]]; then
    CODE_DIR=jan2026_test
    DEFAULT_TRAIN_DIR=.
else
    CODE_DIR=.
    DEFAULT_TRAIN_DIR=..
fi

TRAIN_DIR="${TRAIN_DIR:-$DEFAULT_TRAIN_DIR}"   # folder with the Jan-Feb 2025 training data
TEST_DIR="${TEST_DIR:-$CODE_DIR}"              # jan2026_test, with the January 2026 data
OUT_DIR="${OUT_DIR:-$CODE_DIR/results_different_tries}"
SCRIPT="${SCRIPT:-$CODE_DIR/hpc_two_stage_jan2026_different_tries.py}"

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

export OMP_NUM_THREADS="${NSLOTS:-${SLURM_CPUS_PER_TASK:-1}}"

echo "[$(date)] Starting different tries at ${DISTANCE} km on $(hostname)"
python3 "$SCRIPT" \
    --distance "$DISTANCE" \
    --train-dir "$TRAIN_DIR" \
    --test-dir "$TEST_DIR" \
    --out-dir "$OUT_DIR"
echo "[$(date)] Finished (exit $?)"
