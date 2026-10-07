#!/bin/bash
# ------------------------------------------------------------------
# Run final_presentation_distance.py for several radii in parallel.
#
# One array task = one distance. Works with either scheduler:
#   SGE / UGE (Notre Dame CRC):  qsub run_distances.sh
#   SLURM:                       sbatch run_distances.sh
#   No scheduler (one machine):  bash run_distances.sh   (runs all distances at once)
#
# Before the first run, create the Python environment once:  bash setup_env.sh
#
# To change the distances, edit DISTANCES below AND make the array
# range (-t / --array) match the number of entries.
# ------------------------------------------------------------------

# ---- SGE / UGE ---------------------------------------------------
#$ -N pa_distance
#$ -t 1-7
#$ -pe smp 1
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -q long

# ---- SLURM -------------------------------------------------------
#SBATCH --job-name=pa_distance
#SBATCH --array=1-7
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/pa_distance_%A_%a.out
#SBATCH --error=logs/pa_distance_%A_%a.err

DISTANCES=(2.5 3 5 7 9 11 13)

EPOCHS=150
DATA_DIR="${DATA_DIR:-$PWD}"          # folder with the sensor CSVs
OUT_DIR="${OUT_DIR:-results_by_distance}"
SCRIPT="${SCRIPT:-hpc_performance_by_distance.py}"

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
        --data-dir "$DATA_DIR" \
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
        run_one "$d" > "logs/distance_${d}km.log" 2>&1 &
    done
    wait
    echo "All distances finished."
fi
