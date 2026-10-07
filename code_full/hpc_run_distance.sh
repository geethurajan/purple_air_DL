#!/bin/bash
# ------------------------------------------------------------------
# Run final_presentation_distance.py for several radii in parallel.
#
# One array task = one distance. Works with either scheduler:
#   SGE / UGE (Notre Dame CRC):  qsub run_distances.sh
#   SLURM:                       sbatch run_distances.sh
#   No scheduler (one machine):  bash run_distances.sh   (runs all distances at once)
#
# To change the distances, edit DISTANCES below AND make the array
# range (-t / --array) match the number of entries.
# ------------------------------------------------------------------

# ---- SGE / UGE ---------------------------------------------------
#$ -N pa_distance
#$ -t 1-7
#$ -pe smp 4
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -q long

# ---- SLURM -------------------------------------------------------
#SBATCH --job-name=pa_distance
#SBATCH --array=1-7
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pa_distance_%A_%a.out
#SBATCH --error=logs/pa_distance_%A_%a.err

DISTANCES=(3 5 10 15 20 30 50)        # 50 km covers every sensor

EPOCHS=150
DATA_DIR="${DATA_DIR:-$PWD}"          # folder with the sensor CSVs
OUT_DIR="${OUT_DIR:-results_by_distance}"
SCRIPT="${SCRIPT:-final_presentation_distance.py}"

mkdir -p logs "$OUT_DIR"

# ---- Python environment (edit for your cluster) ------------------
# module load python
# conda activate purpleair

run_one () {
    local d=$1
    echo "[$(date)] Starting distance ${d} km on $(hostname)"
    python "$SCRIPT" \
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
