#!/bin/bash
# ------------------------------------------------------------------
# Run model_2_reno_train_jul-oct21_test_sep22.py for several radii in parallel.
# Reno, Nevada: train on Jul-OCT 2021 (one more month than the Jul-Sep version), test on
# Sep 2022, the same 22 sensors around the Reno4 EPA monitor (site 32-031-0031).
# log_trans_lin_model, reg_lin_model and the two-stage MLP Model 2, f with t-1 weather
# (hidden layers 16-32-16, 150 epochs, ONE MLP with seed 42, no ensemble).
#
# Folder layout (upload the whole reno folder to CRC):
#   reno/
#     train_2021/                  Jul-Sep 2021: Reno4_Site_0031_JulSep2021.csv, sensors
#       oct2021/                   THIS folder: October 2021 data and this code
#         Reno4_Site_0031_Oct2021.csv, Sensors_with_coordinates.csv, sensor_*_history.csv
#         model_2_reno_train_jul-oct21_test_sep22.py
#         run_model_2_reno_train_jul-oct21_test_sep22.sh        (this file)
#         plot_model_2_reno_train_jul-oct21_test_sep22_log1p_scale.py
#     test_2022/                   Sep 2022: Reno4_Site_0031_Sep2022.csv, sensors
#
# Submit from inside the oct2021 folder:
#   cd reno/train_2021/oct2021
#   mkdir -p logs
#   qsub run_model_2_reno_train_jul-oct21_test_sep22.sh
#
# Results: results_model_2_reno_train_jul-oct21_test_sep22/ (inside oct2021)
# After all distances finish, plot them:
#   python3 plot_model_2_reno_train_jul-oct21_test_sep22_log1p_scale.py
#
# One array task = one distance. Works with either scheduler:
#   SGE / UGE (Notre Dame CRC):  qsub run_model_2_reno_train_jul-oct21_test_sep22.sh
#   SLURM:                       sbatch run_model_2_reno_train_jul-oct21_test_sep22.sh
#   No scheduler (one machine):  bash run_model_2_reno_train_jul-oct21_test_sep22.sh   (runs all distances at once)
#
# The logs/ folder must exist in the folder you submit from BEFORE qsub
# (Grid Engine opens the log files before this script runs):  mkdir -p logs
#
# To change the distances, edit DISTANCES below AND make the array
# range (-t / --array) match the number of entries.
# ------------------------------------------------------------------

# ---- SGE / UGE ---------------------------------------------------
#$ -N pa_model_2_reno_juloct
#$ -t 1-9
#$ -pe smp 1
#$ -cwd
#$ -o logs/
#$ -e logs/
#$ -q long

# ---- SLURM -------------------------------------------------------
#SBATCH --job-name=pa_model_2_reno_juloct
#SBATCH --array=1-9
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pa_model_2_reno_juloct_%A_%a.out
#SBATCH --error=logs/pa_model_2_reno_juloct_%A_%a.err

# 9 distances (the same as the Jul-Sep version)
DISTANCES=(2 3 4 5 6 7 8 9 10)

EPOCHS=150
HIDDEN=16,32,16                       # hidden layers of g and f
SEED=42                               # ONE MLP (no ensemble), plain two-stage training

NAME=model_2_reno_train_jul-oct21_test_sep22

TRAIN_DIR="${TRAIN_DIR:-..}"                      # train_2021: Jul-Sep 2021
TRAIN_EXTRA_DIR="${TRAIN_EXTRA_DIR:-.}"           # oct2021 (this folder): October 2021
TEST_DIR="${TEST_DIR:-../../test_2022}"           # Sep 2022
OUT_DIR="${OUT_DIR:-results_$NAME}"
SCRIPT="${SCRIPT:-$NAME.py}"

if [[ ! -f "$SCRIPT" ]]; then
    echo "Cannot find $SCRIPT. Submit from inside reno/train_2021/oct2021." >&2
    exit 1
fi

for f in "$TRAIN_DIR/Reno4_Site_0031_JulSep2021.csv" "$TRAIN_EXTRA_DIR/Reno4_Site_0031_Oct2021.csv" \
         "$TEST_DIR/Reno4_Site_0031_Sep2022.csv"; do
    if [[ ! -f "$f" ]]; then
        echo "Cannot find $f. Check the folder layout above." >&2
        exit 1
    fi
done

mkdir -p logs "$OUT_DIR"

# ---- Python environment ------------------------------------------
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
        run_one "$d" > "logs/${NAME}_${d}km.log" 2>&1 &
    done
    wait
    echo "All distances finished."
fi
