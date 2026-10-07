#!/bin/bash
#$ -M grajan@nd.edu          # email for job notifications
#$ -m abe                    # email when job aborts, begins, ends
#$ -q long                   # queue (long = up to ~14 days on ND CRC)
#$ -pe smp 8                 # number of CPU cores
#$ -N purpleair_distance       # job name
#$ -cwd                      # run from the folder you submit from
#$ -o purpleair_distance.out   # standard output log
#$ -e purpleair_distance.err   # error log

# For a GPU instead, replace the "-q long" and "-pe smp 8" lines with:
#   #$ -q gpu
#   #$ -l gpu_card=1

module load python/3.12      # use a name shown by: module avail python
source ~/purpleair_env/bin/activate

# let PyTorch use the cores requested above
export OMP_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS

# don't try to open plot windows on the cluster
export MPLBACKEND=Agg

echo "Job started on $(hostname) at $(date)"
python distance_experiments.py
echo "Job finished at $(date)"
