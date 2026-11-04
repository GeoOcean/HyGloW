#!/bin/bash
#SBATCH --job-name=snap_test2
#SBATCH --partition=geocean
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4gb
#SBATCH --output=/lustre/geocean/WORK/users/javiert/projects/HyGloW/Netherlands/SWAN_test/snapwave/slurm_logs/%A_%a.out
#SBATCH --error=/lustre/geocean/WORK/users/javiert/projects/HyGloW/Netherlands/SWAN_test/snapwave/slurm_logs/%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --array=1-117%24
module use /nfs/software/geocean/modulefiles
module load snapwave

case_dir=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /lustre/geocean/WORK/users/javiert/projects/HyGloW/Netherlands/SWAN_test/snapwave/case_dirs_2.txt)
cd "$case_dir"
OMP_NUM_THREADS=2 snapwave > wrapper_out.log 2> wrapper_error.log
