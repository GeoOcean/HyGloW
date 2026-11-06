#!/bin/bash
#SBATCH --job-name=swan_test
#SBATCH --partition=geocean
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --output=/lustre/geocean/WORK/users/javiert/projects/HyGloW/Netherlands/SWAN_test/slurm_logs/%A_%a.out
#SBATCH --error=/lustre/geocean/WORK/users/javiert/projects/HyGloW/Netherlands/SWAN_test/slurm_logs/%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --array=1-129%48
module use /nfs/software/geocean/modulefiles
module load swan/4151

case_dir=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /lustre/geocean/WORK/users/javiert/projects/HyGloW/Netherlands/SWAN_test/case_dirs.txt)
cd "$case_dir"
swan.exe > wrapper_out.log 2> wrapper_error.log
