#!/bin/bash
#SBATCH --job-name=score_night
#SBATCH --partition=geocean
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/geocean/WORK/users/javiert/projects/HyGloW/Netherlands/SWAN_test/score_overnight.out
#SBATCH --error=/lustre/geocean/WORK/users/javiert/projects/HyGloW/Netherlands/SWAN_test/score_overnight.err
cd /lustre/geocean/WORK/users/javiert/projects/HyGloW/Netherlands
export GDAL_DATA=/nfs/home/geocean/javiert/miniforge3/envs/bluemath-312/share/gdal PROJ_DATA=/nfs/home/geocean/javiert/miniforge3/envs/bluemath-312/share/proj PYTHONPATH=.
/nfs/home/geocean/javiert/miniforge3/envs/bluemath-312/bin/python SWAN_test/snapwave/compare.py > SWAN_test/snapwave/results_2026-11-03.txt 2>&1
/nfs/home/geocean/javiert/miniforge3/envs/bluemath-312/bin/python SWAN_test/compare_tests_6.py > SWAN_test/results_2026-11-03_round4.txt 2>&1
