#!/bin/bash
#SBATCH --job-name=score_night2
#SBATCH --partition=geocean
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/geocean/WORK/users/javiert/projects/HyGloW/Netherlands/SWAN_test/score_overnight_2.out
#SBATCH --error=/lustre/geocean/WORK/users/javiert/projects/HyGloW/Netherlands/SWAN_test/score_overnight_2.err
cd /lustre/geocean/WORK/users/javiert/projects/HyGloW/Netherlands
export GDAL_DATA=/nfs/home/geocean/javiert/miniforge3/envs/bluemath-312/share/gdal PROJ_DATA=/nfs/home/geocean/javiert/miniforge3/envs/bluemath-312/share/proj PYTHONPATH=.
/nfs/home/geocean/javiert/miniforge3/envs/bluemath-312/bin/python SWAN_test/compare_tests_7.py > SWAN_test/results_2026-11-03_wind110.txt 2>&1
