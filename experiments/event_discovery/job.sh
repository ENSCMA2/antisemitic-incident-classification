#!/bin/bash
#SBATCH --job-name=shj
#SBATCH --output=shj.out
#SBATCH --error=shj.err
#SBATCH --partition=general,r3lit,debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80GB
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00

python adl_study_ollm.py
