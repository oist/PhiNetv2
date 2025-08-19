#!/bin/bash
#SBATCH -p yamadau
#SBATCH -t 5-0
#SBATCH --mem=32G
#SBATCH -c 16
#SBATCH --gres=gpu:1

python3 split_wt_mp4.py
