#!/bin/bash
#SBATCH -p yamadau
#SBATCH -t 7-0
#SBATCH --mem=32G
#SBATCH -c 16
#SBATCH --gres=gpu:1

python3 make_256scale_modified.py --datadir ../datasets/k400/videos
