#!/bin/bash
#SBATCH -p yamadau
#SBATCH -t 7-0
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH --gres=gpu:1

python3 make_7x7split.py --datadir ../../dataset/
