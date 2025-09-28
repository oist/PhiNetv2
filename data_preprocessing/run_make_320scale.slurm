#!/bin/bash
#SBATCH -p yamadau
#SBATCH -t 7-0
#SBATCH --mem=32G
#SBATCH -c 16

python3 make_320scale_modified.py --datadir ../datasets/k400/videos
