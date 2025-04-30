#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 7-0
#SBATCH --mem=32G
#SBATCH -c 16
#SBATCH --gres=gpu:V100:1

./extract_rsp.sh
