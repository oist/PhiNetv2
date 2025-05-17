#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 7-0
#SBATCH --mem=32G
#SBATCH -c 16
#SBATCH --gres=gpu:1

sh k400_extractor.sh
