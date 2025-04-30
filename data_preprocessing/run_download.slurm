#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 5-0
#SBATCH --mem=32G
#SBATCH -c 16
#SBATCH --gres=gpu:1

./k400_downloader.sh
