#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 5-0
#SBATCH --mem=32G
#SBATCH -c 16
#SBATCH --gres=gpu:1


#python3 download_WTours.py --output_folder /work/YamadaU/myamada/Python/WalkingTours
wget -O Dora_WalkingTours_Dataset_WildLife.mp4 "https://uvaauas.figshare.com/ndownloader/files/44469417"
wget -O Dora_WalkingTours_Dataset_Venice.mp4 "https://uvaauas.figshare.com/ndownloader/files/44469489"
wget -O Dora_WalkingTours_Dataset_Singapore.mp4 "https://uvaauas.figshare.com/ndownloader/files/44473088"
wget -O Dora_WalkingTours_Dataset_KualaLumpur.mp4 "https://uvaauas.figshare.com/ndownloader/files/44473407"
wget -O Dora_WalkingTours_Dataset_ChiangMai.mp4 "https://uvaauas.figshare.com/ndownloader/files/44473453"
wget -O Dora_WalkingTours_Dataset_Amsterdam.mp4 "https://uvaauas.figshare.com/ndownloader/files/44474683"
wget -O Dora_WalkingTours_Dataset_Zurich.mp4 "https://uvaauas.figshare.com/ndownloader/files/44509200"
wget -O Dora_WalkingTours_Dataset_Stockholm.mp4 "https://uvaauas.figshare.com/ndownloader/files/44510069"
wget -O Dora_WalkingTours_Dataset_Istanbul.mp4 "https://uvaauas.figshare.com/ndownloader/files/44510591"
wget -O Dora_WalkingTours_Dataset_Bangkok.mp4 "https://uvaauas.figshare.com/ndownloader/files/44511511"
