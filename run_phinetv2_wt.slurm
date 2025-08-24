#!/bin/bash
#SBATCH -p yamadau
#SBATCH -t 7-0
#SBATCH --mem=256G
#SBATCH -c 24
#SBATCH --gres=gpu:2

export DECORD_REWIND_RETRY_MAX=64
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 main_pretrain_phinetv2_walkingtours.py     --batch_size 64    --accum_iter 6     --model phinetv2_vit_small_patch16     --epochs 400     --warmup_epochs 40     --data_path /work/YamadaU/myamada/Python/dataset/WalkingTours2/  --log_dir log     --output_dir output_wt     --repeated_sampling 2 
