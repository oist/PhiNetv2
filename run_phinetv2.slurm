#!/bin/bash
#SBATCH -p yamadau
#SBATCH -t 7-0
#SBATCH --mem=256G
#SBATCH -c 24
#SBATCH --gres=gpu:4

python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 main_pretrain_phinetv2.py     --batch_size 32    --accum_iter 6     --model phinetv2_vit_small_patch16     --epochs 400     --warmup_epochs 40     --data_path /home/m/makoto-yamada/work/Python/PhiNetv2/datasets     --log_dir log     --output_dir output     --norm_pix_loss     --repeated_sampling 2 
