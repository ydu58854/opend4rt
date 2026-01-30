#!/usr/bin/env bash
set -euo pipefail

# Examples for Hydra-based training configs.
# Override any config value via key=value.

# PointOdyssey (default vit_b)
# python train/train_pointodyssey.py data_root=/inspire/qb-ilm/project/wuliqifa/public/dyh_data/pointodyssey 

# python /inspire/hdd/global_user/chenxinyan-240108120066/youjunqi/occupy_gpu.py



#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --standalone --nproc_per_node=8 train/train_pointodyssey.py data_root=/inspire/qb-ilm/project/wuliqifa/public/dyh_data/pointodyssey split=train "$@"

python /inspire/hdd/global_user/chenxinyan-240108120066/youjunqi/occupy_gpu.py
# PointOdyssey with specific ViT
# python opend4rt/train/train_pointodyssey.py data_root=/path/to/pointodyssey model=vit_l
# python opend4rt/train/train_pointodyssey.py data_root=/path/to/pointodyssey model=vit_h
# python opend4rt/train/train_pointodyssey.py data_root=/path/to/pointodyssey model=vit_g

# Enable wandb/tensorboard
# python opend4rt/train/train_pointodyssey.py data_root=/path/to/pointodyssey \
#   wandb.use=true wandb.project=opend4rt wandb.name=run_vit_b \
#   tensorboard.use=true tensorboard.logdir=runs/pointodyssey

