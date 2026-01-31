#!/usr/bin/env bash
set -euo pipefail

# Examples for Hydra-based training configs.
# Override any config value via key=value.

# PointOdyssey (default vit_b)
# python train/train_pointodyssey.py data_root=/inspire/qb-ilm/project/wuliqifa/public/dyh_data/pointodyssey 

# python /inspire/hdd/global_user/chenxinyan-240108120066/youjunqi/occupy_gpu.py
source /inspire/hdd/global_user/chenxinyan-240108120066/youjunqi/condainit2
conda activate d4rt


#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=4,5,6,7 TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=INFO TORCH_NCCL_ASYNC_ERROR_HANDLING=1
torchrun --standalone --nproc_per_node=4 train/train_pointodyssey.py data_root=/inspire/qb-ilm/project/wuliqifa/public/dyh_data/pointodyssey split=train "$@"\
    num_workers=4 tensorboard.logdir=runs/pointodyssey_4workers

# PointOdyssey with specific ViT
# python opend4rt/train/train_pointodyssey.py data_root=/path/to/pointodyssey model=vit_l
# python opend4rt/train/train_pointodyssey.py data_root=/path/to/pointodyssey model=vit_h
# python opend4rt/train/train_pointodyssey.py data_root=/path/to/pointodyssey model=vit_g

# Enable wandb/tensorboard
# python opend4rt/train/train_pointodyssey.py data_root=/path/to/pointodyssey \
#   wandb.use=true wandb.project=opend4rt wandb.name=run_vit_b \
#   tensorboard.use=true tensorboard.logdir=runs/pointodyssey

