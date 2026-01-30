#!/usr/bin/env bash
set -euo pipefail

# Examples for Hydra-based training configs.
# Override any config value via key=value.

# PointOdyssey (default vit_b)
python train/train_pointodyssey.py data_root=/inspire/qb-ilm/project/wuliqifa/public/dyh_data/pointodyssey \

# PointOdyssey with specific ViT
# python opend4rt/train/train_pointodyssey.py data_root=/path/to/pointodyssey model=vit_l
# python opend4rt/train/train_pointodyssey.py data_root=/path/to/pointodyssey model=vit_h
# python opend4rt/train/train_pointodyssey.py data_root=/path/to/pointodyssey model=vit_g

# Enable wandb/tensorboard
# python opend4rt/train/train_pointodyssey.py data_root=/path/to/pointodyssey \
#   wandb.use=true wandb.project=opend4rt wandb.name=run_vit_b \
#   tensorboard.use=true tensorboard.logdir=runs/pointodyssey

