set -euo pipefail
export CUDA_VISIBLE_DEVICES=0  # 只使用 GPU 0

python train/train_pointodyssey.py data_root=/inspire/qb-ilm/project/wuliqifa/public/dyh_data/pointodyssey split=train "$@"

