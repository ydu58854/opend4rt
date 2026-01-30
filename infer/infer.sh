#!/usr/bin/env bash
set -euo pipefail

# Examples for inference

# Depth prediction at target resolution (256x256)
# python opend4rt/infer/infer_pointodyssey.py \
#   --scene-dir /path/to/scene \
#   --checkpoint /path/to/best.pt \
#   --mode depth \
#   --output-resolution target

# Depth prediction at original resolution
# python opend4rt/infer/infer_pointodyssey.py \
#   --scene-dir /path/to/scene \
#   --checkpoint /path/to/best.pt \
#   --mode depth \
#   --output-resolution orig

# Point track from a single point (u,v) in frame t_src
# python opend4rt/infer/infer_pointodyssey.py \
#   --scene-dir /path/to/scene \
#   --checkpoint /path/to/best.pt \
#   --mode track \
#   --u 0.5 --v 0.5 --t-src 0

# Dense pointcloud in a shared reference frame (t_cam_ref)
# python opend4rt/infer/infer_pointodyssey.py \
#   --scene-dir /path/to/scene \
#   --checkpoint /path/to/best.pt \
#   --mode pointcloud \
#   --t-cam-ref 0 \
#   --output-resolution target

# Intrinsics estimation for frame i
# python opend4rt/infer/infer_pointodyssey.py \
#   --scene-dir /path/to/scene \
#   --checkpoint /path/to/best.pt \
#   --mode intrinsics \
#   --i 0

# Extrinsics estimation between frame i and j
# python opend4rt/infer/infer_pointodyssey.py \
#   --scene-dir /path/to/scene \
#   --checkpoint /path/to/best.pt \
#   --mode extrinsics \
#   --i 0 --j 1

# Dense tracking via occupancy grid (Alg.1)
# python opend4rt/infer/infer_pointodyssey.py \
#   --scene-dir /path/to/scene \
#   --checkpoint /path/to/best.pt \
#   --mode dense_tracks \
#   --track-batch-points 256 \
#   --output-resolution target
