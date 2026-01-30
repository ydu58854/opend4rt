import argparse
import json
from collections import Counter
from pathlib import Path

import torch

ckpt_path = "/inspire/hdd/project/wuliqifa/public/dyh/d4rt/checkpoint/VideoMAE2/mae-g/vit_g_hybrid_pt_1200e.pth"
checkpoint = torch.load(ckpt_path, map_location="cpu")
print(checkpoint.keys(),type(checkpoint),checkpoint["model"].keys())