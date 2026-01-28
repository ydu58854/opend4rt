# D4RT Datasets 模块介绍

## 文件结构

```
datasets/
├── __init__.py           # 模块入口，导出所有公开接口
├── base_config.py        # 配置
├── base_dataset.py       # 基类
├── pointodyssey.py       # PointOdyssey 数据集实现
├── trajectory_sampler.py # 查询采样器,gt计算
├── d4rt_dataset.py       # JSON 格式数据集（原始版本）
├── utils.py              # 工具函数
```

## 各文件功能说明

### base_config.py - 配置类

定义数据集配置的基类和具体实现：

| 类名 | 说明 |
|------|------|
| `BaseDatasetConfig` | 基础配置，包含 root、split、num_frames 等 |
| `Base4DDatasetConfig` | 4D 轨迹数据集配置，增加采样比例参数 |
| `PointOdysseyConfig` | PointOdyssey 专用配置 |

**修改默认数据集路径**：编辑文件顶部的 `DATASET_PATHS` 字典。

### base_dataset.py - 基类

| 类名 | 说明 |
|------|------|
| `Base4DTrajectoryDataset` | 4D 轨迹数据集基类，定义数据加载接口 |
| `Base3DDataset` | 3D 共视性数据集基类（预留） |

扩展新数据集时需实现以下抽象方法：
- `_load_scene_list()` - 加载场景列表
- `_load_trajectories()` - 加载轨迹数据
- `_load_camera_params()` - 加载相机参数
- `_load_frames()` - 加载图像帧

### pointodyssey.py - PointOdyssey 数据集

| 类名 | 说明 |
|------|------|
| `PointOdysseyDataset` | 完整的 PointOdyssey 数据集实现 |
| `PointOdysseySingleFrameDataset` | 单帧版本，用于过拟合测试 |

### trajectory_sampler.py - 查询采样器

| 类名 | 说明 |
|------|------|
| `TrajectoryQuerySampler` | 多帧轨迹查询采样，支持边缘/同帧/随机采样 |
| `SingleFrameTrajectoryQuerySampler` | 单帧查询采样 |

### d4rt_dataset.py - JSON 数据集

| 类名/函数 | 说明 |
|-----------|------|
| `D4RTDataset` | 从 JSON 标注文件加载数据 |
| `d4rt_collate_fn` | DataLoader 的 collate 函数 |
| `build_dataloader` | 快速构建 DataLoader |

### utils.py - 工具函数

| 类别 | 函数 |
|------|------|
| 图像加载 | `load_image`, `load_image_tensor`, `load_depth_16bit` |
| 尺寸变换 | `resize_image`, `resize_depth` |
| 坐标变换 | `normalize_uv`, `world_to_camera`, `project_to_image` |
| 帧采样 | `uniform_sample_frames` |
| 相机参数 | `build_intrinsics_matrix`, `scale_intrinsics` |
| 法向量 | `estimate_normals_from_depth` |
| 边缘检测 | `sobel_edge_detection` |

---

## 数据接口

### 单样本输出格式

`dataset[idx]` 返回一个字典，包含以下字段：

```python
{
    "meta": {
        "aspect_ratio": float,      # 宽高比 W/H
        "img_patch_size": int,      # 查询 patch 大小（默认 3）
        "align_corners": bool,      # grid_sample 对齐模式
    },
    "images": Tensor,               # (C, T, H, W) RGB 图像序列，值域 [0, 1]
    "query": Tensor,                # (N, 5) 查询向量
    "targets": {
        "L3D": Tensor,              # (N, 3) 相机坐标系下的 3D 位置
        "L2D": Tensor,              # (N, 2) 目标帧的归一化 UV 坐标
        "Lvis": Tensor,             # (N, 1) 可见性 (0/1)
        "Ldisp": Tensor,            # (N, 3) 2D 视差 [du, dv, 0]
        "Lconf": Tensor,            # (N, 1) 置信度
        "Lnormal": Tensor,          # (N, 3) 表面法向量（相机坐标系）
    }
}
```

### Query 格式

每个查询为 5 维向量 `[u, v, t_src, t_tgt, t_cam]`：

| 字段 | 说明 |
|------|------|
| `u, v` | 源帧像素位置，归一化到 [0, 1] |
| `t_src` | 源帧索引（采样后的索引，0~47） |
| `t_tgt` | 目标帧索引 |
| `t_cam` | 相机帧索引（以概率 0.4 令 t_cam = t_tgt，否则均匀随机采样） |

### 坐标系约定

- **相机坐标系**：OpenCV 约定（+X 右，+Y 下，+Z 前）
- **L3D**：可见点的 Z > 0
- **Lnormal**：朝向相机的表面法向量 Z < 0
- **UV 坐标**：归一化到 [0, 1]，使用 `align_corners=True`

### Batch 输出格式（经过 collate）

```python
{
    "meta": {
        "aspect_ratio": Tensor,     # (B,)
        "img_patch_size": Tensor,   # (B,)
        "align_corners": Tensor,    # (B,)
    },
    "images": Tensor,               # (B, C, T, H, W)
    "query": Tensor,                # (B, N_max, 5) 已 padding
    "targets": {
        "L3D": Tensor,              # (B, N_max, 3)
        "L2D": Tensor,              # (B, N_max, 2)
        "Lvis": Tensor,             # (B, N_max, 1)
        "Ldisp": Tensor,            # (B, N_max, 3)
        "Lconf": Tensor,            # (B, N_max, 1)
        "Lnormal": Tensor,          # (B, N_max, 3)
        "query_mask": Tensor,       # (B, N_max) 有效查询掩码
    }
}
```

---

## 训练使用示例

### 基本用法

```python
from opend4rt.datasets import (
    PointOdysseyDataset,
    PointOdysseyConfig,
    d4rt_collate_fn,
)
from torch.utils.data import DataLoader

# 1. 创建配置
config = PointOdysseyConfig(
    split="train",
    num_frames=48,              # 每个样本的帧数
    num_queries=2048,           # 每个样本的查询数
    target_resolution=(512, 288),  # (W, H)
    query_edge_ratio=0.3,       # 30% 边缘采样，70% 随机采样
    tcam_equals_ttgt_ratio=0.4, # 40% t_cam = t_tgt (按D4RT论文Appendix A)
)

# 2. 创建数据集
dataset = PointOdysseyDataset(config)

# 3. 创建 DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    collate_fn=d4rt_collate_fn,  # 必须使用这个 collate 函数
)

# 4. 训练循环
for batch in dataloader:
    images = batch["images"]        # (B, 3, 48, 288, 512)
    queries = batch["query"]        # (B, N, 5)
    targets = batch["targets"]

    # 获取有效查询掩码
    query_mask = targets["query_mask"]  # (B, N) bool

    # 前向传播
    pred = model(images, queries)

    # 计算损失（只在有效查询上）
    loss = compute_loss(pred, targets, query_mask)
```

### 修改默认数据集路径

编辑 `base_config.py` 顶部：

```python
DATASET_PATHS = {
    "pointodyssey": "/your/path/to/pointodyssey",
    # 添加其他数据集...
}
```

或在创建配置时指定：

```python
config = PointOdysseyConfig(
    root="/custom/path/to/pointodyssey",
    split="train",
)
```

---

## 扩展新数据集

继承 `Base4DTrajectoryDataset` 并实现方法：

```python
from opend4rt.datasets import Base4DTrajectoryDataset, Base4DDatasetConfig

class MyDataset(Base4DTrajectoryDataset):
    def _load_scene_list(self):
        # 返回场景列表 [{"scene_id": ..., ...}, ...]
        pass

    def _load_trajectories(self, scene_id):
        # 返回 {"trajs_2d": ..., "trajs_3d": ..., "valids": ..., "visibs": ...}
        pass

    def _load_camera_params(self, scene_id):
        # 返回 {"intrinsics": ..., "extrinsics": ...}
        pass

    def _load_frames(self, scene_id, frame_indices):
        # 返回 (C, T, H, W) tensor
        pass

    # 可选：实现深度和法向量加载
    def _load_depths(self, scene_id, frame_indices):
        pass

    def _load_normals(self, scene_id, frame_indices):
        pass
```

---
