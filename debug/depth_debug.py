import cv2
import numpy as np

# 读取深度图，确保数据是 uint16 类型
depth1 = cv2.imread('/inspire/hdd/project/wuliqifa/public/dyh/d4rt/opend4rt/infer/ani2_48_depth/depths/depth_00012.png', cv2.IMREAD_UNCHANGED)
depth2 = cv2.imread('/inspire/qb-ilm/project/wuliqifa/public/dyh_data/pointodyssey/test/ani2/depths/depth_00000.png', cv2.IMREAD_UNCHANGED)
# depth1=depth1/65535*1000
# depth2=depth1/65535*1000

print(np.sum(depth1))

# 计算均方误差 (MSE)
mse = np.mean((depth1 - depth2) ** 2)

# 计算平均绝对误差 (MAE)
mae = np.mean(np.abs(depth1 - depth2))

print(f"MSE: {mse}")
print(f"MAE: {mae}")
