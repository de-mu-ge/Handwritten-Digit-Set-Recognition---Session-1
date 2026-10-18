import dateutil.relativedelta
import pandas as pd
import numpy as np
# import cv2


# -----------Train------------
# （28， 28）

from src.sripts.configs.Config import Config
train_data_path = Config.train_data_path
train_data_npy_path = Config.train_data_npy_path

# 读取本地 CSV 文件
df = pd.read_csv(train_data_path)
data = df.to_numpy()

np.save(train_data_npy_path, data)


# ----------Test-------------
test_data_path = Config.test_data_path
test_data_npy_path = Config.test_data_npy_path

df = pd.read_csv(test_data_path)
data = df.to_numpy()
print(data.shape)

add = np.zeros(data.shape[0]).reshape(-1, 1)
print(add.shape)
data = np.concatenate([add, data], axis=1)
print(data.shape)

np.save(test_data_npy_path, data)















# label = data[0, 0]
# data = data[0, 1:]
# data = data.reshape(28, -1)
# # data = data[:, 1:]
# data = data.astype(np.uint8)
# print(data.shape)
#
# # data = np.array(data, dtype="unit8")
#
# cv2.imshow("data", data)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
