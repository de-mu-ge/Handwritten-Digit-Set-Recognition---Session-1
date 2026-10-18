import numpy as np
# from src.sripts import Config
import cv2

import pandas as pd
class ImageData:
    def __init__(self, images, label):
        self.images = images
        self.label = label

train = r"D:\project\GIT\手写数字识别第一场\data\train.npy"
test = r"D:\project\GIT\手写数字识别第一场\data\test.npy"

arr = np.load(test, allow_pickle=True)      # (42000, 2)


# print(arr[17].images.shape)
# print(arr[17].label)

# arr = np.load(train, allow_pickle=True)
# print(arr[17].images)
# print(arr[17].label)
#
# cv2.imshow("arr", arr[79].images.reshape(28, 28))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 通过



from src.sripts.configs.Config import Config
test_data_path = Config.test_data_path
# train_data_npy_path = Config.test_data_npy_path

# # 读取本地 CSV 文件
df = pd.read_csv(test_data_path)
data = df.to_numpy()


#
# data = [data[i, 0] for i in range(data.shape[0])]
#
#
# # print(data)
# print(max(data))
# print(min(data))
# print(len(data))







