# input size : 42000
import numpy as np
from torch.utils.data import Dataset
import torch


class DataSet(Dataset):
    def __init__(self, data:np.ndarray):
        self.images = [torch.tensor(data[i, 1:]).float().reshape(1, 28, 28) for i in range(data.shape[0])]
        self.labels = [data[i, 0] for i in range(data.shape[0])]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]       # tuple


if __name__ == "__main__":
    import numpy as np

    array = np.ones((10, 785))

    # print(array[3].label.shape)
    # print(array[3].images.shape)
    data = DataSet(array)

    print(data[9][0].shape)
    print(data[7][1])