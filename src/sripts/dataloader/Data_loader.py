from torch.utils.data import DataLoader
import numpy as np

from src.sripts.dataset.Data_set import DataSet

from src.sripts import Config
data_rate = Config.data_rate

train_data_npy_path = Config.train_data_npy_path
test_data_npy_path = Config.test_data_npy_path

train_arr = np.load(train_data_npy_path)
train_arr = train_arr[:int(data_rate * train_arr.shape[0])]     # 33600

train_dataloader = DataLoader(DataSet(train_arr), batch_size=Config.batch_size, shuffle=True, num_workers=0)

exam_arr = np.load(train_data_npy_path)
exam_arr = exam_arr[int(data_rate * exam_arr.shape[0]):]
exam_dataloader = DataLoader(DataSet(exam_arr), batch_size=1, shuffle=False, num_workers=0)


# test_arr = np.load(test_data_npy_path)
# test_dataloader = DataLoader(DataSet(test_arr), batch_size=1, shuffle=False, num_workers=0)


if "__main__" == __name__:
    print(train_arr.shape[0])