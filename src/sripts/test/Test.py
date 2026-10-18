import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

#---------------------------
# Config:
from src.sripts.configs.Config import Config

# DataLoader:
from src.sripts.dataset.Data_set import DataSet
data_loader = DataLoader(DataSet(np.load(Config.test_data_npy_path)),
                         batch_size=1,
                         shuffle=False,
                         num_workers=0
)

# Model
from src.sripts.models.Model import Model2828
model = Model2828()
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(Config.model_save_path))
# model = model.to(device)output = torch.max(outputs, 1)[1]
#---------------------------

def test():
    # df = pd.read_csv(Config.output_dir)
    # print(df.to_numpy().shape)
    # print(df.head())
    arr = []

    for index, (data, a) in enumerate(data_loader):

        out = model(data)
        out = torch.max(out, 1)[1]
        out = out.detach().cpu().numpy()[0]

        arr.append(np.array([index + 1, out]))

        #
        # print(out)
        # print(type(out))
        # break
    arr = np.array(arr)
    arr = np.stack(arr)
    print(arr.shape)
    df = pd.DataFrame(arr, columns=['ImageId', 'Label'])

    # 3. 写入 CSV 文件
    # index=False 表示不保存行索引
    df.to_csv('output.csv', index=False)


if "__main__" == __name__:
    # pass
    # print(data_loader)
    test()