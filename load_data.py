import torch
import torch.utils.data as Data
import numpy as np

dataset_name = 'UCI'
batch_size = 64

train_shape_dict = {
    'UCI':[128, 9],
}

category_dict = {
    'UCI': 6,
}


train_x_list = dataset_name + '/train_x.npy'
train_y_list = dataset_name + '/train_y.npy'
test_x_list = dataset_name + '/test_x.npy'
test_y_list = dataset_name + '/test_y.npy'

class HAR(Data.Dataset):
    def __init__(self, filename_x, filename_y):
        self.filename_x = filename_x
        self.filename_y = filename_y

    def HAR_data(self):
        """更改x的维度,加载x和y"""
        data_x = np.load(self.filename_x)
        print(data_x.shape)
        data_x = data_x.reshape(-1, 1, data_x.shape[-2], data_x.shape[-1])
        print(data_x.shape)
        # print(data_x_raw.shape)
        # print(data_x.shape)
        data_y = np.load(self.filename_y)
        # data_y = data_y.argmax(axis=1)  #uci的y格式为【1，0，0，0，0，0】 需要处理y的第二个维度
        print(data_y.shape)
        torch_dataset = Data.TensorDataset(torch.from_numpy(data_x), torch.from_numpy(data_y))
        return torch_dataset

data_train = HAR(train_x_list, train_y_list)
torch_train_dataset = data_train.HAR_data()
print("Train data loading completed")
data_test = HAR(test_x_list, test_y_list)
torch_test_dataset = data_test.HAR_data()
print("Test data loading completed")


train_loader = Data.DataLoader(dataset=torch_train_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=0,)

test_loader = Data.DataLoader(dataset=torch_test_dataset,
                               batch_size=1,
                               shuffle=True,
                               num_workers=0,)