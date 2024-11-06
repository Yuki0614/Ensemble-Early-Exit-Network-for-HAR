from model import Net_EE
from load_data import *
from tqdm import tqdm
import time

test_data_size = len(torch_test_dataset)
print(test_data_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Net_EE(if_train=False, train_shape=train_shape_dict[dataset_name], category=category_dict[dataset_name], exit_threshold=[1, 1, 1])
net.load_state_dict(torch.load(dataset_name + "/net_EE_model", map_location=device))
net.to(device)

# model_dict = net.state_dict()
# for k, v in model_dict.items():
#     print(k)

total_test_acc, total_test_acc_1, total_test_acc_2, total_test_acc_3, total_test_acc_last = 0, 0, 0, 0, 0
test_bar = tqdm(test_loader)
counts1, counts2, counts3, counts_last = 0, 0, 0, 0

start = time.time()
net.eval()
with torch.no_grad():
    for step, (b_x, b_y) in enumerate(test_bar):
        b_x = b_x.to(device).float()
        res, Mark = net(b_x)
        b_y = b_y.to(device).long()

        if Mark == 0:
            counts_last += 1
            acc_test_last = (res[-1].argmax(1) == b_y).sum()
            total_test_acc_last = total_test_acc_last + acc_test_last
        if Mark == 1:
            counts1 += 1
            acc_test_1 = (res[-1].argmax(1) == b_y).sum()
            total_test_acc_1 = total_test_acc_1 + acc_test_1
        if Mark == 2:
            counts2 += 1
            acc_test_2 = (res[-1].argmax(1) == b_y).sum()
            total_test_acc_2 = total_test_acc_2 + acc_test_2
        if Mark == 3:
            counts3 += 1
            acc_test_3 = (res[-1].argmax(1) == b_y).sum()
            total_test_acc_3 = total_test_acc_3 + acc_test_3

        # 对比标签是否相等，然后布尔值求和，相等即为1，不等为0
        acc_test = (res[-1].argmax(1) == b_y).sum()
        total_test_acc = total_test_acc + acc_test

end = time.time()
print("模型为：Ensemble Early Exit")
print("数据集为：{}".format(dataset_name))
print("整体测试集上的正确率：{}".format(total_test_acc / test_data_size))
print("测试集用时：{}".format(end - start))
print("------------------------------")
print("第一个中间出口提前退出的数量：{}".format(counts1))
if counts1 != 0:
    print("第一个中间出口准确率：{}".format(total_test_acc_1 / counts1))
print("------------------------------")
print("第二个中间出口提前退出的数量：{}".format(counts2))
if counts2 != 0:
    print("第二个中间出口准确率：{}".format(total_test_acc_2 / counts2))
print("------------------------------")
print("第三个中间出口提前退出的数量：{}".format(counts3))
if counts3 != 0:
    print("第三个中间出口准确率：{}".format(total_test_acc_3 / counts3))
print("------------------------------")
print("最后出口退出的数量：{}".format(counts_last))
if counts_last != 0:
    print("最后出口准确率：{}".format(total_test_acc_last / counts_last))

