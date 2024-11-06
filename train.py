from model import Net_EE
from load_data import *
from utils import setup_seed
from torch import nn

train_data_size = len(torch_train_dataset)
print(train_data_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
setup_seed(77)

learning_rate, weight_decay, epoch = 0.0001, 0.01, 300
net = Net_EE(if_train=True, train_shape=train_shape_dict[dataset_name], category=category_dict[dataset_name], exit_threshold=[1, 1, 1])
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
total_train_step = 0
max_acc = 0

for i in range(epoch):
    total_train_loss = 0
    total_train_acc_1, total_train_acc_2, total_train_acc_3, total_train_acc_last = 0, 0, 0, 0
    acc_list = [0, 0, 0, 0]
    distribution_list = [0, 0, 0, 0]
    print("第{}轮训练开始".format(i + 1))

    for step, (b_x, b_y) in enumerate(train_loader, 1):
        net.train()
        net.if_train = True
        b_x = b_x.to(device).float()
        res = net(b_x)
        outputs_last, outputs1, outputs2, outputs3 = res[-1], res[0], res[1], res[2]
        b_y = b_y.to(device).long()
        loss1 = loss_function(outputs1, b_y)
        loss2 = loss_function(outputs2, b_y)
        loss3 = loss_function(outputs3, b_y)
        loss_last = loss_function(outputs_last, b_y)
        loss_all = loss_last * 4 + loss1 * 1 + loss2 * 2 + loss3 * 3


        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        total_train_loss = total_train_loss + loss_all
        acc_train_last = (outputs_last.argmax(1) == b_y).sum()
        acc_train_1 = (outputs1.argmax(1) == b_y).sum()
        acc_train_2 = (outputs2.argmax(1) == b_y).sum()
        acc_train_3 = (outputs3.argmax(1) == b_y).sum()
        total_train_acc_last = total_train_acc_last + acc_train_last
        total_train_acc_1 = total_train_acc_1 + acc_train_1
        total_train_acc_2 = total_train_acc_2 + acc_train_2
        total_train_acc_3 = total_train_acc_3 + acc_train_3

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, loss:{}".format(total_train_step, loss_all))
    print("整体训练集上的loss:{}".format(total_train_loss))

    net.eval()
    net.if_train = False

    target_num = torch.zeros((1, category_dict[dataset_name]))
    predict_num = torch.zeros((1, category_dict[dataset_name]))
    acc_num = torch.zeros((1, category_dict[dataset_name]))

    with torch.no_grad():
        for step, (b_x, b_y) in enumerate(test_loader):
            b_x = b_x.to(device).float()
            res, Mark = net(b_x)
            b_y = b_y.to(device).long()
            distribution_list[Mark] += 1
            acc_list[Mark] += (res[-1].argmax(1) == b_y).sum().item()

            _, predicted = res[-1].max(1)
            pre_mask = torch.zeros(res[-1].size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
            predict_num += pre_mask.sum(0)
            tar_mask = torch.zeros(res[-1].size()).scatter_(1, b_y.cpu().view(-1, 1), 1.)
            target_num += tar_mask.sum(0)
            acc_mask = pre_mask * tar_mask
            acc_num += acc_mask.sum(0)

    recall = acc_num / target_num
    precision = acc_num / predict_num
    F1 = 2 * recall * precision / (recall + precision)
    accuracy = 100. * acc_num.sum(1) / target_num.sum(1)

    print("最后分类器在整体测试集上的正确率:{}".format(acc_list[0] / (distribution_list[0] + 1e-4)))
    print("第一个分类器在整体测试集上的正确率:{}".format(acc_list[1] / (distribution_list[1] + 1e-4)))
    print("第二个分类器在整体测试集上的正确率:{}".format(acc_list[2] / (distribution_list[2] + 1e-4)))
    print("第三个分类器在整体测试集上的正确率:{}".format(acc_list[3] / (distribution_list[3] + 1e-4)))
    print("总体准确率：{}".format(sum(acc_list) / sum(distribution_list)))
    print("f1分数为：{}".format(F1.mean()))
    cur_acc = acc_list[0] / (distribution_list[0] + 1e-4)
    if cur_acc > max_acc:
        torch.save(net.state_dict(), dataset_name + "/net_EE_model")
        max_acc = cur_acc
        print("save complete")

# torch.save(net.state_dict(), dataset_name + "/net_EE_model")