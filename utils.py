import torch
import numpy as np
import random
from torch import nn
from load_data import *
from thop import profile

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def count_para_FLOPS(model, num):
    model = model
    input = torch.randn(1, 1, train_shape_dict[dataset_name][0], train_shape_dict[dataset_name][1]).to(device)
    flops, params = profile(model, inputs=(input,))
    print("Para为：{}".format(params))
    print("平均FLOPS为：{}".format(flops))
    print("总FLOPS为：{}".format(flops * num))
