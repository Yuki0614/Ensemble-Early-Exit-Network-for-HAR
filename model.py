import torch
from torch import nn
from load_data import dataset_name
from load_data import category_dict
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class layer(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding):
        super(layer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)

class ensemble_layer(nn.Module):
    def __init__(self, ensemble_mode):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, category_dict[dataset_name]).to(device), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1, category_dict[dataset_name]).to(device), requires_grad=True)
        self.ensemble_mode = ensemble_mode

    def forward(self, x):
        if self.ensemble_mode == 'additive':
            weight = self.weight.exp()
            x = x * weight + self.bias.exp()
            x = x/x.sum(1,keepdim=True)
            return (x +1e-6).log()
        elif self.ensemble_mode == 'geometric':
            weight = torch.exp(self.weight)
            x = x * weight
            return x.log_softmax(-1) + self.bias
        elif self.ensemble_mode == 'normal':
            x = x * self.weight + self.bias
            return x

class Net_EE(nn.Module):
    def __init__(self, if_train, train_shape, category, exit_threshold):
        super(Net_EE, self).__init__()
        self.if_train = if_train
        self.train_shape = train_shape
        self.category = category
        self.exit_threshold1 = torch.tensor([exit_threshold[0]], dtype=torch.float32).to(device)
        self.exit_threshold2 = torch.tensor([exit_threshold[1]], dtype=torch.float32).to(device)
        self.exit_threshold3 = torch.tensor([exit_threshold[2]], dtype=torch.float32).to(device)
        self.backbone = nn.ModuleList()
        self.exits = nn.ModuleList()
        self.build_backbone()
        self.build_exits(train_shape=self.train_shape, category=self.category)
        self.ensemble = nn.ModuleList()
        for i in range(9):
            self.ensemble.append(ensemble_layer(ensemble_mode='normal'))

    def build_backbone(self):
        c1_1 = layer(1, 16, (6, 1), (3, 1), (1, 0))
        c1_2 = layer(16, 16, (3, 1), (1, 1), (1, 0))
        c2 = layer(16, 64, (6, 1), (3, 1), (1, 0))
        c3 = layer(64, 64, (3, 1), (1, 1), (1, 0))
        c4 = layer(64, 128, (6, 1), (3, 1), (1, 0))
        c1 = nn.Sequential(c1_1,c1_2,)
        self.backbone.append(c1)
        self.backbone.append(c2)
        self.backbone.append(c3)
        self.backbone.append(c4)

    def build_exits(self, train_shape, category):
        ee1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, train_shape[-1])),
            nn.Flatten(),
            nn.Linear(16 * (train_shape[-1]), category))
        self.exits.append(ee1)

        ee2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, train_shape[-1])),
            nn.Flatten(),
            nn.Linear(64 * (train_shape[-1]), category))
        self.exits.append(ee2)

        ee3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, train_shape[-1])),
            nn.Flatten(),
            nn.Linear(64 * (train_shape[-1]), category))
        self.exits.append(ee3)

        ee4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, train_shape[-1])),
            nn.Flatten(),
            nn.Linear(128 * (train_shape[-1]), category))
        self.exits.append(ee4)

    def forward(self, x):
        res = []
        Mark = 0
        if self.if_train:
            for i, bb, ee in zip([1,2,3,4], self.backbone, self.exits):
                if i == 1:
                    x = bb(x)
                    output0 = ee(x)
                    res.append(output0)
                if i == 2:
                    x = bb(x)
                    output1 = ee(x)
                    output1 = self.ensemble[0](output0.detach()) + self.ensemble[1](output1)
                    res.append(output1)
                if i == 3:
                    x = bb(x)
                    output2 = ee(x)
                    output2 = self.ensemble[2](output0.detach()) + self.ensemble[3](output1.detach()) + self.ensemble[4](output2)
                    res.append(output2)
                if i == 4:
                    x = bb(x)
                    output3 = ee(x)
                    output3 = self.ensemble[5](output0.detach()) + self.ensemble[6](output1.detach()) + self.ensemble[7](output2.detach()) + self.ensemble[8](output3)
                    res.append(output3)
            return res
        else:
            for i, bb, ee in zip([1,2,3,4], self.backbone, self.exits):
                if i == 1:
                    x = bb(x)
                    output0 = ee(x)
                    res.append(output0)
                    if self.exit_confidence_based(res[-1], self.exit_threshold1):
                        Mark = 1
                        return res, Mark
                if i == 2:
                    x = bb(x)
                    output1 = ee(x)
                    output1 = self.ensemble[0](output0) + self.ensemble[1](output1)
                    res.append(output1)
                    if self.exit_confidence_based(res[-1], self.exit_threshold2):
                        Mark = 2
                        return res, Mark
                if i == 3:
                    x = bb(x)
                    output2 = ee(x)
                    output2 = self.ensemble[2](output0) + self.ensemble[3](output1) + self.ensemble[4](output2)
                    res.append(output2)
                    if self.exit_confidence_based(res[-1], self.exit_threshold3):
                        Mark = 3
                        return res, Mark
                if i == 4:
                    x = bb(x)
                    output3 = ee(x)
                    output3 = self.ensemble[5](output0) + self.ensemble[6](output1) + self.ensemble[7](output2) + self.ensemble[8](output3)
                    res.append(output3)
            return res, Mark

    def exit_confidence_based(self, x, threshold):
        prob = nn.functional.softmax(x, dim=-1)
        max_prob = torch.max(prob)
        return max_prob > threshold

    def exit_entropy_based(self, x, threshold):
        prob = nn.functional.softmax(x, dim=-1)
        log_prob = torch.log2(prob)
        entropy = -1 * torch.sum(prob * log_prob, axis=-1)
        return entropy > threshold

    def exit_patience_based(self, x, patience, last_x):
        if x.argmax(1) == last_x:
            patience += 1
        else:
            patience = 1
        last_x = x.argmax(1)
        return last_x, patience, patience == self.exit_threshold