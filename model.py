import torch
from torch import nn
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
        ensemble_indices = [[], [0, 1], [2, 3, 4], [5, 6, 7, 8]]
        for i, (bb, ee) in enumerate(zip(self.backbone, self.exits), start=1):
            x = bb(x)
            output = ee(x)
            if i > 1:
                output = sum(self.ensemble[idx](res[j].detach() if self.if_train else res[j]) for j, idx in
                             enumerate(ensemble_indices[i - 1])) + output
            res.append(output)
            if not self.if_train and self.exit_confidence_based(res[-1], getattr(self, f"exit_threshold{i}")):
                Mark = i
                return res, Mark
        return (res, Mark) if not self.if_train else res

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