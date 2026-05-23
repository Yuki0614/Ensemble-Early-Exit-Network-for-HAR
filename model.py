import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class NormalEnsembleLayer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, num_classes), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1, num_classes), requires_grad=True)

    def forward(self, x):
        return x * self.weight + self.bias


class GeometricEnsembleLayer(nn.Module):
    def __init__(self, num_sources, num_classes):
        super().__init__()
        self.logit_weights = nn.Parameter(torch.zeros(num_sources, 1), requires_grad=True)
        self.log_bias = nn.Parameter(torch.zeros(1, num_classes), requires_grad=True)

    def forward(self, logits_list):
        log_probs = [F.log_softmax(logits, dim=-1) for logits in logits_list]
        stacked_log_probs = torch.stack(log_probs, dim=0)
        weights = F.softplus(self.logit_weights) + 1e-6
        logits = self.log_bias + (stacked_log_probs * weights.unsqueeze(1)).sum(dim=0)
        return F.log_softmax(logits, dim=-1)


class Net_EE(nn.Module):
    def __init__(
        self,
        train_shape=None,
        category=None,
        exit_threshold=None,
        in_channels=1,
        layers=None,
        backbone_channels=None,
        exit_layers=None,
        ensemble_mode="normal",
    ):
        super().__init__()
        if train_shape is None:
            raise ValueError("train_shape must be provided")
        if category is None:
            raise ValueError("category must be provided")
        if ensemble_mode not in ("none", "normal", "ensemble"):
            raise ValueError(f"Unsupported ensemble mode: {ensemble_mode}")

        self.train_shape = train_shape
        self.category = category
        self.ensemble_mode = ensemble_mode
        self.exit_thresholds = exit_threshold or [1.0, 1.0, 1.0]
        self.layer_configs = layers or self._build_legacy_layer_configs(backbone_channels)
        self.exit_layers = exit_layers or [2, 3, 4, 5]

        for index, threshold in enumerate(self.exit_thresholds, start=1):
            self.register_buffer(f"exit_threshold{index}", torch.tensor(float(threshold)))

        self.backbone = nn.ModuleList()
        self.exits = nn.ModuleList()
        self.normal_ensemble = nn.ModuleList()
        self.geometric_ensemble = nn.ModuleList()

        self.build_backbone(in_channels)
        self.build_exits(train_shape=self.train_shape, category=self.category)
        self.build_ensemble(category, ensemble_mode)

    def set_exit_thresholds(self, thresholds):
        if len(thresholds) != len(self.exit_layers) - 1:
            raise ValueError(f"Expected {len(self.exit_layers) - 1} thresholds, got {len(thresholds)}")
        self.exit_thresholds = thresholds
        for index, threshold in enumerate(thresholds, start=1):
            getattr(self, f"exit_threshold{index}").fill_(float(threshold))

    @staticmethod
    def _build_legacy_layer_configs(backbone_channels):
        channels = backbone_channels or [16, 64, 64, 128]
        return [
            {"kernel_size": [6, 1], "stride": [3, 1], "out_channels": channels[0]},
            {"kernel_size": [3, 1], "stride": [1, 1], "out_channels": channels[0]},
            {"kernel_size": [6, 1], "stride": [3, 1], "out_channels": channels[1]},
            {"kernel_size": [3, 1], "stride": [1, 1], "out_channels": channels[2]},
            {"kernel_size": [6, 1], "stride": [3, 1], "out_channels": channels[3]},
        ]

    @staticmethod
    def _as_tuple(value):
        return tuple(value) if isinstance(value, list) else value

    def build_backbone(self, in_channels):
        current_channels = in_channels
        for layer_config in self.layer_configs:
            out_channels = layer_config["out_channels"]
            kernel_size = self._as_tuple(layer_config["kernel_size"])
            stride = self._as_tuple(layer_config["stride"])
            padding = self._as_tuple(layer_config.get("padding", [1, 0]))
            self.backbone.append(ConvBlock(current_channels, out_channels, kernel_size, stride, padding))
            current_channels = out_channels

    def build_exits(self, train_shape, category):
        sensor_dim = train_shape[-1]
        for layer_number in self.exit_layers:
            channels = self.layer_configs[layer_number - 1]["out_channels"]
            self.exits.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, sensor_dim)),
                    nn.Flatten(),
                    nn.Linear(channels * sensor_dim, category),
                )
            )

    def build_ensemble(self, category, ensemble_mode):
        if ensemble_mode == "none":
            return
        if ensemble_mode == "normal":
            ensemble_layer_count = sum(range(2, len(self.exit_layers) + 1))
            for _ in range(ensemble_layer_count):
                self.normal_ensemble.append(NormalEnsembleLayer(category))
            return
        for num_sources in range(1, len(self.exit_layers) + 1):
            self.geometric_ensemble.append(GeometricEnsembleLayer(num_sources, category))

    def forward(self, x):
        outputs = []
        raw_outputs = []
        exit_index = 0
        ensemble_index = 0

        for layer_index, backbone in enumerate(self.backbone, start=1):
            x = backbone(x)
            if layer_index not in self.exit_layers:
                continue

            current = self.exits[exit_index](x)
            raw_outputs.append(current)
            if self.ensemble_mode == "none":
                output = current
            elif self.ensemble_mode == "ensemble":
                ensemble_inputs = []
                for raw_output in raw_outputs[:-1]:
                    ensemble_inputs.append(raw_output.detach() if self._is_train_forward() else raw_output)
                ensemble_inputs.append(current)
                output = self.geometric_ensemble[exit_index](ensemble_inputs)
            elif exit_index == 0:
                output = current
            else:
                output = 0
                for previous_output in outputs:
                    if self._is_train_forward():
                        previous_output = previous_output.detach()
                    output = output + self.normal_ensemble[ensemble_index](previous_output)
                    ensemble_index += 1
                output = output + self.normal_ensemble[ensemble_index](current)
                ensemble_index += 1

            outputs.append(output)

            if not self._is_train_forward() and exit_index < len(self.exit_layers) - 1:
                threshold = getattr(self, f"exit_threshold{exit_index + 1}")
                if self.should_exit(output, threshold):
                    return outputs, exit_index + 1

            exit_index += 1

        return outputs if self._is_train_forward() else (outputs, 0)

    def _is_train_forward(self):
        return self.training

    def should_exit(self, logits, threshold):
        return self.exit_confidence_based(logits, threshold)

    @staticmethod
    def exit_confidence_based(logits, threshold):
        if logits.size(0) != 1:
            raise ValueError("Confidence-based early exit expects test batch size 1.")
        prob = F.softmax(logits, dim=-1)
        max_prob = prob.max(dim=1).values.item()
        return max_prob >= float(threshold)
