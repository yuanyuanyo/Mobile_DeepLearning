import torch
import torch.nn as nn
import math

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs(math.log2(channels) + b) / gamma)
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_eca=False):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])

        self.conv = nn.Sequential(*layers)
        self.use_eca = use_eca
        if use_eca:
            self.eca = ECALayer(oup)

    def forward(self, x):
        out = self.conv(x)
        if self.use_eca:
            out = self.eca(out)
        if self.use_res_connect:
            return x + out
        return out

class MobileNetV2Student(nn.Module):
    def __init__(self, num_classes=4, width_mult=0.75):  
        super().__init__()
        block = InvertedResidual
        input_channel = 24  
        last_channel = 640  
        
        input_channel = _make_divisible(input_channel * width_mult, 8)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), 8)
        
        self.features = nn.ModuleDict({
            'initial_conv': nn.Sequential(
                nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                nn.BatchNorm2d(input_channel),
                nn.ReLU6(inplace=True)
            ),
            'stage1_block1': block(input_channel, 16, 1, 1, True),

            'stage2_block1': block(16, 24, 2, 6),

            'stage3_block1': block(24, 32, 2, 6),

            'stage4_block1': block(32, 64, 2, 6),
            'stage4_block2': block(64, 64, 1, 6, True),

            'stage5_block1': block(64, 96, 1, 6, True),

            'stage6_block1': block(96, 160, 2, 6),

            'stage7_block1': block(160, 320, 1, 6, True),

            'final_conv': nn.Sequential(
                nn.Conv2d(320, self.last_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.last_channel),
                nn.ReLU6(inplace=True)
            )
        })

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes)
        )

        self._initialize_weights()

    def configure_pruning(self, prune_config):
        
        for block_name, config in prune_config.items():
            target_block = self.features[block_name]
            config_list = [{
                'op_names': config['conv_layers'],
                'sparsity': config['ratio'],
                'op_types': ['Conv2d']
            }]
            pruner = L1NormPruner(target_block, config_list)
            pruner.compress()
    
    def forward(self, x):
        for name, layer in self.features.items():
            x = layer(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

def main():
    model = MobileNetV2Student(num_classes=4)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()