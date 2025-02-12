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

class MobileNetV2ECA(nn.Module):
    def __init__(self, num_classes=4, width_mult=1.0):
        super().__init__()
        block = InvertedResidual
        input_channel = 24
        last_channel = 640
        
        # First layer
        input_channel = _make_divisible(input_channel * width_mult, 8)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), 8)
        
        # Define the inverted residual blocks configuration
        self.features = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
            
            block(input_channel, 16, 1, 1, True),  # 1
            
            block(16, 24, 2, 6),  # 2
            block(24, 24, 1, 6, True),  # 3
            
            block(24, 32, 2, 6),  # 4
            block(32, 32, 1, 6, True),  # 5
            block(32, 32, 1, 6, True),  # 6
            
            block(32, 64, 2, 6),  # 7
            block(64, 64, 1, 6, True),  # 8
            block(64, 64, 1, 6, True),  # 9
            block(64, 64, 1, 6, True),  # 10
            
            block(64, 96, 1, 6, True),  # 11
            block(96, 96, 1, 6, True),  # 12
            block(96, 96, 1, 6, True),  # 13
            
            block(96, 160, 2, 6),  # 14
            block(160, 160, 1, 6, True),  # 15
            block(160, 160, 1, 6, True),  # 16
            
            block(160, 320, 1, 6, True),  # 17
            
            nn.Conv2d(320, self.last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # Global average pooling
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


class MobileNetV2ECAn2(nn.Module):
    def __init__(self, num_classes=4, width_mult=1.0):
        super().__init__()
        block = InvertedResidual
        input_channel = 24
        last_channel = 640
        
        # First layer
        input_channel = _make_divisible(input_channel * width_mult, 8)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), 8)
        
        # Define the inverted residual blocks configuration
        self.features = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
            
            block(input_channel, 16, 1, 1, True),  # 1
            
            block(16, 24, 2, 6),  # 2
            
            block(24, 32, 2, 6),  # 4

            
            block(32, 64, 2, 6),  # 7
            block(64, 64, 1, 6, True),  # 8

            
            block(64, 96, 1, 6, True),  # 11
            
            block(96, 160, 2, 6),  # 14

            
            block(160, 320, 1, 6, True),  # 17
            
            nn.Conv2d(320, self.last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # Global average pooling
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

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=4, width_mult=1.0):
        super().__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        # First layer
        input_channel = _make_divisible(input_channel * width_mult, 8)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), 8)

        # Define the inverted residual blocks configuration
        self.features = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),

            block(input_channel, 16, 1, 1),  # 1

            block(16, 24, 2, 6),  # 2
            block(24, 24, 1, 6),  # 3

            block(24, 32, 2, 6),  # 4
            block(32, 32, 1, 6),  # 5
            block(32, 32, 1, 6),  # 6

            block(32, 64, 2, 6),  # 7
            block(64, 64, 1, 6),  # 8
            block(64, 64, 1, 6),  # 9
            block(64, 64, 1, 6),  # 10

            block(64, 96, 1, 6),  # 11
            block(96, 96, 1, 6),  # 12
            block(96, 96, 1, 6),  # 13

            block(96, 160, 2, 6),  # 14
            block(160, 160, 1, 6),  # 15
            block(160, 160, 1, 6),  # 16

            block(160, 320, 1, 6),  # 17

            nn.Conv2d(320, self.last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # Global average pooling
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
    # Create model instance
    model = MobileNetV2(num_classes=4)
    
    # Test input
    batch_size = 2
    channels = 3
    height = 224
    width = 224
    
    # Create random input tensor
    x = torch.randn(batch_size, channels, height, width)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model architecture:\n{model}")

if __name__ == "__main__":
    main()