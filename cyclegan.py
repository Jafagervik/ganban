import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class ResidualBlock(nn.Module):
    def __init__(self, features: int, kernel_size: int) -> None:
        super().__init__()
        self.reflectpad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(features, features, kernel_size)
        self.instnorm = nn.InstanceNorm2d(features)
        self.relu = nn.ReLU(inplace=True)

    def block(self, x):
        x = self.reflectpad(x)
        x = self.conv(x)
        x = self.instnorm(x)

        x = self.relu(x)

        x = self.reflectpad(x)
        x = self.conv(x)
        x = self.instnorm(x)

        return x

    def forward(self, x: torch.Tensor):
        return x + self.block(x)

class SamplingBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int, stride: int, padding: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding)
        self.instnorm = nn.InstanceNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.instnorm(x)
        x = self.relu(x)
        return x

class Generator(nn.Module):
    def __init__(self, input_shape: int) -> None:
        super().__init__()

        self.model = nn.Sequential(
            # Encoder
            nn.ReflectionPad2d(input_shape),
            nn.Conv2d(input_shape, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            SamplingBlock(64, 128, 3, 2, 1),
            SamplingBlock(128, 256, 3, 2, 1),

            # Transformer/9 blocks
            ResidualBlock(256, 3),
            ResidualBlock(256, 3),
            ResidualBlock(256, 3),
            ResidualBlock(256, 3),
            ResidualBlock(256, 3),
            ResidualBlock(256, 3),
            ResidualBlock(256, 3),
            ResidualBlock(256, 3),
            ResidualBlock(256, 3),

            # Decoder/transpose conv
            nn.Upsample(scale_factor=2),
            SamplingBlock(256, 128, 3, 1, 1),
            nn.Upsample(scale_factor=2),
            SamplingBlock(128, 64, 3, 1, 1),

            nn.ReflectionPad2d(input_shape),
            nn.Conv2d(64, input_shape, kernel_size=7),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)

class DiscBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int, stride: int, padding: int, norm=True) -> None:
        super().__init__()
        self.norm = norm
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding)
        self.instnorm = nn.InstanceNorm2d(out_features)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        if self.norm:
            x = self.instnorm(x)
        x = self.leakyrelu(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_shape: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            DiscBlock(input_shape, 64, 4, 2, 1, norm=False),
            DiscBlock(64, 128, 4, 2, 1),
            DiscBlock(128, 256, 4, 2, 1),
            DiscBlock(256, 512, 4, 2, 1),
            nn.ZeroPad2d((1, 0, 1, 0)), # 4,1,16,16 // 4,1,15,15
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)
