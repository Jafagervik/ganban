import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int, stride: int, padding: int) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_features, out_features,
                              kernel_size, stride, padding)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.act(x)
        x = self.mp(x)
        x = self.dropout(x)

        return x


class Net(nn.Module):
    """Insane homemade network

    Never ever have such an insane net been created
    """

    def __init__(
        self,
        input_shape: int,
        hidden_units: int,
        output_shape: int
    ) -> None:
        super().__init__()

        self.b1 = Block(input_shape, 10, 3, 1, 1)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x = self.b1(x)
        x = self.act(x)
        return x
