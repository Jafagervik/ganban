import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, latent_shape, output_shape):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_shape, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, output_shape),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        x = x.view(x.size(0), 1, 64, 64) # fix later
        return x
