import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models.wgan_gp as gan

manual_seed = 999
num_epochs = 3
data_dir = "data/"
batch_size = 64
lr = 0.01
betas = (0.5, 0.999)
latent_shape = 100
image_size = 64
image_channels = 1
num_critic = 5
lambda_gp = 10
print_itr = 250

def gradient_penalty(dsc, real, fake, device):
    b_size, chn, h, w = real.shape
    alpha = torch.rand((b_size, 1, 1, 1)).repeat(1, chn, w, h).to(device)
    interpolated = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)
    dsc_scores = dsc(interpolated)

    grad = torch.autograd.grad(
        inputs=interpolated,
        outputs=dsc_scores,
        grad_outputs=torch.ones_like(dsc_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.view(grad.shape[0], -1)
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()

def main():
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    #torch.use_deterministic_algorithms(True)

    dataset = datasets.MNIST(
        root=data_dir,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]),
        download=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    image_shape = image_channels * image_size * image_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = gan.Generator(latent_shape, image_shape).to(device)
    discriminator = gan.Discriminator(image_shape).to(device)

    opt_gen = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    opt_dsc = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for batch_index, (data, _) in enumerate(dataloader, 0):
            b_size = data.size(0)
            real = data.to(device)

            opt_dsc.zero_grad()
            noise = torch.randn(b_size, latent_shape, 1, 1, device=device)
            fake = generator(noise)

            real_pred = discriminator(real)
            fake_pred = discriminator(fake)

            grad_pen = gradient_penalty(discriminator, real, fake, device)
            loss_dsc = -torch.mean(real_pred) + torch.mean(fake_pred) + lambda_gp * grad_pen

            loss_dsc.backward()
            opt_dsc.step()

            opt_gen.zero_grad()
            if batch_index % num_critic == 0:
                #noise = torch.randn(b_size, latent_shape, 1, 1, device=device)
                fake = generator(noise)
                fake_pred = discriminator(fake)
                loss_gen = -torch.mean(fake_pred)

                loss_gen.backward()
                opt_gen.step()

                if batch_index % print_itr == 0:
                    print(
                        '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                     % (epoch, num_epochs - 1, batch_index, len(dataloader), loss_dsc.item(), loss_gen.item())
                    )

if __name__ == "__main__":
    main()
