import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import os
import yaml
from PIL import Image

from config import *
import cyclegan
import datasetup
import engine

def setup_dataset(root_dir):
    train_transform = transforms.Compose([
            transforms.Resize(256, Image.Resampling.BICUBIC),
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_transform = transforms.Compose([
            transforms.Resize(256, Image.Resampling.BICUBIC),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_set = datasetup.cycleGanDataset(root_dir, train=True, transform=train_transform)
    test_set = datasetup.cycleGanDataset(root_dir, train=False, transform=test_transform)

    return train_set, test_set

def main():
    # Init Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gen_AB = cyclegan.Generator(NUM_CHANNELS).to(device)
    gen_BA = cyclegan.Generator(NUM_CHANNELS).to(device)
    disc_A = cyclegan.Discriminator(NUM_CHANNELS).to(device)
    disc_B = cyclegan.Discriminator(NUM_CHANNELS).to(device)

    gen_AB.apply(cyclegan.weights_init)
    gen_BA.apply(cyclegan.weights_init)
    disc_A.apply(cyclegan.weights_init)
    disc_B.apply(cyclegan.weights_init)

    # Init datasets
    train_ds, test_ds = setup_dataset(DATASET_PATH)

    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        shuffle=True,
        batch_size = BATCH_SIZE,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_ds,
        shuffle=False,
        batch_size = TEST_BATCH_SIZE,
    )

    # Init optimizers
    optim_gen = optim.Adam(
        list(gen_AB.parameters()) + list(gen_BA.parameters()),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
    )
    optim_disc_A = optim.Adam(
        disc_A.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
    )
    optim_disc_B = optim.Adam(
        disc_B.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
    )

    # Setup criterion
    criterion_gan = torch.nn.MSELoss().to(device)
    criterion_cyc = torch.nn.L1Loss().to(device)
    criterion_idn = torch.nn.L1Loss().to(device)

    # Setup scheduling
    lr_l = lambda epoch: 1.0 if epoch < EPOCH_DECAY_START else max(0.0, 1.0 - (epoch - EPOCH_DECAY_START) / 100.0)
    sched_gen = optim.lr_scheduler.LambdaLR(optim_gen, lr_lambda=lr_l)
    sched_disc_A = optim.lr_scheduler.LambdaLR(optim_disc_A, lr_lambda=lr_l)
    sched_disc_B = optim.lr_scheduler.LambdaLR(optim_disc_B, lr_lambda=lr_l)

    # Setup tensorboard
    writer = SummaryWriter()

    engine.train(
        gen_AB=gen_AB,
        gen_BA=gen_BA,
        disc_A=disc_A,
        disc_B=disc_B,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion_gan=criterion_gan,
        criterion_cyc=criterion_cyc,
        criterion_idn=criterion_idn,
        optimizer_gen=optim_gen,
        optimizer_disc_A=optim_disc_A,
        optimizer_disc_B=optim_disc_B,
        scheduler_gen=sched_gen,
        scheduler_disc_A=sched_disc_A,
        scheduler_disc_B=sched_disc_B,
        device=device,
        writer=writer,
    )

    print("Finished training!")

if __name__ == "__main__":
    main()
