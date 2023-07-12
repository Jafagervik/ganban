import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import os
import yaml
from PIL import Image

from models.cyclegan import *
import engine
from helpers import datasetup, utils

# i will put these in config later...
batch_size = 2 # lots of vram...
input_shape = (3, 256, 256)
c, w, h = input_shape

def setup_dataset(root_dir):
    train_transform = transforms.Compose([
            transforms.Resize((w, h), Image.Resampling.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_transform = transforms.Compose([
            transforms.Resize((w, h), Image.Resampling.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_set = datasetup.cycleGanDataset(root_dir, train=True, transform=train_transform)
    test_set = datasetup.cycleGanDataset(root_dir, train=False, transform=test_transform)

    return train_set, test_set

def main():
    """
    Setup world and rank to distribute dataset over multiple gpus
    """
    # Parse cml args
    args = utils.parse_args()

    # Seed all rngs
    # utils.seed_all(args.seed)

    # Open config flie
    with open("configs/config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    # Init datasets
    root_dir = os.path.join(config['data_path'], config['data_set'])
    train_ds, test_ds = setup_dataset(root_dir)

    # Check whether or not to run on cuda
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if not use_cuda:
        # TODO: Serial setup
        print("Setup serial run")

        #serial(args, config, train_ds, test_ds)

        print("Finished serial training!")
        return

    if args.debug:
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_properties(i))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gen_AB = Generator(c).to(device)
    gen_BA = Generator(c).to(device)
    disc_A = Discriminator(c).to(device)
    disc_B = Discriminator(c).to(device)

    gen_AB.apply(weights_init)
    gen_BA.apply(weights_init)
    disc_A.apply(weights_init)
    disc_B.apply(weights_init)

    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        shuffle=True,
        batch_size = config['batch_size'],
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_ds,
        shuffle=False,
        batch_size = 6,
    )

    optim_gen = optim.Adam(
        list(gen_AB.parameters()) + list(gen_BA.parameters()),
        lr=float(config['learning_rate']),
        betas=(float(config['beta1']), float(config['beta2'])),
    )
    optim_disc_A = optim.Adam(
        disc_A.parameters(),
        lr=float(config['learning_rate']),
        betas=(float(config['beta1']), float(config['beta2'])),
    )
    optim_disc_B = optim.Adam(
        disc_B.parameters(),
        lr=float(config['learning_rate']),
        betas=(float(config['beta1']), float(config['beta2'])),
    )

    criterion_gan = torch.nn.MSELoss().to(device)
    criterion_cyc = torch.nn.L1Loss().to(device)
    criterion_idn = torch.nn.L1Loss().to(device)

    lr_l = lambda epoch: 1.0 if epoch < 100 else max(0.0, 1.0 - (epoch - 100) / 100.0)
    sched_gen = optim.lr_scheduler.LambdaLR(optim_gen, lr_lambda=lr_l)
    sched_disc_A = optim.lr_scheduler.LambdaLR(optim_disc_A, lr_lambda=lr_l)
    sched_disc_B = optim.lr_scheduler.LambdaLR(optim_disc_B, lr_lambda=lr_l)

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
        config=config,
        args=args,
        writer=writer,
    )

    print("Finished training!")


if __name__ == "__main__":
    main()
