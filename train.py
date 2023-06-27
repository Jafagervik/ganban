"""Main file for running and setting up training loop"""
import torch
import torch.nn as nn
from torch import optim
from helpers.datasetup import init_datasets, init_dataloaders
from models.unet import UNet
from models.toymod import ToyMpModel
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

import engine
from helpers import utils
import os
import yaml


# OS SETUP
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12356'
torch.set_float32_matmul_precision('high')
WORLD_SZ = torch.cuda.device_count() if torch.cuda.is_available() else 2


def run_process(rank, args, config, train_dataset, test_dataset):
    """
    This function is to be run on every single processor 

    Data is distributed evenly among processors, who all 
    distribute data to corresponding gpus

    Args:
        rank, int: id of processor
        args: struct: flags used when running script
        config, yaml dict: all hyperparams and setup info
    """
    # if WORLD_SZ > 1:
    dist.init_process_group("nccl", world_size=WORLD_SZ, rank=rank)

    # We here set the correct gpu to run on based on which rank we're in
    device = torch.device(f"cuda:{rank}")


    # model = UNet(channels=[1, 64, 128, 256, 512, 512, 4096, 4096, 10]).to(rank)
    model = ToyMpModel()
    compiled = torch.compile(model).to(rank)


    map_location = None
    # Wrap the model
    if WORLD_SZ > 2:
        compiled = DDP(compiled, device_ids=[rank])
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}

    # Transfer learning
    if args.load_model:
        compiled.load_state_dict(torch.load(config['checkpoint_best'], map_location=map_location if map_location else None))

    # Creates dataloaders based on ranks
    #train_dataloader, test_dataloader, class_names = create_dataloaders(config, world_size=WORLD_SZ, rank=rank)

    train_kwargs = {'batch_size': 32}
    test_kwargs = {'batch_size': 1000}
                   
    cuda_kwargs = {'num_workers': config['use_cpu'],
                    'pin_memory': config['pin_memory']}

    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_sampler, test_sampler = None, None

    if WORLD_SZ > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=WORLD_SZ,
            rank=rank,
            shuffle=config['shuffle']
        )

        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=WORLD_SZ,
            rank=rank,
            shuffle=config['shuffle']
        )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                   shuffle=False if train_sampler else True, 
                                                   sampler=train_sampler if train_sampler else None, 
                                                   **train_kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                    shuffle=False if test_sampler else True, 
                                                  sampler=test_sampler if test_sampler else None,
                                                  **test_kwargs)

    optimizer = optim.Adadelta(compiled.parameters(), lr=float(config['learning_rate']))
    criterion = nn.CrossEntropyLoss().cuda(rank)

    scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=float(config['gamma']))

    # writer = SummaryWriter() if args.debug else None
    writer = None


    print(f"Rank: {rank} | Device: {device=}")

    engine.train(
        model=compiled,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        args=args,
        rank=rank,
        writer=writer)

    if WORLD_SZ > 1:
        dist.barrier()
    dist.destroy_process_group()


def setup_dataset():
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Handle downloads in it's own folder
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)

    return dataset1, dataset2


def serial(args, config, train_dataset, test_dataset):
    """
    Serial implementation running on cpu 
    """
    # model = UNet(channels=[1, 64, 128, 256, 512, 512, 4096, 4096, 10]).to(rank)
    model = ToyMpModel()
    compiled = torch.compile(model)

    # Transfer learning
    if args.load_model:
        compiled.load_state_dict(torch.load(config['checkpoint_best']))

    # Creates dataloaders based on ranks
    #train_dataloader, test_dataloader, class_names = create_dataloaders(config, world_size=WORLD_SZ, rank=rank)

    train_kwargs = {'batch_size': 32}
    test_kwargs = {'batch_size': 1000}

    train_dataloader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    optimizer = optim.Adadelta(compiled.parameters(), lr=float(config['learning_rate']))
    criterion = nn.CrossEntropyLoss()

    scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=float(config['gamma']))

    # writer = SummaryWriter()
    writer = None

    # We here set the correct gpu to run on based on which rank we're in
    device = torch.device("cpu")

    engine.train(
        model=compiled,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        args=args,
        writer=writer)


def main():
    """
    Setup world and rank to distribute dataset over multiple gpus
    """
    # Parse cml args
    args = utils.parse_args()

    # Seed all rngs
    utils.seed_all(args.seed)
    
    # Open config flie
    with open("configs/config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    # Init datasets
    train_ds, test_ds = setup_dataset()

    # Check whether or not to run on cuda
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if not use_cuda:
        # TODO: Serial setup
        print("Setup serial run")

        serial(args, config, train_ds, test_ds)

        print("Finished serial training!")
        return

    if args.debug:
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_properties(i))

    arg_list = (args, config, train_ds, test_ds)

    mp.spawn(fn=run_process,
             nprocs=WORLD_SZ, 
             args=arg_list)

    print("Finished training!")


if __name__ == "__main__":
    main()
