from argparse import ArgumentParser
import torch
import numpy as np


def parse_args():
    """
    Argument parser for handling some of the configs outside the yaml file
    """
    parser = ArgumentParser(description="Image segmentation models in pytorch")
    parser.add_argument('--no-cuda', action='store_true', default=False, 
                        help='disable cuda')
    parser.add_argument('--debug', action='store_true', default=True, 
                        help='To plot data and info straight after training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--num_gpus', type=int, default=2, metavar='S',
                        help='Number of gpus to run on (default: 2)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='for Saving the current Model')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='for Loading the best model')
    return parser.parse_args()


def seed_all(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


