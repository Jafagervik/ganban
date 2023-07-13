from pathlib import Path
import torch
from torch import nn
from torchvision import transforms
from argparse import ArgumentParser


def parse_args():
    """
    Argument parsing in cuda
    """
    parser = ArgumentParser(description='PyTorch MNIST Example using RNN')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='learning rate step gamma (default: 0.7)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='for Saving the current Model')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='for Loading the best model')
    return parser.parse_args()


def print_number_of_parameters(net: nn.Module):
    n_parameters = sum(p.numel() for p in net.parameters())
    print(f"number of params (M): {(n_parameters / 1.e6):2f}")


def setup_transforms():
    """
    Setup transforms on images
    """
    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        # TODO: Should we even normalize?
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.ToTensor()
    ])

    return data_transform


def random_seed_all(rand_seed: int):
    """Randomize all seeds

    Args:
        rand_seed (int): seed to use
    """
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)


def load_model(net: nn.Module, path: str) -> nn.Module:
    """
    Loads pretrained weights back into a model

    Args:
        net: Model
        path: full path to .pt or .pth file
    """
    net.load_state_dict(torch.load(path))
    return net

def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time

def calc_accuracy(y_true, y_pred):
    """Calculates accuracy

    Args:
        y_true: Ground Truth
        y_pred: Predictions

    Returns:
        Accuracy value between y_true and y_pred
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc