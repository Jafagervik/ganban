from pathlib import Path
import torch
from torch import nn
from torchvision import transforms
from argparse import ArgumentParser


def argsparser():
    parser = ArgumentParser(prog="SugmaNet", description="no", epilog="byebye")
    parser.add_argument("model")
    parser.add_argument("batch_size")
    parser.add_argument("lr")
    parser.add_argument("epochs")


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
