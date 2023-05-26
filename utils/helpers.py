from pathlib import Path
import torch
from torchvision import transforms
from argparse import ArgumentParser


def argsparser():
    parser = ArgumentParser(prog="SugmaNet", description="no", epilog="byebye")
    parser.add_argument("model")
    parser.add_argument("batch_size")
    parser.add_argument("lr")
    parser.add_argument("epochs")


def setup_transforms():
    """
    Setup transforms
    """
    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
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


def load_model(path: str) -> torch.nn.Module:
    pass
