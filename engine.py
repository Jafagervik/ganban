"""
engine.py basically defines what a train and a test step is
"""
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from typing import Tuple

from utils import helpers
from config import BEST_CHCKPT_PATH

def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Performs one epoch worth of training

    Returns:
        train loss and accuracy
    """

    model.train()  # First of we set model in train mode

    train_loss, train_acc = 0, 0

    # Go through the batches in current epoch
    for batch_idx, (X, y) in dataloader:
        X = X.to(device)
        y = y.to(device)

        y_hat = model(X)

        loss = criterion(y_hat, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_hat, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_hat)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


@torch.inference_mode()
def test_step(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """ Performs one epoch worth of testing

    Returns:
        test loss and accuracy
    """

    model.eval()  # First of we set model in train mode

    test_loss, test_acc = 0, 0

    # Go through the batches in current epoch
    with torch.inference_mode():  # TODO: find out if this is needed anymore
        for _, (X, y) in dataloader:
            X = X.to(device)
            y = y.to(device)

            test_pred_logits = model(X)

            loss = criterion(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() /
                         len(test_pred_labels))

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss, test_acc


def validate_step():
    pass


def train(
    model: nn.Module,
    train_dataloader,
    test_dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: StepLR,
    epochs: int,
    device: torch.device
):
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    highest_acc = 0.0

    for epoch in tqdm(range(1, epochs + 1)):
        train_loss, train_acc = train_step(
            model, train_dataloader, criterion, optimizer, device)

        test_loss, test_acc = test_step(
            model, test_dataloader, criterion,  device)

        if test_acc > highest_acc:
            helpers.save_model(model=model,
                            target_dir="checkpoints",
                            model_name=BEST_CHCKPT_PATH)

        scheduler.step()

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
