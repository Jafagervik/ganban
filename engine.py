"""
engine.py basically defines what a train and a test step is
"""
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist

import time
from typing import Tuple
from helpers import graphs


def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer = None
) -> Tuple[float, float]:
    """Performs one epoch worth of training

    Returns:
        train loss and accuracy
    """
    model.train() 

    train_loss, train_acc = 0, 0

    # Go through the batches in current epoch
    for batch_idx, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        y_hat = model(X)

        loss = criterion(y_hat, y)
        train_loss += loss.item()

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_hat, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_hat)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)


    return train_loss, train_acc

# @torch.no_grad
# def val_step(
#     model: nn.Module, 
#     dataloader: DataLoader,
#     criterion: nn.Module,
#     device: torch.device,
#     epoch: int) -> Tuple[float, float]:
#     model.eval()
#
#     val_loss = 0.0
#     val_acc = 0.0
#     early_stop = False
#
#     for _, (X, y) in enumerate(dataloader):
#         X = X.to(device)
#         y = y.to(device)
#
#         val_pred_logits = model(X)
#
#         loss = criterion(val_pred_logits, y)
#         val_loss += loss.item()
#
#         val_pred_labels = val_pred_logits.argmax(dim=1)
#         val_acc += ((val_pred_labels == y).sum().item() /
#                         len(val_pred_labels))
#
#
#     val_loss /= len(dataloader)
#     val_acc /= len(dataloader)
#
#     return val_loss, val_acc, early_stop 
#

@torch.inference_mode()
def test_step(
    model: nn.Module,
    highest_acc: float,
    best_epoch: int,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    config
) -> Tuple[float, float, float, int]:
    """ Performs one epoch worth of testing

    Returns:
        test loss and accuracy as well as highest accuracy
    """
    model.eval()

    test_loss, test_acc = 0, 0
    
    # Go through the batches in current epoch
    for _, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        test_pred_logits = model(X)

        loss = criterion(test_pred_logits, y)
        test_loss += loss.item()

        test_pred_labels = test_pred_logits.argmax(dim=1)
        test_acc += ((test_pred_labels == y).sum().item() /
                        len(test_pred_labels))


        if test_acc > highest_acc:
            highest_acc = test_acc
            best_epoch = epoch
            torch.save(model.state_dict(), config['checkpoint_best'])
        # elif epoch - best_epoch > config['early_stop_tresh']:
        #     print(f"Early stopping at epoch: {epoch}")
        #     # TODO: Add validation set to prevent overfitting and move early stopping there
        #     early_stop = True
        #     break

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss, test_acc, highest_acc, best_epoch


def train(
    model,
    train_dataloader,
    test_dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: StepLR,
    device: torch.device,
    config,
    args,
    rank: int = 0,
    writer = None
):
    results = {
        "0": {
            "train_loss": [0.0],
            "train_acc": [0.0],
            "test_loss": [0.0],
            "test_acc": [0.0],
        },
        "1": {
            "train_loss": [0.0],
            "train_acc": [0.0],
            "test_loss": [0.0],
            "test_acc": [0.0],
        },
    }


    highest_acc = 0.0
    best_epoch = -1

    start = time.time() 

    for epoch in tqdm(range(1, config['epochs']+ 1)):
        train_loss, train_acc = train_step(
            model, train_dataloader, criterion, optimizer, device, epoch, writer if writer else None)

        # val_step()

        test_loss, test_acc, highest_acc, best_epoch = test_step(
            model, highest_acc, best_epoch, test_dataloader, criterion, device, epoch, config)

        scheduler.step()

        if args.dry_run:
            break

        print(
            f"Rank {rank} | "
            f"Epoch: {epoch} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["%d" % rank]["train_acc"].append(train_loss)
        results["%d" % rank]["train_loss"].append(train_loss)
        results["%d" % rank]["test_acc"].append(train_loss)
        results["%d" % rank]["test_loss"].append(train_loss)

        # if early_stop: 
        #     break


    if writer:
        writer.flush()

    end = time.time()

    if rank == 0:
        print(f"Training complete in {end - start} seconds")
        print(f"Best epoch was: {best_epoch}")

        # Save model to file if selected
        if args.save_model:
            torch.save(model.state_dict(), config['checkpoint_last'])

        graphs.plot_acc_loss(results)
    

