from __future__ import print_function

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

# Print a statusbar in your training
from tqdm import tqdm

def seed_all(seed):
    """Random number generator needs seeding am i rite xd"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def parse_args():
    """How to parseargs 101"""
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example using RNN')
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
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
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


def setup_dataloaders(args, kwargs):
    """
    Set up some beatiful dataloaders 
    """
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader


class OmegaNet(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=64, batch_first=True)
        self.batchnorm = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # Shape of input is (batch_size,1, 28, 28)
        # converting shape of input to (batch_size, 28, 28)
        # as required by RNN when batch_first is set True
        x = x.reshape(-1, 28, 28)
        x, hidden = self.rnn(x)

        # RNN output shape is (seq_len, batch, input_size)
        # Get last output of RNN
        x = x[:, -1, :]
        x = self.batchnorm(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(args, model, device, test_loader, highest_acc):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.inference_mode():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            if args.dry_run:
                break

    test_loss /= len(test_loader.dataset)

    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))

    if acc > highest_acc:
        highest_acc = acc
        torch.save(model.state_dict(), "best_model.pt")

    return highest_acc


def main():
    # Training settings
    args = parse_args() 
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    seed_all(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader, test_loader = setup_dataloaders(args, kwargs)

    # Can be stored as a hyperparam in a config file or something
    input_size = 28

    model = OmegaNet(input_size).to(device)
    # model = torch.compile(model) New in torch 2.0!

    # Transfer learning
    if args.load_model:
        model.load_state_dict(torch.load("checkpoints/best_model.pt"))

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # This is a bit ugly, and frankly an antipattern in torch, but it works
    highest_acc = 0.0

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in tqdm(range(1, args.epochs + 1)):
        train(args, model, device, train_loader, optimizer, epoch)
        highest_acc = test(args, model, device, test_loader, highest_acc)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "checkpoints/mnist_rnn.pt")


if __name__ == '__main__':
    main()
