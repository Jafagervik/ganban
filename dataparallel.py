from __future__ import print_function

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import time
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm
from typing import List


def parse_args():
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
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='for Saving the current Model')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='for Loading the current Model')
    return parser.parse_args()


# def plotting(class_names, data, samples: int = 16):
#     fig = plt.figure(figsize=(9, 9))
#     n = np.sqrt(samples)
#     rows, cols = n, n 
#
#     for i in range(1, rows * cols + 1):
#         random_idx = torch.randint(0, len(train_data), size=[1]).item()
#         img, label = train_data[random_idx]
#         fig.add_subplot(rows, cols, i)
#         plt.imshow(img.squeeze(), cmap="gray")
#         plt.title(class_names[label])
#         plt.axis(False);


class ConvBlock(nn.Module):
    """
    https://learnopencv.com/understanding-convolutional-neural-networks-cnn/

    This is a more concise version of this:
    https://github.com/chongwar/vgg16-pytorch/blob/master/vgg16.py


    Utilizing the power of reuse, we can easily set up a general block and 
    just reuse this many times
    """
    def __init__(self, in_channels: int, out_channels: int, convs: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        # Batch_norm is here the same as out_channels, so we don't need to add an extra parameter
        # See the github link for explenations
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.a = nn.ReLU(True)
        self.mp = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.convs = convs

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.a(x)

        # Some layers do 2 convolutions first, some do 3
        for _ in range(self.convs):
            x = self.conv2(x)
            x = self.batch_norm(x)
            x = self.a(x)

        x = self.mp(x)

        return x


class ClassifierLayer(nn.Module):
    """
    Classifier layer. this is actually redundant due to how little it does,
    but it shows how you don't need to keep all nn info 
    in a single class
    """

    def __init__(self, inp: int, out: int):
        super().__init__()
        self.dense = nn.Linear(inp, out) # DENSE LAYER is called Linear in pytorch
        self.act = nn.ReLU(True)
        self.drop = nn.Dropout(p=0.65)

    def forward(self, x):
        x = self.dense(x)
        x = self.act(x)
        x = self.drop(x)

        return x


class VGG16(nn.Module):
    def __init__(self, channels: List[int]):
        super().__init__()

        assert(len(channels) == 9), "Dimension of channels does not match VGG16 net"

        # conv_layers = [1, 64, 128, 256, 512, 512, 4096, 4096, 10]

        self.conv1 = ConvBlock(in_channels=channels[0], out_channels=channels[1], convs=2)
        self.conv2 = ConvBlock(in_channels=channels[1], out_channels=channels[2], convs=2)

        self.conv3 = ConvBlock(in_channels=channels[2], out_channels=channels[3], convs=3)
        self.conv4 = ConvBlock(in_channels=channels[3], out_channels=channels[4], convs=3)
        self.conv5 = ConvBlock(in_channels=channels[4], out_channels=channels[5], convs=3)

        self.classifier = nn.Sequential(
            ClassifierLayer(channels[5], channels[6]),
            ClassifierLayer(channels[6], channels[7]), 
            nn.Linear(channels[7], channels[-1]), # DENSE LAYER
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.detach().zero_()

        
    def forward(self, x):
        # FEATURE EXTRACTOR 
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Flatten
        x = x.view(x.size(0), -1) 
        
        # Classifier
        x = self.classifier(x)
        
        # Softmax before output
        x = F.softmax(x, dim=1) 

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
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #                100. * batch_idx / len(train_loader), loss.item()))
        #     if args.dry_run:
        #         break


@torch.no_grad()
def test(args, model, device, test_loader, highest_acc: float):
    model.eval()
    test_loss = 0
    correct = 0
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

    if acc > highest_acc:
        highest_acc = acc 
        torch.save(model.state_dict(), f"best_model.pt")


    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return highest_acc


def get_transforms():
    """Set up some transforms. These 3 are fairly common"""
    return transforms.Compose([
        transforms.Resize((32,32)), # BAD
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def main():
    # Training settings
    args = parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"{device=}")

    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i))

    transforms = get_transforms()

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=True, download=True, transform=transforms),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=False, transform=transforms),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # ==========================================================================
    #
    #       Dataparalllel 
    #
    # ==========================================================================
    model = VGG16(channels=[1, 64, 128, 256, 512, 512, 4096, 4096, 10])

    model = nn.DataParallel(model)

    compiled = torch.compile(model)

    if args.load_model:
        compiled.load_state_dict(torch.load("best_model.pt"))

    compiled.to(device)

    optimizer = optim.Adadelta(compiled.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    highest_acc = 0.0

    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train(args, compiled, device, train_loader, optimizer, epoch)
        highest_acc = test(args, compiled, device, test_loader, highest_acc)
        scheduler.step()

    end = time.time()
    elapsed = end - start

    if args.save_model:
        torch.save(compiled.state_dict(), "mnist_vgg16.pt")

    print(f"Elapsed time: {elapsed} seconds")


if __name__ == '__main__':
    main()
