"""
Entry point for training
"""

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
import datasetup
import engine
from model import Net
from utils import helpers

from config import (BATCH_SIZE, EPOCHS, HIDDEN_UNITS,
                    IMG_SIZE, DEVICE, LR, NUM_CHANNELS, RANDOM_SEED, TRAIN_DIR,
                    TEST_DIR, SAVE_MODEL, LOAD_MODEL, GAMMA)


def main():
    """
    ENTRY POINT
    """
    args = helpers.argsparser()
    helpers.random_seed_all(RANDOM_SEED)

    # TODO: Parse args and use them while training

    # Create transforms
    data_transform = helpers.setup_transforms()

    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader, class_names = datasetup.create_dataloaders(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )

    # Create model with help from model_builder.py
    model = Net(
        input_shape=NUM_CHANNELS,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    )


    compiled = torch.compile(model)
    # Parallel model in case of multiple gpus
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(DEVICE)

    if LOAD_MODEL:
        model = helpers.load_model(model, "path")

    # Set up loss function, optimizer and learning rate scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LR)
    scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA)

    # Start training with help from engine.py
    engine.train(model=model,
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 criterion=criterion,
                 optimizer=optimizer,
                 scheduler=scheduler,
                 epochs=EPOCHS,
                 device=DEVICE)

    helpers.print_number_of_parameters(model)

    # Save the model with help from utils.py
    if SAVE_MODEL:
        helpers.save_model(model=model,
                           target_dir="models",
                           model_name="somewhere.pth")


if __name__ == "__main__":
    main()
