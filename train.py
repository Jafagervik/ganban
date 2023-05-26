"""
Entry point for training
"""

import torch
import datasetup
import engine
from model import Net
from utils import helpers

from config import (BATCH_SIZE, EPOCHS, HIDDEN_UNITS,
                    IMG_SIZE, DEVICE, LR, NUM_CHANNELS, TRAIN_DIR, TEST_DIR)


def main():
    """
    ENTRY POINT
    """
    args = helpers.argsparser()

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
    ).to(DEVICE)

    # Set loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LR)

    # Start training with help from engine.py
    engine.train(model=model,
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 criterion=criterion,
                 optimizer=optimizer,
                 epochs=EPOCHS,
                 device=DEVICE)

    # Save the model with help from utils.py
    helpers.save_model(model=model,
                       target_dir="models",
                       model_name="05_going_modular_script_mode_tinyvgg_model.pth")


if __name__ == "__main__":
    main()
