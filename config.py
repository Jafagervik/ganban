import torch
import os


NUM_WORKERS = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu "
BATCH_SIZE = 32
RANDOM_SEED = 42
EPOCHS = 10
HIDDEN_UNITS = 10
LR = 1e-3  # Learning rate
IMG_SIZE = 28
NUM_CHANNELS = 3  # Number of color channels for an image
TRAIN_DIR = "data/pizza_steak_sushi/train"
TEST_DIR = "data/pizza_steak_sushi/test"
