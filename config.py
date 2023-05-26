import torch
import os

"""Hyperparameters"""

# Normal parameters
NUM_WORKERS = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================
#            HYPERPARAMS
# =============================================
BATCH_SIZE = 32
RANDOM_SEED = 42
EPOCHS = 10
HIDDEN_UNITS = 10
LR = 1e-3
IMG_SIZE = 28
NUM_CHANNELS = 3
MOMENTUM = 1e-3
DROPOUT_RATE = 0.2
SAVE_MODEL = True
LOAD_MODEL = False

TRAIN_RATIO = 0.65
VALIDATE_RATIO = 0.15
TEST_RATIO = 0.2
GAMMA = 0.7

# Directories
TRAIN_DIR = "data/anime/train"
TEST_DIR = "data/anime/test"
