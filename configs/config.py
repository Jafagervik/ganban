import os

"""Hyperparameters"""

# Normal parameters
NUM_WORKERS = os.cpu_count() or 4

# =============================================
#            HYPERPARAMS
# =============================================
BATCH_SIZE = 32
RANDOM_SEED = 42
HIDDEN_UNITS = 10
IMG_SIZE = 28
NUM_CHANNELS = 3
MOMENTUM = 1e-3
DROPOUT_RATE = 0.2

TRAIN_RATIO = 0.65
VALIDATE_RATIO = 0.15
TEST_RATIO = 0.2

# Directories
TRAIN_DIR = "data/audio/train"
BEST_CHCKPT_PATH = "ganban_best_model.pt"
TEST_DIR = "data/audio/test"
