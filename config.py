import os
# =============================================
#            HYPERPARAMS
# =============================================
BATCH_SIZE = 2
TEST_BATCH_SIZE = 8
IMG_SIZE = 256
NUM_CHANNELS = 3
LEARNING_RATE = 0.0002
EPOCHS = 200
EPOCH_DECAY_START = 100
BETA1 = 0.5
BETA2 = 0.999
LAMBDA1 = 10.0
LAMBDA2 = 2.5

# Directories
DATA_DIR = "data"
DATASET = "damaged2fixedcolor"
DATASET_PATH = os.path.join(DATA_DIR, DATASET)
