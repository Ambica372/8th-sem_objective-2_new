import os

# Dataset path
# Accounting for Windows OneDrive Desktop architecture
BASE_PATH = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "8th_sem_obj_2")

EEG_ZIP = os.path.join(BASE_PATH, "eeg_feature_smooth.zip")
EYE_ZIP = os.path.join(BASE_PATH, "eye_feature_smooth.zip")

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50

# Validation setup
TEST_SPLIT_SIZE = 0.2
CROSS_VALIDATION_FOLDS = 5

# PCA
PCA_VARIANCE = 0.95

# Random seed
RANDOM_STATE = 42
