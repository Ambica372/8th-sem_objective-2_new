import numpy as np

def feature_level_fusion(eeg_matrix, eye_matrix):
    """
    Early Fusion: Concatenates modalities at the feature representation level.
    """
    print(f"[FUSION] Feature Level Fusion Executed.")
    return np.concatenate([eeg_matrix, eye_matrix], axis=1)
