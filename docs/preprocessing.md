# Preprocessing

## Steps

1. Raw EEG and Eye-tracking data loaded
2. Feature extraction performed
3. Modalities fused into a single feature vector (58 features)

## Critical Fix

Corrupted samples detected:
- NaN values
- Infinite values

Removed 26 samples before training.

## Final Dataset

- Shape: (37549, 58)
- Cleaned and scaled using StandardScaler
