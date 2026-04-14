# Training

## Configuration

- Epochs: 30
- Batch size: 64
- Learning rate: 1e-4

## Process

1. Train/test split (80/20)
2. Stratified sampling
3. Standard scaling applied

## Stability Fix

- Gradient clipping applied
- NaN detection added
- Training stops if corruption detected
