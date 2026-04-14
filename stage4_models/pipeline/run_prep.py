import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data
from preprocessing import preprocess_eeg, preprocess_eye
from fusion import feature_level_fusion

# Exact path mappings per requirements
ROOT = r"c:\Users\swamy\OneDrive\Desktop\8th_sem_new\stage4_pipeline"
OUT_DIR = os.path.join(ROOT, "processed_data")
REP_DIR = os.path.join(ROOT, "reports")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REP_DIR, exist_ok=True)

def run():
    print("Initiating Locked Preprocessing Pipeline...")
    X_eeg, X_eye, y = load_data()
    
    print("\nProcessing EEG Features...")
    X_eeg_pca, pca_model, eeg_scaler = preprocess_eeg(X_eeg)
    
    print("Processing Eye Features...")
    X_eye_clean, eye_scaler = preprocess_eye(X_eye)
    
    print("Executing Feature-Level Fusion...")
    X_fused = feature_level_fusion(X_eeg_pca, X_eye_clean)
    
    print("\n--- VALIDATION ---")
    print(f"EEG reduced shape: {X_eeg_pca.shape}")
    print(f"Eye cleaned shape: {X_eye_clean.shape}")
    print(f"Fused shape:      {X_fused.shape}")
    
    print("\nSaving Data Arrays...")
    np.save(os.path.join(OUT_DIR, "X_eeg_pca.npy"), X_eeg_pca)
    np.save(os.path.join(OUT_DIR, "X_eye_clean.npy"), X_eye_clean)
    np.save(os.path.join(OUT_DIR, "X_fused.npy"), X_fused)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)
    
    print("Generating Academic Report...")
    report_content = f"""# Preprocessing and Fusion Report

## 1. Feature Preprocessing

### EEG Modality (Differential Entropy)
* **Initial dimensions:** {X_eeg.shape[1]} features.
* **Standardization:** Applied Global `StandardScaler`.
* **Reduction Strategy:** Principal Component Analysis (PCA) engineered to retain explicitly 95% total variance.
* **Final EEG features:** {X_eeg_pca.shape[1]} features.

### Eye Modality
* **Initial dimensions:** {X_eeg.shape[1] if hasattr(X_eeg, 'shape') else 0} -- Wait dynamically generating:
* Initial dimensions: {X_eye.shape[1]} features.
* **Filtering:** Removed duplicate tracking nodes at explicitly stated target index 5 and 9.
* **Standardization:** Applied Global `StandardScaler`.
* **Final Eye features:** {X_eye_clean.shape[1]} features.

## 2. Feature Fusion
* **Mechanism:** Feature-level vector concatenation (`X_fused = concatenate(X_eeg_pca, X_eye_processed)`).
* **Dimensional Alignment:** Chronological trial concatenation is natively preserved. Random-shuffling logic via train/test boundaries across subject trials acts purely post-fusion sequentially.
* **Final Unified Dimension:** {X_fused.shape[1]} integrated features.

*This static dimensionality ensures completely balanced evaluations across MLP, DNN, Attention, and Hybrid architectures.*
"""
    with open(os.path.join(REP_DIR, "preprocessing_report.md"), "w") as f:
        f.write(report_content)
        
    print("Preprocessing operations completely locked and written.")

if __name__ == "__main__":
    run()
