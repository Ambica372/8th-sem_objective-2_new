# EEG + Eye-Tracking Emotion Classification (Window-Based)

## 📌 Overview

This project focuses on emotion classification using multimodal physiological signals:

- EEG (brain signals)
- Eye-tracking data

We use a *window-based approach (NO aggregation)* to preserve temporal information and improve learning stability.

---

## 🎯 Objective

- Build multiple deep learning models
- Compare their performance
- Identify the best architecture
- Ensure stable training on full dataset
- Handle real-world issues like corrupted data (NaN/Inf)

---

## 📂 Project Structure

8th_sem_new/ │ ├── stage1_data/                # Raw data ├── stage2_preprocessing/       # Cleaning + feature extraction ├── stage3_feature_analysis/    # Feature understanding ├── stage4_models/              # Model training │   ├── mlp/ │   ├── dnn/ │   ├── attention/ │   ├── hybrid/ │   ├── decision_fusion/ │   └── comparison/             # Final results │ ├── stage4_pipeline/            # Processed numpy arrays ├── docs/                       # Documentation └── run_models.py               # Main training script

---

## 🧠 Approach

### 1. Window-Based Training (IMPORTANT)

- Each sample = time window (NOT trial aggregation)
- Preserves temporal information
- Prevents loss of signal patterns

---

### 2. Data Pipeline

- EEG → PCA features
- Eye → cleaned features
- Combined → X_fused (58 features)

Final inputs:
- X_fused.npy
- X_eeg_pca.npy
- X_eye_clean.npy
- y.npy

---

### 3. Data Cleaning (CRITICAL FIX)

Problem:
- Full dataset training caused *NaN loss*
- Models collapsed

Root Cause:
- Corrupt data (NaN / Inf values)

Fix:
```python
bad_rows = np.isnan(...) | np.isinf(...)

Result:

Removed 26 corrupted samples

Dataset became stable



---

📊 Dataset Details

Metric	Value

Total samples (original)	37,575
Removed corrupted samples	26
Final samples used	37,549
Features per sample	58
Classes	4



---

⚙️ Models Implemented

1. MLP (Baseline)

2 hidden layers

BatchNorm + Dropout



---

2. Deep Neural Network (DNN)

Deeper architecture

Higher capacity learning



---

3. Attention Model

Learns feature importance

Applies attention weights



---

4. Hybrid Model (BEST)

Combines dense + attention

Strong feature interaction



---

5. Decision Fusion

Separate EEG + Eye models

Late fusion (averaging)



---

🚀 Training Details

Parameter	Value

Epochs	30
Batch Size	64
Learning Rate	1e-4
Loss	CrossEntropy
Split	80% Train / 20% Test



---

📈 Final Results

🏆 Best Model: Hybrid

Model	Accuracy

MLP	76.03%
DNN	90.35%
Attention	68.25%
Hybrid	92.84%
Decision Fusion	50.54%



---

📊 Key Observations

✅ What Worked

Data cleaning fixed NaN collapse

Feature scaling improved convergence

Hybrid model captured complex patterns

Window-based approach preserved signal structure



---

❌ What Failed

Decision Fusion performed poorly (~50%)

Attention alone was weaker than hybrid

Raw full dataset without cleaning caused total failure



---

⚠️ Critical Issue Encountered

Problem

NaN loss detected at epoch 1

Impact

Models stopped learning

Accuracy dropped to random (~25%)


Solution

Removed corrupted samples

Applied scaling


Result

Training stabilized → Accuracy jumped to 92.84%


---

📉 Why Hybrid Model Won

Combines:

Dense learning (global patterns)

Attention (feature importance)


Better representation of multimodal data

Balanced predictions across classes



---

📊 Outputs Generated

Each model folder contains:

accuracy.txt

classification_report.txt

confusion_matrix.png


Comparison folder:

model_comparison.csv



---

📌 Conclusion

Window-based training is stable and effective

Data quality is critical

Hybrid model provides best performance

Proper preprocessing > complex models



---

🔮 Future Work

Hyperparameter tuning

Advanced fusion methods

Transformer-based models

Real-time prediction system



---
