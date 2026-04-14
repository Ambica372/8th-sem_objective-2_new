---

# 🧠 EEG + Eye-Tracking Emotion Classification (Window-Based Deep Learning)

---

## 📌 Project Overview

This project performs *emotion classification using multimodal physiological signals*:

- EEG (brain signals)
- Eye-tracking data

We use a *window-based approach (NO trial aggregation)* to preserve temporal dynamics and improve model learning.

---

## 🎯 Objectives

- Build multiple deep learning models  
- Compare performance across architectures  
- Handle real-world data issues (NaN / Inf corruption)  
- Ensure stable full-dataset training  
- Identify the best-performing model  

---

## 📂 Project Structure

8th_sem_new/ │ ├── stage1_data/ │   └── Raw EEG and Eye-tracking data │ ├── stage2_preprocessing/ │   └── Data cleaning and feature extraction │ ├── stage3_feature_analysis/ │   └── Feature inspection and validation │ ├── stage4_pipeline/ │   └── processed_data/ │       ├── X_fused.npy │       ├── X_eeg_pca.npy │       ├── X_eye_clean.npy │       └── y.npy │ ├── stage4_models/ │   ├── mlp/ │   ├── dnn/ │   ├── attention/ │   ├── hybrid/ │   ├── decision_fusion/ │   └── comparison/ │       ├── model_comparison.csv │       └── performance_plot.png │ ├── docs/ ├── run_models.py └── README.md

---

## 🔁 Workflow

Raw Data → Preprocessing → Feature Extraction → Model Training → Evaluation → Comparison

---

## 🧠 Methodology

### 1. Window-Based Learning (KEY IDEA)

- Each sample = time window  
- No trial-level aggregation  
- Preserves temporal patterns  

---

### 2. Data Pipeline

| Component | Description |
|----------|------------|
| EEG | PCA-reduced features |
| Eye | Cleaned eye features |
| Fusion | Combined 58 features |

Final arrays used:

- X_fused.npy
- X_eeg_pca.npy
- X_eye_clean.npy
- y.npy

---

## ⚠️ Critical Issue Encountered

### Problem

NaN loss detected at epoch 1

### Impact

- Training failed  
- Accuracy dropped to ~25% (random)  
- Model collapse (single-class prediction)  

---

### Root Cause

- NaN values  
- Infinite values  
- Invalid feature distributions  

---

### Solution

```python
bad_rows = np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1)

Removed corrupted samples



---

📊 Dataset Details

Metric	Value

Original samples	37,575
Corrupted removed	26
Final dataset	37,549
Features	58
Classes	4



---

⚙️ Training Configuration

Parameter	Value

Epochs	30
Batch Size	64
Learning Rate	1e-4
Loss	CrossEntropy
Split	80/20



---

🧠 Models Implemented

Model	Description

MLP	Baseline dense network
DNN	Deep neural network
Attention	Feature weighting
Hybrid	Dense + Attention
Decision Fusion	EEG + Eye late fusion



---

📊 Final Results (PROOF)

Model	Accuracy	Precision	Recall	F1-score

MLP	76.03%	76.67%	76.03%	75.96%
DNN	90.35%	90.48%	90.35%	90.33%
Attention	68.25%	68.39%	68.25%	68.24%
Hybrid	92.84%	92.86%	92.84%	92.84%
Decision Fusion	50.54%	51.06%	50.54%	50.44%



---

📈 Key Observations

What Worked

Data cleaning fixed training instability

Feature scaling improved convergence

Hybrid model captured best patterns

Window-based learning preserved signal info



---

What Failed

Decision Fusion performed poorly (~50%)

Attention alone underperformed

Unclean data caused total failure



---

📉 Before vs After Fix

Stage	Result

Before cleaning	NaN loss, failure
After cleaning	Stable training
Final	92.84% accuracy



---

🧠 Why Hybrid Model Performs Best

Combines:

Dense learning (global patterns)

Attention (feature importance)


Strong interaction between EEG + Eye features



---

📊 Outputs Generated

Each model folder contains:

accuracy.txt

classification_report.txt

confusion_matrix.png


Comparison folder:

model_comparison.csv

performance_plot.png



---

📚 Research Alignment

This project aligns with:

EEG-based emotion recognition research

Multimodal fusion improving performance

Attention mechanisms enhancing feature selection



---

🧾 Conclusion

Window-based training is stable and effective

Data quality is critical

Hybrid model achieved best performance (92.84%)

Proper preprocessing > complex modeling



---

🔮 Future Work

Transformer-based models

Real-time emotion detection

Advanced fusion strategies

Hyperparameter tuning



---

👨‍💻 Final Note

This project demonstrates:

Debugging real-world ML issues

Model comparison and evaluation

Handling corrupted datasets

Achieving strong classification performance



---
