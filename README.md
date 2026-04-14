# 🧠 EEG + Eye-Tracking Emotion Classification (Window-Based Deep Learning)

---

## 🚀 What this project does

This project classifies human emotions using:

- EEG signals (brain activity)
- Eye-tracking data

---
## 📚 Literature Review (Research Justification)

To design an effective emotion classification system, multiple research papers and prior works on EEG, eye-tracking, deep learning, and multimodal fusion were studied.

### 🔹 Key Areas Explored

1. *EEG-based Emotion Recognition*
   - EEG signals capture brain activity patterns linked to emotional states
   - Deep learning models (DNN, CNN, RNN) show strong performance
   - Challenge: Noisy signals and high dimensionality

2. *Eye-Tracking for Emotion Detection*
   - Eye movement patterns (fixation, saccades) correlate with attention and emotion
   - Works well as a complementary modality
   - Limitation: weaker standalone performance

3. *Deep Learning Models*
   - MLP: baseline model for structured data
   - DNN: deeper architecture improves feature learning
   - Attention models: focus on important features
   - Hybrid models: combine multiple learning strategies

4. *Multimodal Fusion Techniques*
   - Feature-level fusion: combines EEG + Eye features before training
   - Decision-level fusion: combines outputs of separate models
   - Research shows multimodal fusion improves robustness and accuracy

---

### 🔹 Key Insights from Literature

| Concept | Insight |
|--------|--------|
| EEG | Strong signal for emotion but noisy |
| Eye-tracking | Useful but weaker alone |
| Deep models | Better than traditional ML |
| Attention | Helps focus on important features |
| Fusion | Improves overall performance |

---

### 🔹 How Literature Influenced This Work

Based on research findings:

- Used *window-based learning* (instead of trial averaging)
- Implemented *multiple deep learning models* for comparison
- Applied *feature-level fusion (X_fused)*
- Implemented *decision-level fusion model*
- Evaluated all models using standard metrics

---
### Approach:
- Window-based learning (NO trial aggregation)
- Deep learning models for classification
- Multimodal feature fusion

---
## 🚀 Final Result (Key Outcome)

- *Best Model:* Hybrid Model  
- *Accuracy:* *92.84%*  
- *Precision:* 92.86%  
- *Recall:* 92.84%  
- *F1-score:* 92.84%  

This model outperformed all other architectures including MLP, DNN, Attention, and Decision Fusion.

## 🧾 Final Results (PROOF)

| Model            | Accuracy | Precision | Recall | F1-score |
|------------------|---------|----------|--------|---------|
| MLP              | 76.03%  | 76.67%   | 76.03% | 75.96%  |
| DNN              | 90.35%  | 90.48%   | 90.35% | 90.33%  |
| Attention        | 68.25%  | 68.39%   | 68.25% | 68.24%  |
| Hybrid           | *92.84%* | *92.86%* | *92.84%* | *92.84%* |
| Decision Fusion  | 50.54%  | 51.06%   | 50.54% | 50.44%  |

✅ *Best Model: Hybrid (92.84%)*

---

## 📁 Project Structure


8th_sem_new/
│
├── stage1_data/              # Raw data
├── stage2_preprocessing/     # Cleaning + feature extraction
├── stage3_feature_analysis/  # Feature understanding
├── stage4_models/            # Model training
│   ├── mlp/
│   ├── dnn/
│   ├── attention/
│   ├── hybrid/
│   ├── decision_fusion/
│   └── comparison/
│
├── stage4_pipeline/          # Processed numpy arrays
├── docs/                     # Detailed explanations
├── run_models.py             # Main script
└── README.md

---
## ▶️ How to Run

1. Ensure processed data is available in:
processed_data/

2. Run the training script:
python stage4_models/run_models.py

3. Outputs generated:
- Model weights
- Confusion matrices
- Model comparison CSV
- Performance plot
---

## ⚠️ Critical Issue Discovered

When scaling from 5,000 → 37,575 samples:

- ❌ Training failed
- ❌ Accuracy dropped to ~25% (random baseline)
- ❌ Models collapsed (predicting single class)

---

## 🔍 Root Cause

Dataset contained corrupted values:

- NaN values
- Infinite values
- Invalid feature scaling (division by zero)

These caused:
- Loss = NaN
- Gradients = invalid
- Model collapse

---

## ✅ Solution Applied

Corrupted samples removed before training:

python
bad_rows = np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1)
X = X[~bad_rows]


---

## 📊 Dataset After Fix

| Metric              | Value |
|--------------------|------|
| Original samples   | 37,575 |
| Corrupted removed  | 26 |
| Final dataset      | 37,549 |
| Features           | 58 |
| Classes            | 4 |

---

## ⚙️ Training Configuration

| Parameter   | Value |
|------------|------|
| Epochs     | 30 |
| Batch Size | 64 |
| Learning Rate | 1e-4 |
| Split      | 80% Train / 20% Test |

---

## 🧠 Models Implemented

- MLP (Baseline)
- Deep Neural Network (DNN)
- Attention Model
- Hybrid Model (Best)
- Decision Fusion (EEG + Eye)

---

## 📊 Model Insights

- Hybrid model achieved highest accuracy (*92.84%*)
- DNN performed strongly (~90%)
- Attention model alone underperformed
- Decision Fusion failed (~50%)

### Why Hybrid Works:
- Combines deep learning + attention
- Captures complex feature relationships
- More robust than single models

---

## 📉 Before vs After Fix

| Scenario            | Result |
|--------------------|--------|
| Before cleaning    | ~25% accuracy (random) |
| After cleaning     | 92.84% accuracy |
| Model stability    | Fixed |
| NaN loss issue     | Eliminated |

---

## 🧠 Key Learnings

- Data quality is critical (NaN breaks training completely)
- Feature scaling must be handled carefully
- Hybrid architectures outperform single models
- Window-based learning preserves signal information

---

## 🔬 Research Alignment

This project follows principles from:

- Multimodal emotion recognition research
- EEG-based deep learning classification
- Attention-based feature weighting methods

Consistent findings:
- Multimodal fusion improves performance
- Deep models outperform shallow methods
- Data preprocessing is critical

---

## 🚀 Future Work

- Transformer-based architectures
- Real-time emotion detection
- Advanced fusion strategies
- Hyperparameter tuning

---

## 📌 Final Note

This project demonstrates:

- Real-world ML debugging (NaN failure → fix)
- Model comparison across architectures
- Handling corrupted datasets
- Achieving high classification performance (92.84%)
