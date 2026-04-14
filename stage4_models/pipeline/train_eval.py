import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pipeline.config import OUTPUT_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name, sub_folder, is_decision_fusion=False):
    """
    Standardized Training and Evaluation Loop.
    is_decision_fusion indicates if the model requires separated X matrices.
    """
    model_dir = os.path.join(OUTPUT_DIR, sub_folder)
    ensure_dir(model_dir)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    y_train_t = torch.LongTensor(y_train)
    y_test_t = torch.LongTensor(y_test)
    
    if is_decision_fusion:
        X_tr_t = (torch.FloatTensor(X_train[0]), torch.FloatTensor(X_train[1]))
        X_ts_t = (torch.FloatTensor(X_test[0]), torch.FloatTensor(X_test[1]))
    else:
        X_tr_t = torch.FloatTensor(X_train)
        X_ts_t = torch.FloatTensor(X_test)
        
    num_samples = y_train_t.size(0)
    print(f"--- Training {model_name} ---")
    model.train()
    
    for epoch in range(EPOCHS):
        permutation = torch.randperm(num_samples)
        epoch_loss = 0.0
        
        for i in range(0, num_samples, BATCH_SIZE):
            indices = permutation[i:i+BATCH_SIZE]
            batch_y = y_train_t[indices]
            
            optimizer.zero_grad()
            
            if is_decision_fusion:
                outputs = model(X_tr_t[0][indices], X_tr_t[1][indices])
            else:
                outputs = model(X_tr_t[indices])
                
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} -> Loss: {epoch_loss/max(1, (num_samples/BATCH_SIZE)):.4f}")
            
    # Evaluation Phase
    model.eval()
    with torch.no_grad():
        if is_decision_fusion:
            test_outputs = model(X_ts_t[0], X_ts_t[1])
        else:
            test_outputs = model(X_ts_t)
            
        _, predicted = torch.max(test_outputs, 1)
        pred_numpy = predicted.numpy()
        true_numpy = y_test_t.numpy()
        
    acc = accuracy_score(true_numpy, pred_numpy)
    cm = confusion_matrix(true_numpy, pred_numpy)
    report = classification_report(true_numpy, pred_numpy, zero_division=0)
    
    with open(os.path.join(model_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Model: {model_name}\nAccuracy: {acc:.4f}\n\n{report}")
        
    if PLOTTING_AVAILABLE:
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} Confusion Matrix')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'))
        plt.close()
    else:
        np.save(os.path.join(model_dir, 'confusion_matrix.npy'), cm)
    
    print(f"Completed {model_name} - Accuracy: {acc*100:.2f}%\n")
    return acc
