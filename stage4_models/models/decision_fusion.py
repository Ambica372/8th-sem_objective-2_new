import torch.nn as nn
import torch.nn.functional as F

class DecisionFusion(nn.Module):
    def __init__(self, eeg_dim, eye_dim, num_classes=4):
        super(DecisionFusion, self).__init__()
        self.eeg_fc1 = nn.Linear(eeg_dim, 64)
        self.eeg_out = nn.Linear(64, num_classes)
        
        self.eye_fc1 = nn.Linear(eye_dim, 32)
        self.eye_out = nn.Linear(32, num_classes)

    def forward(self, eeg_x, eye_x):
        eeg_pred = self.eeg_out(F.relu(self.eeg_fc1(eeg_x)))
        eye_pred = self.eye_out(F.relu(self.eye_fc1(eye_x)))
        fused_pred = (eeg_pred + eye_pred) / 2.0
        return fused_pred
