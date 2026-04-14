import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridModel(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super(HybridModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.attention_weights = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        attn_scores = torch.sigmoid(self.attention_weights(x))
        context = x * attn_scores
        x = F.relu(self.fc2(context))
        x = self.out(x)
        return x
