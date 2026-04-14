import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModel(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super(AttentionModel, self).__init__()
        self.attention_weights = nn.Linear(input_dim, input_dim)
        self.fc1 = nn.Linear(input_dim, 64)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        attn_scores = torch.sigmoid(self.attention_weights(x))
        context = x * attn_scores
        x = F.relu(self.fc1(context))
        x = self.out(x)
        return x
