import torch.nn as nn
import torch.nn.functional as F

class BaselineMLP(nn.Module):
    """
    Model 1: Baseline MLP
    Input -> Dense(128) -> Dropout(0.3) -> Dense(64) -> Output classes
    """
    def __init__(self, input_dim, num_classes=4):
        super(BaselineMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
