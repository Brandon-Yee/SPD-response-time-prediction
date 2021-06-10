import torch
import torch.nn as nn


OUT_DIM = 1

class LinearModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(input_dim, OUT_DIM)

    def forward(self, x):
        # n x d input
        return self.fc(x)
