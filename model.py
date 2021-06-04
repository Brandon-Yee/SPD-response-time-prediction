# -*- coding: utf-8 -*-
"""
SPD Model Definitions
"""

#import torch
import torch.nn as nn
import torch.nn.functional as F
#import pandas as pd

OUT_DIM = 1

class NN(nn.Module):
    def __init__(self, layer_sizes):
        super(NN, self).__init__()
        self.layer_sizes = layer_sizes
        self.hidden_layers = nn.ModuleList()
        for i in range(len(layer_sizes)-1):
            self.hidden_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        self.hidden_layers.append(nn.Linear(self.layer_sizes[-1], OUT_DIM))

    def forward(self, x):
        # n x d input
        for i in range(len(self.hidden_layers) - 1):
            x = self.hidden_layers[i](x)
            x = F.relu(x)
        x = self.hidden_layers[-1](x)
        return x
