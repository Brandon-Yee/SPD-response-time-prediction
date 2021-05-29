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
    def __init__(self, layer_sizes, embeddings_dict, transform):
        super(NN, self).__init__()
        self.layer_sizes = layer_sizes
        self.embeddings_dict = embeddings_dict
        self.transform = transform
        self.hidden_layers = []
        for i in range(len(layer_sizes)-1):
            self.hidden_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
           
    def forward(self, x):
        # n x d input
        x = self.transform(x, self.embeddings_dict)
        for i in range(len(self.hidden_layers)):
            x = self.hidden_layers[i](x)
            x = F.relu(x)
        x = nn.Linear(self.layer_sizes[-1], OUT_DIM)(x)
        return x