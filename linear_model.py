import pandas as pd
import numpy as np
from model import NN
import torch
import torch.nn as nn
import torch.optim as opt
from data_analysis import gen_partition_idxs
from data_analysis import SPDCallDataset
from data_analysis import load_data
import FeatureExtractor as feat
import pickle
from datetime import datetime


OUT_DIM = 1

class LinearModel(nn.Module):
    def __init__(self, input_dim):
        super(NN, self).__init__()
        self.fc = nn.Linear(input_dim, OUT_DIM)

    def forward(self, x):
        # n x d input
        return self.fc(x)
