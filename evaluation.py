# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 15:00:48 2021

@author: 888bk
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import NN
import torch
import torch.nn as nn
import torch.optim as opt
from data_analysis import gen_partition_idxs
from data_analysis import SPDCallDataset
from data_analysis import load_data
from training import load_pickled_df
import FeatureExtractor as feat
import pickle
from datetime import datetime

MODEL_FILE = './model_df_202164_1837.pickle'
FILE_PATH = r'data/Call_Data_2018.csv'

def main():
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # evaluation relies on random seed to be the same so that data split is same as used in training
    rng = np.random.default_rng()

    # create dataset
    pct_test = 0.15
    parts = gen_partition_idxs(FILE_PATH, pct_test=pct_test, pct_val=pct_test)
    
    # initialize feature extractor
    w2v_path = 'c:/Users/mrhoa/Documents/Education/ee511/project/' \
        + 'article-publisher-classifier/GoogleNews-vectors-negative300.bin'
    feat_extractor = feat.FeatureExtractor(from_file='embed_dict.pickle')
    
    # generate datasets
    train_dataset = SPDCallDataset(parts['train'], FILE_PATH, feat_extractor)
    val_dataset = SPDCallDataset(parts['val'], FILE_PATH, feat_extractor)
    test_dataset = SPDCallDataset(parts['test'], FILE_PATH, feat_extractor)
    
    # load the model file
    model_df = load_pickled_df(MODEL_FILE)
    num_models = len(model_df)
    
    # iterate and evaluate all the models in the file
    for i in range(num_models):
        # Pass whatever datasets you want to be evaluated to this function
        evaluate_model(model_df.iloc[i], device, train_dataset=train_dataset)
        
    
def evaluate_model(model_df, device, train_dataset=None, val_dataset=None, test_dataset=None):
    train_batch_size = int(model_df['training_batch_size'])
    num_workers = 2
    if train_batch_size == 500:
        num_workers = 4
    elif train_batch_size == 200:
        num_workers = 8
    
    model = model_df['model']
    model.to(device)
    
    # plot the loss, prefers batch training for more accurate picture
    plt.figure()
    #if 'batch_train_loss' in model_df.columns():
    #    plt.plot(model_df['batch_train_loss'], label='Train')
    #else:
    plt.plot(model_df['training_loss'], label='Train')
    plt.plot(model_df['validation_loss'], label='Validation')
        
    plt.title('Training and Validation Loss')
    plt.xlabel('Training Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    
    if train_dataset:
        train_dataset, _ = torch.utils.data.random_split(train_dataset, [10000, len(train_dataset) - 10000])
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=train_batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=True)

    if val_dataset:
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=train_batch_size,
                                                 num_workers=num_workers,
                                                 shuffle=False)
    
    if test_dataset:
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=train_batch_size,
                                                  num_workers=num_workers,
                                                  shuffle=False)
    
    loss_func = nn.MSELoss()
    
    # batch average loss
    train_loss = 0
    val_loss = 0
    test_loss = 0
    
    train_predictions = np.empty(0)
    train_y = np.empty(0)
    with torch.no_grad():
        if train_dataset:
            for batch_num, data_batch in enumerate(train_loader):
                X = data_batch[0].to(device)
                y = data_batch[1].to(device)
        
                y_hat = model(X)
        
                # calc loss
                batch_loss = loss_func(y_hat.squeeze(), y.float())
                train_loss += batch_loss.item()
                train_predictions = np.hstack([train_predictions, y_hat.cpu().detach().numpy().squeeze()])
                train_y = np.hstack([train_y, y.cpu().detach().numpy()])
                if batch_num % (len(train_loader)//10) == 0:
                    print(f'Train {batch_num}/{len(train_loader)}: loss = {train_loss}')
        
        if val_dataset:
            for batch_num, data_batch in enumerate(val_loader):
                X = data_batch[0].to(device)
                y = data_batch[1].to(device)
        
                y_hat = model(X)
        
                # calc loss
                batch_loss = loss_func(y_hat.squeeze(), y.float())
                val_loss += batch_loss.item()

                if batch_num % (len(val_loader)//10) == 0:
                    print(f'Val {batch_num}/{len(val_loader)}: loss = {val_loss}')
                    
        if test_dataset:
            for batch_num, data_batch in enumerate(test_loader):
                X = data_batch[0].to(device)
                y = data_batch[1].to(device)
        
                y_hat = model(X)
        
                # calc loss
                batch_loss = loss_func(y_hat.squeeze(), y.float())
                test_loss += batch_loss.item()

                if batch_num % (len(test_loader)//10) == 0:
                    print(f'Test {batch_num}/{len(test_loader)}: loss = {test_loss}')
    
    plt.figure()
    plt.scatter(train_predictions, train_y)
    plt.xlabel('Predicted Response Time')
    plt.ylabel('Response Time')
    plt.draw()
    return [train_loss, val_loss, test_loss], [train_predictions], [train_y]

if __name__ == '__main__':
    model_df = main()
