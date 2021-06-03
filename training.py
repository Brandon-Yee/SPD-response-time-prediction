import pandas as pd
import numpy as np
from model import NN
import torch
import torch.nn as nn
import torch.optim as opt
from data_analysis import gen_partition_idxs
from data_analysis import SPDCallDataset
import pickle
from datetime import datetime


NUM_MODELS_TO_TRAIN = 10
D = 1341
FILE_PATH = r'data/Call_Data_2018.csv'


def main():
    rng = np.random.default_rng()
    start = pd.to_datetime(datetime.now())

    # create empty storage for model dictionary
    # will result in a list of dictionaries where each dictionary
    # will contain all the information about a trained model
    models = []

    # create dataset
    pct_test = 0.15
    parts = gen_partition_idxs(FILE_PATH, pct_test=pct_test, pct_val=pct_test)

    train_dataset = SPDCallDataset(parts['train'], FILE_PATH)
    val_dataset = SPDCallDataset(parts['val'], FILE_PATH)
    test_dataset = SPDCallDataset(parts['test'], FILE_PATH)


    # establish lists of potential hyperparameter values
    epochs = [10, 20, 30, 40]
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [200, 500, 1000]
    optims = [opt.SGD, opt.Adam, opt.Adagrad, opt.RMSprop]
    decays = [0, 0.01, 0.1, 0.2, 0.5, 1]
    num_hidden_layers = [2, 3, 4, 5, 6]
    num_nodes = [50, 100, 200, 250]
    # add other hyperparams

    # create feature extractor here
    # TODO
    feat_extractor = None

    for i in range(NUM_MODELS_TO_TRAIN):
        # randomly select hyperparams
        ep = epochs[rng.integers(len(epochs))]
        lr = learning_rates[rng.integers(len(learning_rates))]
        bs = batch_sizes[rng.integers(len(batch_sizes))]
        opti = optims[rng.integers(len(optims))]
        decay = decays[rng.integers(len(decays))]

        num_h = num_hidden_layers[rng.integers(len(num_hidden_layers))]
        nodes = [D]
        for j in range(num_h):
            nodes.append(num_nodes[rng.integers(len(num_nodes))])

        print(f'Starting model # {i + 1} out of {NUM_MODELS_TO_TRAIN} -------')
        # train each model
        models.append(train_model(ep, bs, lr, opti, decay, nodes,
                                  feat_extractor, train_dataset, val_dataset))

    model_df = pd.DataFrame(models)

    # timing info
    end = pd.to_datetime(datetime.now())
    print(f"Script started at {start}\n\tand ended at {end}")
    print(f"Duration: {end - start}")

    # pickle here!
    save_file = "model_df_" + str(end.year) + str(end.month) + str(end.day) \
        + '_' + str(end.hour) + str(end.minute) + '.pickle'
    with open(save_file, 'wb') as f:
        pickle.dump(model_df, f)

    return model_df


def train_model(num_epochs, train_batch_size, learning_rate, optimizer, decay,
                nodes, feat_extractor, train_dataset, val_dataset):
    model_info = {}
    start_time = pd.to_datetime(datetime.now())
    # store hyperparams
    model_info['num_epochs'] = num_epochs
    model_info['learning_rate'] = learning_rate
    model_info['training_batch_size'] = train_batch_size
    model_info['num_hidden_layers'] = len(nodes) - 1
    model_info['num_nodes'] = nodes
    model_info['weight_decay'] = decay

    # prep data
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch_size,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=len(val_dataset),
                                             shuffle=False)

    num_train_batches = len(train_loader)


    # define model
    model = NN(nodes, feat_extractor)

    # loss function
    loss_func = nn.MSELoss()

    # optimizer
    optim = optimizer(model.parameters(), lr=learning_rate, weight_decay=decay)

    # setup tracked values
    # loss
    loss_list = np.zeros((num_epochs,))
    val_loss_list = np.zeros((num_epochs,))

    # train & validate model
    for epoch in range(num_epochs):
        epoch_loss = 0

        # train loop over each batch
        model.train()
        for batch_num, data_batch in enumerate(train_loader):
            X = data_batch[0]
            y = data_batch[1]

            # zero gradient
            optim.zero_grad()

            # predict
            y_hat = model(X)

            # calc loss
            batch_loss = loss_func(y_hat, y)

            # back prop
            batch_loss.backward()

            # step downhill
            optim.step()

            epoch_loss += batch_loss.item()
            epoch_num_tot += y.shape[0]
            print(f'Epoch - {epoch + 1} / {num_epochs} - :\tEnd of Batch\t--'
                  + f' {batch_num + 1} / {num_train_batches} ---')

        # store loss for the epoch
        loss_list[epoch] = epoch_loss
        print(f"End of Epoch\t{epoch + 1} -----------")
        print(f"Training Loss:\t\t{epoch_loss}")

        # evaluate on validation data
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:

                # predict
                y_val_hat = model(X_val)

                # calc loss
                val_loss += loss_func(y_val_hat, y_val).item()

        # store validation loss
        val_loss_list.append(val_loss)
        print(f"\tValidation loss:\t{val_loss}")
        print()

    end_time = pd.to_datetime(datetime.now())
    # store info
    model_info['training_time'] = end_time - start_time
    model_info['model'] = model
    model_info['training_loss'] = loss_list
    model_info['validation_loss'] = val_loss_list
    model_info['final_val_loss'] = val_loss_list[-1]

    return model_info


def load_pickled_df(file_path):
    with open(file_path, 'rb') as f:
        model_df = pickle.load(f)
    return model_df


if __name__ == '__main__':
    model_df = main()