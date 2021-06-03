import pandas as pd
import numpy as np
from model import NN
import torch.optim as opt


NUM_MODELS_TO_TRAIN = 10


def main():
    rng = np.random.default_rng()

    # create empty storage for model dictionary
    # will result in a list of dictionaries where each dictionary
    # will contain all the information about a trained model
    models = []

    # establish lists of potential hyperparameter values
    epochs = [10, 20, 30, 40]
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [200, 500, 1000]
    optims = [opt.SGD, opt.Adam, opt.Adagrad, opt.RMSprop]
    # add other hyperparams

    for i in range(NUM_MODELS_TO_TRAIN):
        # randomly select hyperparams
        ep = epochs[rng.integers(len(epochs))]
        lr = learning_rates[rng.integers(len(learning_rates))]
        bs = batch_sizes[rng.integers(len(batch_sizes))]
        optimizer = optims[rng.integers(len(optims))]

        print(f'Starting model # {i + 1} out of {NUM_MODELS_TO_TRAIN} -------')
        models.append(train_model(ep, bs, lr, optimizer))

    model_df = pd.DataFrame(models)
    return model_df


def train_model(num_epochs, train_batch_size, learning_rate, optimizer,
                train_pct=0.7):
    model_info = {}
    # store hyperparams
    model_info['num_epochs'] = num_epochs
    model_info['learning_rate'] = learning_rate
    model_info['training_batch_size'] = train_batch_size
    model_info['training_pct_split'] = train_pct

    # prep data
    ## TODO
    num_train_batchs = None

    # define model
    model = NN()

    # loss function
    loss_func = None

    # optimizer
    optim = optimizer(model.parameters(), lr=learning_rate)

    # setup tracked values
    # loss
    loss_list = np.zeros((num_epochs,))
    val_loss_list = np.zeros((num_epochs,))

    # train & validate model
    for epoch in range(num_epochs):
        epoch_loss = 0

        # train loop over each batch
        for batch_num, data_batch in enumerate(train):
            X = data_batch[0]
            y = data_batch[1]
            
            # zero gradient
            optim.zero_grad()

            # predict
            y_hat = model(X)

            # calc loss
            batch_loss = loss_func(y_hat, y)

            # back prop

            # step downhill

            epoch_loss += batch_loss.item()
            epoch_num_tot += y.shape[0]
            print(f'Epoch - {epoch + 1} / {num_epochs} - :\tEnd of Batch\t--'
                  + f' {batch_num + 1} / {num_train_batches} ---')
    pass


if __name__ == '__main__':
    model_df = main()
