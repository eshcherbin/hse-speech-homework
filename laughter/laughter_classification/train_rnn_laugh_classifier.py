import os
import click

from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from laughter_classification.rnn_laugh_classifier import *


# DATASET_FILE = 'data/sspnet_dataset_50.csv'
# DATASET_FILE = 'data/sspnet_dataset_500.csv'
DATASET_FILE = 'data/sspnet_dataset_all.csv'
CORPUS_ROOT = 'vocalizationcorpus'
FRAME_LEN_SEC = 0.1
N_MFCC = 20
N_MEL = 128
N_MFCC_HID = 10
N_MEL_HID = 50
N_EPOCHS = 50
LR = 0.01
SEED = 117
MODEL_PATH = 'models/model_dataset_all.pth'


def generate_data():
    from laughter_classification.sspnet_data_sampler import SSPNetDataSampler
    s = SSPNetDataSampler(CORPUS_ROOT)
    return s.create_sampled_df(FRAME_LEN_SEC, save_path=DATASET_FILE,
                               force_save=True)


def prepare_dataset(dataset):
    snames = dataset.SNAME.unique()
    train_snames, test_snames = train_test_split(snames)
    xs_train = [dataset[dataset.SNAME == sname].iloc[:, :-2].values for sname in train_snames]
    ys_train = [dataset[dataset.SNAME == sname].iloc[:, -2].values for sname in train_snames]
    xs_test = [dataset[dataset.SNAME == sname].iloc[:, :-2].values for sname in test_snames]
    ys_test = [dataset[dataset.SNAME == sname].iloc[:, -2].values for sname in test_snames]
    return [((torch.Tensor(x[:, :N_MFCC]), torch.Tensor(x[:, N_MFCC:])), torch.LongTensor(y)) for x, y in zip(xs_train, ys_train)], \
           [((torch.Tensor(x[:, :N_MFCC]), torch.Tensor(x[:, N_MFCC:])), torch.LongTensor(y)) for x, y in zip(xs_test, ys_test)]


@click.command()
@click.option('model_to_load', '--model', type=click.Path(), default=None)
def main(model_to_load):
    if os.path.exists(DATASET_FILE):
        dataset = pd.read_csv(DATASET_FILE)
    else:
        dataset = generate_data()

    dataset_train, dataset_test = prepare_dataset(dataset)
    model = RnnLaughClassifier(N_MFCC, N_MEL, N_MFCC_HID, N_MEL_HID)
    if model_to_load is None:
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        model = train_model(model, dataset_train, optimizer, loss_fn, n_epochs=N_EPOCHS, seed=SEED, save_path=MODEL_PATH)
    else:
        model.load_state_dict(torch.load(model_to_load))

    with torch.no_grad():
        mean_acc = 0
        mean_jacc = 0
        for X, y in dataset_test:
            y_pred = F.softmax(model(X)[1], dim=1).data
            y_pred = y_pred[:, 1] > y_pred[:, 0]
            # print(y)
            # print(y.data)
            # print(y_pred)
            mean_acc += (y_pred.numpy() == y.numpy()).mean()
            tmp = (y_pred.numpy() | y.numpy()).sum()
            mean_jacc += 1 if tmp == 0 else (y_pred.numpy() & y.numpy()).sum() / tmp
        mean_acc /= len(dataset_test)
        mean_jacc /= len(dataset_test)
        print(f'Mean test accuracy: {mean_acc:.6f}')
        print(f'Mean test Jaccard similarity: {mean_jacc:.6f}')


if __name__ == '__main__':
    main()

