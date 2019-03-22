import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

import numpy as np
from timeit import default_timer as timer


class SpeakerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SpeakerClassifier, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden = None

    def init_hidden(self, batch_len, cuda=False):
        if cuda:
            self.hidden = (torch.zeros(1, batch_len, self.hidden_dim).cuda(),
                           torch.zeros(1, batch_len, self.hidden_dim).cuda())
        else:
            self.hidden = (torch.zeros(1, batch_len, self.hidden_dim),
                           torch.zeros(1, batch_len, self.hidden_dim))

    def forward(self, X, lens):
        X = pack_padded_sequence(X, lens, batch_first=True)
        _, (hout, _) = self.lstm(X, self.hidden)
        out = self.fc(hout[-1])
        return out

    # def predict(self, X, cuda=False):
    #     self.init_hidden(cuda)
    #     out = self.forward(X)
    #     p = F.softmax(out, dim=1)
    #     return torch.argmax(p, dim=1)


def gen_batches(dataset, batch_len):
    for i in range(0, len(dataset), batch_len):
        lens = np.array([t[0].size(0) for t in dataset[i:i+batch_len]])
        idx = np.argsort(lens)[::-1]
        Xs, ys = zip(*(dataset[i + j] for j in idx))
        Xs, ys = list(Xs), list(ys)
        X = pad_sequence(Xs, batch_first=True)
        y = torch.cat(ys)
        yield X, y, lens[idx]


def train_model(model, optimizer, loss_fn, train_dataset, test_dataset,
                batch_len, n_epochs, save_path=None,
                seed=None, use_cuda=False, n_epochs_to_log=None):
    if seed is not None:
        torch.manual_seed(seed)

    if use_cuda:
        model = model.cuda()
        train_dataset = [(X.cuda(), y.cuda())
                         for X, y in train_dataset]
        test_dataset = [(X.cuda(), y.cuda())
                        for X, y in test_dataset]

    start_time = timer()
    test_losses = []
    test_accs = []
    for epoch in range(n_epochs):
        epoch_start_time = timer()

        mean_train_loss = 0
        for X, y, lens in gen_batches(train_dataset, batch_len):
            model.zero_grad()
            model.init_hidden(len(lens), cuda=use_cuda)
            out = model(X, lens)
            # print(out, out.size(), y.size(), sep='\n')
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            mean_train_loss += loss.item()
        mean_train_loss /= len(train_dataset)

        mean_test_loss = 0
        mean_test_acc = 0
        with torch.no_grad():
            for X, y, lens in gen_batches(test_dataset, batch_len):
                model.init_hidden(len(lens), cuda=use_cuda)
                out = model(X, lens)
                loss = loss_fn(out, y)
                y_pred = torch.argmax(out, dim=1)
                mean_test_acc += torch.sum(y == y_pred).item()
                mean_test_loss += loss.item()
        mean_test_loss /= len(test_dataset)
        mean_test_acc /= len(test_dataset)
        test_losses.append(mean_test_loss)
        test_accs.append(mean_test_acc)

        if n_epochs_to_log is not None and (epoch + 1) % n_epochs_to_log == 0:
            print(f'Epoch {epoch + 1} finished in {timer() - epoch_start_time:.3f} '
                  f'seconds, mean train loss: {mean_train_loss:.5f}, '
                  f'mean test loss: {mean_test_loss:.5f}, '
                  f'mean test accuracy: {mean_test_acc:.4f}')
    print(f'Learning finished in'
          f' {timer() - start_time:.3f} seconds')

    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model, test_losses, test_accs
