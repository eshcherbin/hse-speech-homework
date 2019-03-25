from timeit import default_timer as timer

import torch
import torch.nn as nn


class RnnLaughClassifier(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size1, hidden_size2):
        super(RnnLaughClassifier, self).__init__()

        self.input_size1 = input_size1
        self.hidden_size1 = hidden_size1
        self.hidden1 = None
        self.lstm1 = nn.LSTM(input_size1, hidden_size1)
        self.fc1 = nn.Linear(hidden_size1, 2)

        self.input_size2 = input_size2
        self.hidden_size2 = hidden_size2
        self.hidden2 = None
        self.lstm2 = nn.LSTM(input_size2, hidden_size2)
        self.fc2 = nn.Linear(hidden_size1 + hidden_size2, 2)

        self.init_hidden()

    def init_hidden(self, device='cpu'):
        self.hidden1 = (
            torch.zeros(1, 1, self.hidden_size1).to(device),
            torch.zeros(1, 1, self.hidden_size1).to(device)
        )
        self.hidden2 = (
            torch.zeros(1, 1, self.hidden_size2).to(device),
            torch.zeros(1, 1, self.hidden_size2).to(device)
        )

    def forward(self, X):
        # X1, X2 = torch.split(X, [self.input_size1, self.input_size2], dim=1)
        X1, X2 = X

        # print(X1.shape)
        lstm_out1, _ = self.lstm1(X1.contiguous().view(-1, 1, self.input_size1),
                                  self.hidden1)
        out1 = self.fc1(lstm_out1.view(-1, self.hidden_size1))

        lstm_out2, _ = self.lstm2(X2.contiguous().view(-1, 1, self.input_size2),
                                  self.hidden2)

        lstm_out = torch.cat([lstm_out1, lstm_out2], dim=2)
        out2 = self.fc2(lstm_out.view(-1,
                                      self.hidden_size1 + self.hidden_size2))

        return out1, out2


def train_model(model, train_dataset, optimizer, loss_fn, n_epochs=10,
                save_path=None, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    model = model.cuda()
    train_dataset = [((mfcc.cuda(), mel.cuda()), labels.cuda()) for (mfcc, mel), labels in train_dataset]

    start_time = timer()
    for epoch in range(n_epochs):
        epoch_start_time = timer()

        mean_loss = 0
        # print(dataset)
        for audio, labels in train_dataset:
            # print(labels.is_cuda)
            # print(labels.dtype)
            model.zero_grad()
            model.init_hidden(device='cuda')
            # model.init_hidden()
            out1, out2 = model(audio)
            loss1 = loss_fn(out1, labels)
            loss2 = loss_fn(out2, labels)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            mean_loss += loss.item()
        mean_loss /= len(train_dataset)

        print(f'Epoch {epoch} finished in {timer() - epoch_start_time:.3f} '
              f'seconds, mean loss: {mean_loss:.3f}, ')
    print(f'Learning finished in'
          f' {timer() - start_time:.3f} seconds')

    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model
