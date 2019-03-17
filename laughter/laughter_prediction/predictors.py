from laughter_classification.train_rnn_laugh_classifier import *
import torch
import torch.nn.functional as F


class Predictor:
    """
    Wrapper class used for loading serialized model and
    using it in classification task.
    Defines unified interface for all inherited predictors.
    """

    def predict(self, X):
        """
        Predict target values of X given a model

        :param X: numpy.ndarray, dtype=float, shape=[n_samples, n_features]
        :return: numpy.array predicted classes
        """
        raise NotImplementedError("Should have implemented this")

    def predict_proba(self, X):
        """
        Predict probabilities of target class

        :param X: numpy.ndarray, dtype=float, shape=[n_samples, n_features]
        :return: numpy.array target class probabilities
        """
        raise NotImplementedError("Should have implemented this")


class RnnPredictor(Predictor):
    def __init__(self, model_='models/model_dataset_all.pth'):
        if isinstance(model_, str):
            self.model = RnnLaughClassifier(N_MFCC, N_MEL, N_MFCC_HID, N_MEL_HID)
            self.model.load_state_dict(torch.load(model_))
        else:
            self.model = model_

    def predict(self, X):
        p = self.predict_proba(X)
        return p[:, 1] > p[:, 0]

    def predict_proba(self, X):
        X1, X2 = X[:, :N_MFCC], X[:, N_MFCC:]
        X1 = torch.Tensor(X1)
        X2 = torch.Tensor(X2)
        with torch.no_grad():
            self.model.init_hidden()
            _, out2 = self.model((X1, X2))
            p = F.softmax(out2, dim=1)
        return p.data
