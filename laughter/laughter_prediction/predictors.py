import numpy as np

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from laughter_classification.rnn_laugh_classifier import RnnLaughClassifier
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


class XgboostPredictor(Predictor):
    """Parametrized wrapper for xgboost-based predictors"""

    def __init__(self, model_path, threshold, scaler=None):
        self.threshold = threshold
        self.clf = joblib.load(model_path)
        self.scaler = scaler

    def _simple_smooth(self, data, n=50):
        dlen = len(data)

        def low_pass(data, i, n):
            if i < n // 2:
                return data[:i]
            if i >= dlen - n // 2 - 1:
                return data[i:]
            return data[i - n // 2: i + n - n // 2]

        sliced = np.array([low_pass(data, i, n) for i in range(dlen)])
        sumz = np.array([np.sum(x) for x in sliced])
        return sumz / n

    def predict(self, X):
        y_pred = self.clf.predict_proba(X)
        ypreds_bin = np.where(y_pred[:, 1] >= self.threshold, np.ones(len(y_pred)), np.zeros(len(y_pred)))
        return ypreds_bin

    def predict_proba(self, X):
        X_scaled = self.scaler.fit_transform(X) if self.scaler is not None else X
        not_smooth = self.clf.predict_proba(X_scaled)[:, 1]
        return self._simple_smooth(not_smooth)


class StrictLargeXgboostPredictor(XgboostPredictor):
    """
    Predictor trained on 3kk training examples, using PyAAExtractor
    for input features
    """
    def __init__(self, threshold=0.045985743):
        XgboostPredictor.__init__(self, model_path="models/XGBClassifier_3kk_pyAA10.pkl",
                                  threshold=threshold, scaler=StandardScaler())


class RnnPredictor(Predictor):
    def __init__(self, model_='models/model_dataset_500.pth'):
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
            _, out2 = self.model((X1, X2))
            p = F.softmax(out2, dim=1)
        return p.data
