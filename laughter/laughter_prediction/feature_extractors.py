import os
import tempfile

import pandas as pd
import numpy as np
import librosa


class FeatureExtractor:
    def extract_features(self, wav_path):
        """
        Extracts features for classification ny frames for .wav file

        :param wav_path: string, path to .wav file
        :return: pandas.DataFrame with features of shape (n_chunks, n_features)
        """
        raise NotImplementedError("Should have implemented this")


class MyExtractor(FeatureExtractor):
    def __init__(self, frame_len_sec=0.1):
        self.frame_len_sec = frame_len_sec

    def extract_features(self, wav_path):
        y, sr = librosa.load(wav_path, sr=None)
        # print(y)
        frame_len = int(sr * self.frame_len_sec)
        mfcc = [librosa.feature.mfcc(y[frame:frame + frame_len],
                                     sr).mean(axis=1)
                for frame in range(0, len(y) - frame_len + 1, frame_len)]
        mfcc = np.vstack(mfcc)
        # mfcc = librosa.feature.mfcc(y, sr, hop_length=frame_len)
        # print(mfcc.shape)
        # print(len(y) - frame_len + 1, frame_len // 4)
        melspec = [librosa.feature.melspectrogram(y[frame:frame + frame_len],
                                                  sr).mean(axis=1)
                   for frame in range(0, len(y) - frame_len + 1, frame_len)]
        melspec = np.vstack(melspec)
        # melspec = librosa.feature.melspectrogram(y, sr, hop_length=frame_len)
        melspec = np.log10(melspec)
        # print(melspec.shape)
        # print(np.hstack([mfcc.T, melspec.T]).shape)
        df = pd.DataFrame(np.hstack([mfcc, melspec]))
        cols = [f'MFCC_{i}' for i in range(mfcc.shape[1])] + \
               [f'MEL_{i}' for i in range(melspec.shape[1])]
        df.columns = cols
        # print(df.head())
        return df
