import os
import tempfile

import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
import librosa
from laughter_prediction.sample_audio import sample_wav_by_time
# import laughter_classification.psf_features as psf_features


class FeatureExtractor:
    def extract_features(self, wav_path):
        """
        Extracts features for classification ny frames for .wav file

        :param wav_path: string, path to .wav file
        :return: pandas.DataFrame with features of shape (n_chunks, n_features)
        """
        raise NotImplementedError("Should have implemented this")


class PyAAExtractor(FeatureExtractor):
    """Python Audio Analysis features extractor"""
    def __init__(self):
        self.extract_script = "./laughter_prediction/extract_pyAA_features.py"
        self.py_env_name = "ipykernel_py2"

    def extract_features(self, wav_path):
        with tempfile.NamedTemporaryFile() as tmp_file:
            feature_save_path = tmp_file.name
            cmd = "python \"{}\" --wav_path=\"{}\" " \
                  "--feature_save_path=\"{}\"".format(self.extract_script, wav_path, feature_save_path)
            os.system('bash -c "source activate {}; {}"'.format(self.py_env_name, cmd))

            feature_df = pd.read_csv(feature_save_path)
        return feature_df


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
