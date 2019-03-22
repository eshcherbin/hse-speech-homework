# Adapted from https://github.com/pykaldi/pykaldi/blob/master/examples/notebooks/mfcc-extraction.ipynb

from kaldi.feat.mfcc import Mfcc, MfccOptions
from kaldi.feat.fbank import Fbank, FbankOptions
from kaldi.matrix import SubVector, SubMatrix
from kaldi.util.table import SequentialWaveReader
from kaldi.util.table import MatrixWriter
import numpy as np
from sklearn.preprocessing import scale


SAMP_FREQ = 48000


def extract_mfcc(scp_file, result_file, samp_freq=SAMP_FREQ):
    mfcc_opts = MfccOptions()
    mfcc_opts.frame_opts.samp_freq = samp_freq
    mfcc = Mfcc(mfcc_opts)

    assert scp_file.endswith('.scp')
    assert result_file.endswith('.ark')
    rspec = f'scp:{scp_file}'
    wspec = f'ark:{result_file}'
    with SequentialWaveReader(rspec) as reader, \
            MatrixWriter(wspec) as writer:
        for key, wav in reader:
            assert (wav.samp_freq == samp_freq)

            s = wav.data()
            m = SubVector(np.mean(s, axis=0))
            f = mfcc.compute_features(m, samp_freq, 1.0)
            f = SubMatrix(scale(f))
            writer[key] = f


def extract_fbank(scp_file, result_file, samp_freq=SAMP_FREQ):
    fbank_opts = FbankOptions()
    fbank_opts.frame_opts.samp_freq = samp_freq
    fbank = Fbank(fbank_opts)

    assert scp_file.endswith('.scp')
    assert result_file.endswith('.ark')
    rspec = f'scp:{scp_file}'
    wspec = f'ark:{result_file}'
    with SequentialWaveReader(rspec) as reader, \
            MatrixWriter(wspec) as writer:
        for key, wav in reader:
            assert (wav.samp_freq == samp_freq)

            s = wav.data()
            m = SubVector(np.mean(s, axis=0))
            f = fbank.compute_features(m, samp_freq, 1.0)
            f = SubMatrix(scale(f))
            writer[key] = f
