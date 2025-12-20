#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import numpy as np
import random
from numpy import sum, sqrt
from numpy.random import standard_normal, uniform, randint
import torchvision
from scipy import signal
from torch import nn
from sklearn.model_selection import train_test_split
import pywt
import matplotlib.pyplot as plt
from PIL import Image


















def rms_normalize_per_sample(x: np.ndarray, eps=1e-12) -> np.ndarray:
    rms = np.sqrt(np.mean(np.abs(x)**2) + eps)
    return x / rms

def awgn_per_packet(x: np.ndarray, snr_db: float) -> np.ndarray:
    snr_lin = 10.0 ** (snr_db / 10.0)
    n0 = 1.0 / snr_lin
    n = np.sqrt(n0 / 2.0) * (standard_normal(x.shape) + 1j * standard_normal(x.shape))
    return x + n

def add_awgn_batch(x_batch: np.ndarray, split: str) -> np.ndarray:
    out = np.empty_like(x_batch)
    for i in range(x_batch.shape[0]):
        snr = uniform(0, 10) if split == 'train' else uniform(-5, 20)
        out[i] = awgn_per_packet(x_batch[i], snr)
    return out

class STFTSpectrogram():
    def __init__(self, ):
        pass

    def _normalization(self, data):
        s_norm = np.zeros(data.shape, dtype=complex)

        for i in range(data.shape[0]):
            sig_amplitude = np.abs(data[i])
            rms = np.sqrt(np.mean(sig_amplitude ** 2))
            s_norm[i] = data[i] / rms

        return s_norm


    def _spec_crop(self, x):
        num_row = x.shape[0]
        x_cropped = x[round(num_row * 0.3):round(num_row * 0.7)]

        return x_cropped

    def gen_stft_spectrogram(self, sig, win_len= 256, overlap= 128):
        f, t, spec = signal.stft(sig,
                                 window='boxcar',
                                 nperseg=win_len,
                                 noverlap=overlap,
                                 nfft=win_len,
                                 return_onesided=False,
                                 padded=False,
                                 boundary=None)
        spec = np.fft.fftshift(spec, axes=0)
        chan_ind_spec = spec[:, 1:] / spec[:, :-1]


        chan_ind_spec_amp = np.log10(np.abs(spec) ** 2)

        return chan_ind_spec_amp

    def STFT_spectrogram(self, data):
        data = self._normalization(data)

        num_sample = data.shape[0]

        num_row = int(256 * 0.4)

        num_column = int(np.floor((data.shape[1] - 256) / 128)+1)

        data_dspec = np.zeros([num_sample, 1, num_row, num_column])

        for i in range(num_sample):
            stft_spectrogram = self.gen_stft_spectrogram(data[i])
            stft_spectrogram = self._spec_crop(stft_spectrogram)
            data_dspec[i, 0, :, :] = stft_spectrogram

        return data_dspec
class LoadDataset():
    def __init__(self, ):
        self.dataset_name = 'data'
        self.labelset_name = 'label'

    def _convert_to_complex(self, data):
        num_row = data.shape[0]
        num_col = data.shape[1]
        data_complex = np.zeros([num_row, num_col // 2], dtype=np.complex64)

        data_complex[:] = data[:, 0::2] + 1j * data[:, 1::2]
        return data_complex

    def load_iq_samples(self, file_path, dev_range):

        f = h5py.File(file_path, 'r')
        label = f[self.labelset_name][:]
        label = label.astype(int)
        label = np.transpose(label)
        label = label - 1

        label_start = int(label.flatten()[0]) + 1
        label_end = int(label.flatten()[-1]) + 1
        num_dev = label_end - label_start + 1
        num_pkt = len(label)
        num_pkt_per_dev = int(num_pkt / num_dev)
        sample_index_list = []

        for dev_idx in dev_range:
            num_pkt = np.count_nonzero(label == dev_idx)
            pkt_range = np.arange(0, num_pkt, dtype=int)
            sample_index_dev = np.where(label == dev_idx)[0][
                pkt_range].tolist()
            sample_index_list.extend(sample_index_dev)
            print('Dev ' + str(dev_idx + 1) + ' have ' + str(num_pkt) + ' packets.')

        sample_index_list = np.array(sample_index_list, dtype=np.int64)
        sample_index_list = np.unique(sample_index_list)
        sample_index_list.sort()
        data = f[self.dataset_name][sample_index_list]
        data = self._convert_to_complex(data)
        label = label[sample_index_list]
        f.close()
        return data, label


def read_train_data(file_path='/data02/lpl/github/SH_hybrid/data/9ft_train.h5',
              dev_range = np.arange(0, 7, dtype=int)):

    data_stft_all = []
    y_all = []

    LoadDatasetObj = LoadDataset()

    data_cx, y_ch0 = LoadDatasetObj.load_iq_samples(file_path,
                                                 dev_range)

    data_clean = np.stack([rms_normalize_per_sample(x) for x in data_cx], axis=0)
    data_ch0 = add_awgn_batch(data_clean, split='train')

    STFTSpectrogramObj = STFTSpectrogram()

    data_stft = STFTSpectrogramObj.STFT_spectrogram(data_ch0)
    data_stft_all.append(data_stft)
    y_all.append(y_ch0)

    data_stft_all = np.concatenate(data_stft_all, axis=0)
    y_all = np.concatenate(y_all)
    X_train, X_val, Y_train, Y_val = train_test_split(data_stft_all, y_all, test_size=0.2, random_state=32)
    return X_train, X_val, Y_train, Y_val

def read_test_data(file_path='/data02/lpl/github/SH_hybrid/data/9ft_test.h5',
              dev_range = np.arange(0, 7, dtype=int)):
    data_stft_all = []
    y_all = []
    LoadDatasetObj = LoadDataset()

    data_cx, y_ch0 = LoadDatasetObj.load_iq_samples(file_path,
                                                     dev_range)

    data_clean = np.stack([rms_normalize_per_sample(x) for x in data_cx], axis=0)
    data_ch0 = add_awgn_batch(data_clean, split='test')
    STFTSpectrogramObj = STFTSpectrogram()
    data_stft = STFTSpectrogramObj.STFT_spectrogram(data_ch0)
    data_stft_all.append(data_stft)
    y_all.append(y_ch0)
    X_test = np.concatenate(data_stft_all, axis=0)
    Y_test = np.concatenate(y_all)
    return X_test, Y_test
if __name__ == "__main__":
    X_train, X_val, Y_train, Y_val = read_train_data()
    print(X_train.shape)
    X_test, Y_test = read_test_data()







