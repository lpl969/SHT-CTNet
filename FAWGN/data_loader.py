#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import h5py
import numpy as np
from numpy.random import uniform
from scipy import signal
from sklearn.model_selection import train_test_split
import torch




def rms_normalize_per_sample(x: np.ndarray, eps=1e-12) -> np.ndarray:
    rms = np.sqrt(np.mean(np.abs(x)**2) + eps)
    return x / max(rms, eps)



def make_uca_positions(C: int, radius: float, z: float = 0.0) -> np.ndarray:
    idx = np.arange(C, dtype=np.float64)
    ang = 2.0 * np.pi * idx / float(C)
    x = radius * np.cos(ang)
    y = radius * np.sin(ang)
    z = np.full_like(x, fill_value=z, dtype=np.float64)
    xyz = np.stack([x, y, z], axis=1)
    return xyz

def simulate_array_freq_phase_only(
    X_F_T: np.ndarray,
    xyz: np.ndarray,
    doa_theta: float,
    doa_phi: float,
    carrier_hz: float = 2.4435e9,
    wave_speed: float = 3.0e8,
) -> np.ndarray:
    C = xyz.shape[0]
    F, T = X_F_T.shape

    ux = np.sin(doa_theta) * np.cos(doa_phi)
    uy = np.sin(doa_theta) * np.sin(doa_phi)
    uz = np.cos(doa_theta)
    u = np.array([ux, uy, uz], dtype=np.float64)

    k = 2.0 * np.pi * (carrier_hz / wave_speed)
    phase_c = np.exp(1j * k * (xyz @ u))[:, None, None]

    Xc = (phase_c * X_F_T[None, :, :]).astype(np.complex64)
    return Xc

def add_awgn_mics_freq(Xc: np.ndarray, snr_db: float) -> np.ndarray:
    C, F, T = Xc.shape
    sig_pow = np.mean(np.abs(Xc)**2)
    snr_lin = 10.0 ** (snr_db / 10.0)
    n0 = sig_pow / max(snr_lin, 1e-12)
    noise = np.sqrt(n0/2.0) * (np.random.randn(C, F, T) + 1j * np.random.randn(C, F, T))
    return (Xc + noise.astype(np.complex64)).astype(np.complex64)

def pick_snr(split: str, snr_train_range_db=(0.0, 10.0), snr_eval_range_db=(-5.0, 20.0)) -> float:
    lo, hi = snr_train_range_db if split == 'train' else snr_eval_range_db
    return float(uniform(lo, hi))
    
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

    def gen_stft_complex(self, sig, win_len=256, overlap=128):
        _, _, spec = signal.stft(
            sig,
            window='boxcar',
            nperseg=win_len,
            noverlap=overlap,
            nfft=win_len,
            return_onesided=False,
            padded=False,
            boundary=None
        )
        spec = np.fft.fftshift(spec, axes=0)
        return spec

    def gen_stft_spectrogram(self, sig, win_len=256, overlap=128):
        spec = self.gen_stft_complex(sig, win_len=win_len, overlap=overlap)

        chan_ind_spec_amp = np.log10(np.abs(spec) ** 2 + 0.0).astype(np.float32)
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




def build_dataset_from_h5(
    file_path: str,
    split: str,
    dev_range=np.arange(0,7,dtype=int),


    sample_rate_hz: float = 50e6,
    n_fft=256, hop=128,


    C_mic: int = 8,
    radius: float = 0.07,


    carrier_hz: float = 2.4435e9,
    wave_speed: float = 3.0e8,


    doa_theta: float = np.pi/2,
    doa_phi: float = 0.0,
    random_doa_train: bool = True,
    doa_theta_jitter_deg: float = 5.0,
    snr_train_range_db=(0.0, 10.0),
    snr_eval_range_db=(-5.0, 20.0),
):
    LD = LoadDataset()
    data_cx, Y = LD.load_iq_samples(file_path, dev_range)
    Y = Y.astype(np.int64)


    data_norm = np.stack([rms_normalize_per_sample(x) for x in data_cx], axis=0)



    stft = STFTSpectrogram()


    xyz = make_uca_positions(C=C_mic, radius=radius, z=0.0)

    X_list = []
    for i in range(data_norm.shape[0]):

        if random_doa_train and split == 'train':
            doa_phi_i = float(uniform(0.0, 2.0*np.pi))
            jitter = np.deg2rad(doa_theta_jitter_deg)
            doa_theta_i = float(np.clip(doa_theta + uniform(-jitter, jitter), 0.0, np.pi))
        else:
            doa_phi_i = float(doa_phi)
            doa_theta_i = float(doa_theta)



        X = stft.gen_stft_complex(data_norm[i], win_len=n_fft, overlap=hop)

        Xc = simulate_array_freq_phase_only(
                X, xyz, doa_theta=doa_theta_i, doa_phi=doa_phi_i,
                carrier_hz=carrier_hz, wave_speed=wave_speed)
        snr_db = pick_snr(split, snr_train_range_db, snr_eval_range_db)
        _Xc_noisy = add_awgn_mics_freq(Xc, snr_db=snr_db)
        X_used = _Xc_noisy.mean(axis=0)
        S = np.log10(np.abs(X_used) ** 2 + 0.0).astype(np.float32)
        S = stft._spec_crop(S) 
        X_list.append(S[None, ...])


    X = np.stack(X_list, axis=0).astype(np.float32)
    return X, Y


def read_train_data(file_path, **kwargs):
    X, Y = build_dataset_from_h5(file_path, split='train', **kwargs)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=32, stratify=Y
    )
    return X_train, X_val, Y_train, Y_val

def read_test_data(file_path, **kwargs):
    X, Y = build_dataset_from_h5(file_path, split='test', **kwargs)
    return X, Y
