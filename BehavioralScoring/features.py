#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Feature extraction
"""


import abc
import numpy as np
from collections import OrderedDict


# -----------------------------------------------------------------------------
# feature extraction
# -----------------------------------------------------------------------------

class FeatureExtractor(object):
    """sklearn-like feature extractor base class"""

    __meta__ = abc.ABCMeta

    @abc.abstractmethod
    def fit(self, X):
        return

    @abc.abstractmethod
    def transform(self, X):
        return


def segment(data, seg_len, shift, zero_padding=True):

    total_len = data.shape[0]

    if zero_padding is True:
        num_seg = np.int(np.ceil((total_len - seg_len + shift) / shift))
    else:
        num_seg = np.int(np.floor((total_len - seg_len + shift) / shift))

    out = np.zeros([num_seg, seg_len])
    for i in range(num_seg):
        i0 = i * shift
        i1 = i0 + seg_len
        if i1 <= data.shape[0]:
            out[i, :] = data[i0:i1]

        else:
            j1 = data.shape[0] - i0
            out[i, :j1] = data[i0:]

    return out


def compute_spectrogram(x,
                        winlen=256,
                        shift=128,
                        nfft=256,
                        samplerate=1000.):

        win = np.hanning(winlen)
        win /= np.sum(win*win)

        if winlen % 2:
            nwin_half = int(np.floor(winlen/2.) + 1)
        else:
            nwin_half = winlen/2

        n_start = nwin_half - 1
        n_end = nwin_half

        x_begin = np.zeros((n_start,))
        x_end = np.zeros((n_end,))
        x = np.concatenate((x_begin, x, x_end))

        # The fast way ...
        X = np.multiply(win, segment(x, winlen, shift))
        S = np.abs(np.fft.rfft(X, n=nfft, axis=1)) ** 2

        # Compenstate for power of omitted negative frequencies except for
        # DC and Nyquist frequencies
        S[:, 1:-1] *= 2.

        if nfft % 2:
            n_freqs = (nfft + 1)//2
        else:
            n_freqs = nfft//2 + 1

        f_all = np.fft.fftfreq(nfft, 1./samplerate)
        f = np.abs(f_all[:n_freqs])

        t = 1./samplerate * np.arange(0, S.shape[0]*shift, shift)

        return S, t, f


class TemporalFeatureExtractor(FeatureExtractor):

    def __init__(self, maxlag=0.3, stepsize=0.05, samplerate=100.,
                 center=False):

        self.maxlag = maxlag
        self.stepsize = stepsize
        self.samplerate = samplerate
        self.center = center

    def fit(self, X, y):

        return self

    def transform(self, X, verbose=True):

        N, n_channels = X.shape

        fs = self.samplerate

        if not self.center:
            # align to end of time window
            lags = np.arange(0, self.maxlag+1e-6, self.stepsize) * fs
            lag_ind = -1*np.round(lags).astype(np.int)[::-1]
            n = len(lag_ind)
            n_lag = -lag_ind[0]

            XX = np.zeros((N, n_channels * n))
            for i in range(n_channels):

                for j in range(n_lag, N):
                    XX[j, i*n:(i+1)*n] = X[lag_ind+j, i]

        else:
            # align to center of time window
            lags = np.arange(-self.maxlag/2., self.maxlag/2.+1e-6,
                             self.stepsize) * fs
            lag_ind = np.round(lags).astype(np.int)
            n = len(lag_ind)
            n_lag = lag_ind[-1]

            XX = np.zeros((N, n_channels * n))
            for i in range(n_channels):

                for j in range(n_lag, N-n_lag):
                    XX[j, i*n:(i+1)*n] = X[lag_ind+j, i]

        if verbose:
            print "Time-lagged feature dimensionality: %d x %d" % XX.shape

        return XX


class STFTFeatureExtractor(FeatureExtractor):

    def __init__(self, winlen=256, nfft=256, shift=1, samplerate=100.,
                 f_lower=1., f_upper=50., dynamic_range=60.):

        self.winlen = winlen
        self.nfft = nfft
        self.shift = shift
        self.samplerate = samplerate
        self.f_lower = f_lower
        self.f_upper = f_upper
        self.dynamic_range = dynamic_range

    def get_frequencies(self):

        if self.nfft % 2:
            n_freqs = (self.nfft + 1)//2
        else:
            n_freqs = self.nfft//2 + 1

        f_all = np.fft.fftfreq(self.nfft, 1./self.samplerate)
        f = np.abs(f_all[:n_freqs])

        f = f[np.logical_and(f >= self.f_lower,
                             f <= self.f_upper)]

        return f

    def fit(self, X, y):

        return self

    def transform(self, X, verbose=True):

        nfft = self.nfft
        fs = self.samplerate
        shift = self.shift
        winlen = self.winlen

        n_channels = X.shape[1]

        Y = []
        for i in range(n_channels):

            if verbose:
                print("transforming channel {}/{}".format(i+1, n_channels))

            Pxx, t, f = compute_spectrogram(X[:, i],
                                            winlen=winlen,
                                            shift=shift,
                                            nfft=nfft,
                                            samplerate=fs)

            valid = np.logical_and(f >= self.f_lower,
                                   f <= self.f_upper)
            Pxx = Pxx[:, valid].T
            f = f[valid]

            if 1:
                dyn_range = self.dynamic_range
                Pxx = 10 * np.log10(Pxx)
                P_max = np.max(Pxx)
                Pxx[Pxx < P_max - dyn_range] = P_max - dyn_range
                Pxx[np.isnan(Pxx)] = P_max - dyn_range
                Pxx[np.isinf(Pxx)] = P_max - dyn_range

            Y.append(Pxx)

        Y = np.asarray(Y).T
        YY = np.reshape(Y, (Y.shape[0], n_channels * Y.shape[1]))

        if verbose:
            print("{} features x {} observations".format(YY.shape[1],
                                                         YY.shape[0]))

        return YY


class CWTFeatureExtractor(FeatureExtractor):

    def __init__(self, n_scales=20, scale_range=(0.1, 2.),
                 samplerate=100.):

        self.n_scales = n_scales
        self.scale_range = scale_range
        self.samplerate = samplerate

    def get_widths(self):

        raise NotImplementedError("no time to do this ...")

    def get_frequencies(self):

        return 1. / self.get_widths()

    def fit(self, X, y):

        return self

    def transform(self, X, verbose=True):

        from matplotlib import mlab

        nfft = self.nfft
        fs = self.samplerate
        shift = self.shift

        n_channels = X.shape[1]
        X = np.concatenate((np.zeros((nfft/2, n_channels)),
                            X,
                            np.zeros((nfft/2 - 1, n_channels))))

        Y = []
        for i in range(n_channels):

            print("transforming channel {}/{}".format(i+1, n_channels))

            Pxx, f, t = mlab.specgram(X[:, i],
                                      NFFT=nfft, Fs=fs,
                                      noverlap=(nfft - shift))

            valid = np.logical_and(f >= self.f_lower,
                                   f <= self.f_upper)
            Pxx = Pxx[valid, :]
            f = f[valid]

            if 1:
                dyn_range = 60.
                Pxx = 10 * np.log10(Pxx)
                P_max = np.max(Pxx)
                Pxx[Pxx < P_max - dyn_range] = P_max - dyn_range
                Pxx[np.isnan(Pxx)] = P_max - dyn_range
                Pxx[np.isinf(Pxx)] = P_max - dyn_range

            Y.append(Pxx)

        Y = np.asarray(Y)
        YY = np.reshape(Y, (n_channels * Y.shape[1], Y.shape[2])).T

        if verbose:
            print("{} features x {} observations".format(YY.shape[1],
                                                         YY.shape[0]))

        return YY


# -----------------------------------------------------------------------------
# feature manipulation
# -----------------------------------------------------------------------------

class FeatureLagger(FeatureExtractor):
    """create (time-)lagged features"""

    def __init__(self, maxlag=0.5, stepsize=0.1, samplerate=1000.):

        self.maxlag = maxlag
        self.stepsize = stepsize
        self.samplerate = samplerate

    def fit(self, X, y):

        return self

    def transform(self, X, verbose=True):

        step_size = int(np.round(self.stepsize * self.samplerate))
        n_steps = int(np.round(float(self.maxlag) / self.stepsize)) + 1

        N, D = X.shape
        Y = np.zeros((N, D * n_steps))
        for i in range(n_steps*step_size, N):

            ind = i + np.arange(n_steps * step_size)
            Y[i, :] = X[ind, :].ravel()

        if verbose:
            print "Time-lagged feature dimensionality: %d x %d" % Y.shape

        return Y


# -----------------------------------------------------------------------------
# feature container
# -----------------------------------------------------------------------------


class Features(object):
    """simple feature container class to append multiple types of features"""

    def __init__(self, N):

        self.N = N
        self._data = OrderedDict()

    def add(self, name, X,
            maxlag=None,
            samplerate=None,
            label=None,
            ticks=None,
            unit=None):

        if isinstance(X, dict):
            maxlag = X['maxlag']
            samplerate = X['samplerate']
            ticks = X['ticks']
            label = X['label']
            X = X['X']

        if X.shape[0] != self.N:
            raise ValueError('Number of observations {}!= N {}'.format(
                X.shape[0], self.N))

        self._data[name] = {'X': X,
                            'maxlag': maxlag,
                            'samplerate': samplerate,
                            'ndim': X.shape[1],
                            'label': label,
                            'ticks': ticks,
                            'unit': unit}

    def as_matrix(self, normalize=False):

        X = None

        n_dims = sum([self._data[k]['ndim'] for k in self._data])
        if n_dims > 0:
            X = np.zeros((self.N, n_dims))
            col = 0
            for i, name in enumerate(self._data):

                D = self._data[name]
                X[:, col:col+D['ndim']] = D['X']
                col += D['ndim']

        if normalize:
            # TODO: add standard normalization procedure ...
            pass

        return X

    @property
    def names(self):

        return self._data.keys()

    def get_indices(self, name):

        n_dims = [self._data[k]['ndim'] for k in self._data]
        index = self._data.keys().index(name)

        if index > 0:
            i1 = np.sum(n_dims[:index])
        else:
            i1 = 0

        i2 = i1 + n_dims[index]
        ind = np.arange(i1, i2)

        return ind

    def get_ticks(self, name):

        return self._data[name]['ticks']

    def get_unit(self, name):

        return self._data[name]['unit']

    def get_label(self, name):

        return self._data[name]['label']
