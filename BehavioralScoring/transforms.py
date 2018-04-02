#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Time-frequency transformations
"""

import abc
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# -----------------------------------------------------------------------------
# custom wavelet classes
# -----------------------------------------------------------------------------


class Wavelet(object):

    __meta__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def generate(self):
        pass

    @abc.abstractmethod
    def is_complex(self):
        return

    def plot(self, ax=None, show=True):

        if ax is None:
            ax = plt.gca()

        phi, t = self.generate()
        ax.plot(t, np.real(phi), 'k-')
        ax.plot(t, np.imag(phi), 'r-')

        if show:
            plt.show()


class Morlet(Wavelet):
    """Complex-valued Morlet wavelet"""

    def __init__(self, sigma=5., lb=-4, ub=4., width=100):

        self.sigma = sigma
        self.lb = lb
        self.ub = ub
        self.width = width

    def is_complex(self):

        return True

    def generate(self):

        sig = self.sigma

        t = np.linspace(self.lb, self.ub, self.width)

        c_sigma = (1 + np.exp(-sig ** 2) - 2*np.exp(-3/4.*sig**2)) ** (-.5)
        phi = c_sigma * np.pi**(-.25) * np.exp(-.5*t**2) * \
            (np.exp(1j*sig*t) - np.exp(-.5*sig**2))

        return phi, t


class CWT(object):

    def __init__(self, wavelet):

        self.wavelet = wavelet

    def process(self, x, widths):

        wl = self.wavelet

        N = x.shape[0]
        n_scales = len(widths)

        if wl.is_complex():
            dtype = np.complex
        else:
            dtype = np.float
        X = np.zeros((N, n_scales), dtype=dtype)

        for i, width in enumerate(widths):

            wl.width = width
            phi, _ = wl.generate()

            if wl.is_complex():
                xr = signal.convolve(x, np.real(phi), mode='same')
                xi = signal.convolve(x, np.imag(phi), mode='same')
                X[:, i] = xr + 1j*xi
            else:
                X[:, i] = signal.convolve(x, phi, mode='same')

        return X
