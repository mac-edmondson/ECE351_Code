#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 21:17:50 2022

@author: mac-edmondson
"""

# import sigsyslib as ss
import zplane
# import control as con
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fftpack as spfft
import pandas as pd
import control as con

C = 3.5 * 1e-6
L = 2.01 * 1e-3
R = 9.4

num = [(R/L), 0]
den = [1, (R/L), (1/(L*C))]

sys = con.TransferFunction(num, den)
syssig =sig.lti(num, den)

step_size = 1e2
w = np.arange(1, (1e6 * 2*np.pi)+step_size, step_size)
bodeW, bodeMag, bodePhase = sig.bode(syssig, w)

plt.figure(figsize=(11, 15))
plt.subplot(2, 1, 1)
plt.semilogx(bodeW/(2*np.pi), bodeMag)
plt.grid(True, which='both', ls='-')
plt.ylabel('Magnitude (dB)')
plt.yticks(np.arange(-90, 10, 10))
plt.title('Magnitude and Phase Bode Plot [spsig.bode]')
plt.xlabel('freq (rad/s)')
plt.show()

plt.figure(figsize=(10, 11))
plt.ylim(0, 10)
_ = con.bode(sys, omega=None, dB = True, Hz = True, deg = True, Plot = True)
plt.xlim(1.8e3, 2e3)
plt.subplot(2, 1, 1)
plt.ylim(-.5, 0)
plt.title('Magnitude and Phase Bode Plot [con.bode] (Hz)')
plt.show()

plt.figure(figsize=(10, 11))
_ = con.bode(sys, omega=None, dB = True, Hz = True, deg = True, Plot = True)
plt.xlim(0, 1e6)
plt.subplot(2, 1, 1)
plt.title('Magnitude and Phase Bode Plot [con.bode] (Hz)')
plt.show()

z, p= spsig.bilinear(num, den, fs=fs)
y = spsig.lfilter(z, p, x)

