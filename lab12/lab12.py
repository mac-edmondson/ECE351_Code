################################################################
#                                                              #    
#                                                              #
# Macallyster S. Edmondson                                     #
#                                                              #
# ECE351-53                                                    #
#                                                              #
# Lab #12                                                      #
#                                                              #
# 04/12/2022                                                   #
#                                                              #
#                                                              #
################################################################

# import sigsyslib as ss
import zplane
# import control as con
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as spsig
import scipy.fftpack as spfft
import pandas as pd

# Get data from CSV into arrays
df = pd.read_csv('NoisySignal.csv')

t = df['0'].values
sensor_sig = df['1'] .values

# Plot given signal
plt.figure(figsize = (10, 7))
plt.plot(t, sensor_sig)
plt.grid()
plt.title('Noisy Input Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [V]')
plt.show()

## Task 1 ##

step_size = t[1] - t[0]; # Get time step of input signal
print("Step Size is: " + str(step_size))
freq_sample = 1/step_size

# Perfom FFT (Imported from Lab9)
def myFFT(x, fs, clean=False):
    N = len(x)
    X_fft = spfft.fft(x)
    X_fft_shifted = spfft.fftshift(X_fft)
    
    freq = np.arange(-N/2, N/2)*fs/N
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    if(clean):
        for i in range(0, len(X_phi)):
            if (X_mag[i] < (1e-10)) :
                X_phi[i] = 0
        
    return freq, X_mag, X_phi

freq, X_mag, X_phi = myFFT(sensor_sig, freq_sample, clean=False)

plt.figure(figsize = (15, 17))
plt.subplot(3, 1, 1)
plt.plot(t, sensor_sig, "b-")
plt.grid()
plt.ylabel('x(t)')
plt.xlabel('t')
plt.title('FFT of Input Signal, x(t)')
plt.subplot(3, 2, 3)
plt.stem(freq, X_mag, "b-")
plt.grid()
plt.ylabel('|X(f)|')
plt.subplot(3, 2, 4)
plt.stem(freq, X_mag, "b-")
plt.xlim(0, 100e3)
# if(i > 1) :
#     plt.xlim(-20, 20)
# else:
#     plt.xlim(-2, 2)
plt.grid()
plt.subplot(3, 2, 5)
plt.stem(freq, X_phi, "b-")
plt.grid()
plt.ylabel('/_X(f)')
plt.xlabel('f [Hz]')
plt.subplot(3, 2, 6)
plt.stem(freq, X_phi, "b-")
plt.xlim(0, 100e3)
# if(i > 1) :
#     plt.xlim(-20, 20)
# else:
#     plt.xlim(-2, 2)
plt.grid()
plt.xlabel('f [Hz]')


# #GLOBAL
# step_size = 1e3

# ## PART 1 ##
# # 3
# num = [2, -40]
# den = [1, -10, 16]
# r, p, k = spsig.residuez(num, den)
# print("Partial Fraction Results")
# print("Residues: " + str(r))
# print("Poles: " + str(p))
# print("Coefficents: " + str(k))

# # 4
# zplane.zplane(num, den)

# #5
# w, h = spsig.freqz(num, den, whole=True)

# plt.figure(figsize = (10, 11))
# plt.subplot(2, 1, 1)
# plt.plot(w/np.pi, 20*np.log10(abs(h)), "b-")
# plt.grid(True, which='both', ls='-')
# plt.ylabel('Magnitude [dB]')
# plt.title('Magnitude and Phase of H(z)')
# plt.subplot(2, 1, 2)
# plt.grid(True, which='both', ls='-')
# plt.plot(w/np.pi, np.angle(h, deg=True), "b-")
# plt.ylabel('angle(H(S)) [deg]')
# plt.xlabel('Frequency [pi/sample]')

# plt.figure(figsize = (10, 11))
# plt.subplot(2, 1, 1)
# plt.semilogx(w/2/np.pi, 20*np.log10(abs(h)), "b-")
# plt.grid(True, which='both', ls='-')
# plt.ylabel('Magnitude [dB]')
# plt.title('Magnitude and Phase of H(z)')
# plt.subplot(2, 1, 2)
# plt.grid(True, which='both', ls='-')
# plt.semilogx(w/2/np.pi, np.angle(h, deg=True), "b-")
# plt.ylabel('angle(H(S)) [deg]')
# plt.xlabel('Frequency [pi/sample]')