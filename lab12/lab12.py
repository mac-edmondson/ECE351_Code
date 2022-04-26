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
# import zplane
# import control as con
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as spsig
import scipy.fftpack as spfft
import pandas as pd
import control as con

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

plt.grid()
plt.subplot(3, 2, 5)
plt.stem(freq, X_phi, "b-")
plt.grid()
plt.ylabel('/_X(f)')
plt.xlabel('f [Hz]')
plt.subplot(3, 2, 6)
plt.grid()
plt.stem(freq, X_phi, "b-")
plt.xlim(0, 100e3)

plt.figure(figsize = (15,17))
plt.subplot(1, 1, 1)
plt.stem(freq, X_mag, "b-")
plt.xticks(np.arange(0, 3e3 + 100, 100))
plt.xlim(0, 3e3)
plt.title("Unfiltered Signal Magnitudes [0 to 3k Hz]")
plt.grid()
plt.ylabel('Mag.')
plt.xlabel('f [Hz]')

C = 3.5 * 1e-6
L = 2.01 * 1e-3
R = 9.4

num = [(R/L), 0]
den = [1, (R/L), (1/(L*C))]

sys = con.TransferFunction(num, den)
syssig = spsig.lti(num, den)

step_size = 1e2
w = np.arange(1, (1e6 * 2*np.pi)+step_size, step_size)
bodeW, bodeMag, bodePhase = spsig.bode(syssig, w)

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

step_size = t[1] - t[0];
fs = 1/step_size

z, p= spsig.bilinear(num, den, fs=fs)
filtered_sig = spsig.lfilter(z, p, sensor_sig)

freq, X_mag, X_phi = myFFT(filtered_sig, freq_sample, clean=False)

plt.figure(figsize = (15, 17))
plt.subplot(3, 1, 1)
plt.plot(t, filtered_sig, "b-")
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

plt.grid()
plt.subplot(3, 2, 5)
plt.stem(freq, X_phi, "b-")
plt.grid()
plt.ylabel('/_X(f)')
plt.xlabel('f [Hz]')
plt.subplot(3, 2, 6)
plt.stem(freq, X_phi, "b-")
plt.xlim(0, 100e3)

plt.grid()
plt.xlabel('f [Hz]')

plt.figure(figsize = (15,17))
plt.subplot(1, 1, 1)
plt.stem(freq, X_mag, "b-")
# plt.grid(True, which='both', ls='-')
plt.xticks(np.arange(0, 3e3 + 100, 100))
plt.xlim(0, 3e3)
plt.title("Filtered Signal Magnitudes [0 to 3k Hz]")
plt.grid()
plt.ylabel('Mag.')
plt.xlabel('f [Hz]')