################################################################
#                                                              #    
#                                                              #
# Macallyster S. Edmondson                                     #
#                                                              #
# ECE351-53                                                    #
#                                                              #
# Lab #8                                                       #
#                                                              #
# 03/08/2022                                                   #
#                                                              #
#                                                              #
################################################################

# import sigsyslib as ss
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as spfft

#GLOBAL
step_size = 1e-2

# PART 1

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

# PART 1

t = np.arange(0, 2, step_size)
y = [np.cos(2*np.pi*t), 5*np.sin(2*np.pi*t),
     2*np.cos((2*np.pi*2*t)-2)+(np.sin((2*np.pi*6*t) + 3))**2]

for i in range(len(y)):
    freq, X_mag, X_phi = myFFT(y[i], 100)
    
    plt.figure(figsize = (15, 17))
    plt.subplot(3, 1, 1)
    plt.plot(t, y[i], "b-")
    plt.grid()
    plt.ylabel('x(t)')
    plt.xlabel('t')
    plt.title('Task ' + str(i+1) + ' - User-Defined FFT of x(t)')
    plt.subplot(3, 2, 3)
    plt.stem(freq, X_mag, "b-")
    plt.grid()
    plt.ylabel('|X(f)|')
    plt.subplot(3, 2, 4)
    plt.stem(freq, X_mag, "b-")
    if(i > 1) :
        plt.xlim(-20, 20)
    else:
        plt.xlim(-2, 2)
    plt.grid()
    plt.subplot(3, 2, 5)
    plt.stem(freq, X_phi, "b-")
    plt.grid()
    plt.ylabel('/_X(f)')
    plt.xlabel('f [Hz]')
    plt.subplot(3, 2, 6)
    plt.stem(freq, X_phi, "b-")
    if(i > 1) :
        plt.xlim(-20, 20)
    else:
        plt.xlim(-2, 2)
    plt.grid()
    plt.xlabel('f [Hz]')
    
for i in range(len(y)):
    freq, X_mag, X_phi = myFFT(y[i], 100, clean=True)
    
    plt.figure(figsize = (15, 17))
    plt.subplot(3, 1, 1)
    plt.plot(t, y[i], "b-")
    plt.grid()
    plt.ylabel('x(t)')
    plt.xlabel('t')
    plt.title('Task 1 - User-Defined FFT of x(t)=cos(2[pi]t)')
    plt.subplot(3, 2, 3)
    plt.stem(freq, X_mag, "b-")
    plt.grid()
    plt.ylabel('|X(f)|')
    plt.subplot(3, 2, 4)
    plt.stem(freq, X_mag, "b-")
    if(i > 1) :
        plt.xlim(-20, 20)
    else:
        plt.xlim(-2, 2)
    plt.grid()
    plt.subplot(3, 2, 5)
    plt.stem(freq, X_phi, "b-")
    plt.grid()
    plt.ylabel('/_X(f)')
    plt.xlabel('f [Hz]')
    plt.subplot(3, 2, 6)
    plt.stem(freq, X_phi, "b-")
    if(i > 1) :
        plt.xlim(-20, 20)
    else:
        plt.xlim(-2, 2)
    plt.grid()
    plt.xlabel('f [Hz]')
    

#4

#1
def b(k) :
    b_k = (2/(k*np.pi))*(1-np.cos(k*np.pi))
    return b_k

#2
T = 8;
w0 = 2*np.pi/T
def x(t, N) :
    x = 0
    for k in range(1, N+1, 1) :
        x += b(k) * np.sin(k*w0*t)
    return x

# fList = [1, 3, 15, 50, 150, 1500]

# for i in fList :
#     t = np.arange(0, 20 + step_size, step_size)
#     y1 = x(t, i)
    
#     plt.figure(figsize = (10, 11))
#     plt.subplot(1, 1, 1)
#     plt.plot(t, y1, "b-")
#     plt.grid()
#     plt.ylabel('x(t) [N = ' + str(i) + ']')
#     plt.xlabel('t')
#     plt.title('Fourier Series Output for N = ' + str(i))
    
t = np.arange(0, 16, step_size)
y = x(t, 15)

freq, X_mag, X_phi = myFFT(y, 100, clean=True)

plt.figure(figsize = (15, 17))
plt.subplot(3, 1, 1)
plt.plot(t, y, "b-")
plt.grid()
plt.ylabel('x(t)')
plt.xlabel('t')
plt.title('Task ' + str(4) + ' - User-Defined FFT of x(t)')
plt.subplot(3, 2, 3)
plt.stem(freq, X_mag, "b-")
plt.grid()
plt.ylabel('|X(f)|')
plt.subplot(3, 2, 4)
plt.stem(freq, X_mag, "b-")
plt.xlim(-3, 3)
plt.grid()
plt.subplot(3, 2, 5)
plt.stem(freq, X_phi, "b-")
plt.grid()
plt.ylabel('/_X(f)')
plt.xlabel('f [Hz]')
plt.subplot(3, 2, 6)
plt.stem(freq, X_phi, "b-")
plt.xlim(-3, 3)
plt.grid()
plt.xlabel('f [Hz]')

# #1
# def b(k) :
#     b_k = (2/(k*np.pi))*(1-np.cos(k*np.pi))
#     return b_k

# # a_k = 0; Thus is not defined.

# print("a_k & b_k for 4 values.")
# for i in range(1, 5, 1) :
#     print("a_", i, "=", 0, " ; ", "b_", i, "=", b(i))

# #2
# T = 8;
# w0 = 2*np.pi/T
# def x(t, N) :
#     x = 0
#     for k in range(1, N+1, 1) :
#         x += b(k) * np.sin(k*w0*t)
#     return x

# # fList = [1, 3, 15, 50, 150, 1500]
# fList = [10]

# for i in fList :
#     t = np.arange(0, 20 + step_size, step_size)
#     y1 = x(t, i)
    
#     plt.figure(figsize = (10, 11))
#     plt.subplot(1, 1, 1)
#     plt.plot(t, y1, "b-")
#     plt.grid()
#     plt.ylabel('x(t) [N = ' + str(i) + ']')
#     plt.xlabel('t')
#     plt.title('Fourier Series Output for N = ' + str(i))