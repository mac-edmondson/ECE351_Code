################################################################
#                                                              #    
#                                                              #
# Macallyster S. Edmondson                                     #
#                                                              #
# ECE351-53                                                    #
#                                                              #
# Lab #10                                                      #
#                                                              #
# 03/29/2022                                                   #
#                                                              #
#                                                              #
################################################################

# import sigsyslib as ss
import zplane
import control as con
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as spsig

#GLOBAL
step_size = 1e3

## PART 1 ##
# 3
num = [2, -40]
den = [1, -10, 16]
r, p, k = spsig.residuez(num, den)
print("Partial Fraction Results")
print("Residues: " + str(r))
print("Poles: " + str(p))
print("Coefficents: " + str(k))

# 4
zplane.zplane(num, den)

#5
w, h = spsig.freqz(num, den, whole=True)

plt.figure(figsize = (10, 11))
plt.subplot(2, 1, 1)
plt.plot(w/np.pi, 20*np.log10(abs(h)), "b-")
plt.grid(True, which='both', ls='-')
plt.ylabel('Magnitude [dB]')
plt.title('Magnitude and Phase of H(z)')
plt.subplot(2, 1, 2)
plt.grid(True, which='both', ls='-')
plt.plot(w/np.pi, np.angle(h, deg=True), "b-")
plt.ylabel('angle(H(S)) [deg]')
plt.xlabel('Frequency [pi/sample]')

plt.figure(figsize = (10, 11))
plt.subplot(2, 1, 1)
plt.semilogx(w/2/np.pi, 20*np.log10(abs(h)), "b-")
plt.grid(True, which='both', ls='-')
plt.ylabel('Magnitude [dB]')
plt.title('Magnitude and Phase of H(z)')
plt.subplot(2, 1, 2)
plt.grid(True, which='both', ls='-')
plt.semilogx(w/2/np.pi, np.angle(h, deg=True), "b-")
plt.ylabel('angle(H(S)) [deg]')
plt.xlabel('Frequency [1/sample]')

# # PART 1

# R = 1e3 # Ohms
# L = 27e-3 # H
# C = 100e-9 # F

# # 1
# def mag_db(omega) :
#     out = (((1/(R*C))*omega)/(np.sqrt((w**4) + ((((1/(R*C))**2) - (2/(L*C))) * (omega**2)) + ((1/(L*C))**2))))
#     out_db = 20 * np.log10(out)
#     return out_db

# def ang_deg(omega) :
#     ang = np.pi/2 - np.arctan(((1/(R*C))*omega)/((1/(L*C))-(omega**2)))
#     ang_deg = (180/np.pi) * ang
#     return ang_deg

# w = np.arange(1e3, 1e6 + step_size, step_size)
# mag = mag_db(w)
# ang = ang_deg(w)

# for i in range(len(ang)):
#     if(ang[i] > 90):
#         ang[i] -= 180

# plt.figure(figsize = (10, 11))
# plt.subplot(2, 1, 1)
# plt.semilogx(w, mag, "b-")
# plt.grid(True, which='both', ls='-')
# plt.ylabel('|H(S)| [dB]')
# plt.title('Magnitude')
# plt.subplot(2, 1, 2)
# plt.grid(True, which='both', ls='-')
# plt.semilogx(w, ang, "b-")
# plt.ylabel('angle(H(S)) [deg]')
# plt.xlabel('w [rad/s]')

# # 2
# num = [1/(R*C), 0]
# den = [1, 1/(R*C), 1/(L*C)]

# sys = con.TransferFunction(num, den)

# plt.figure(figsize=(10,11))
# _ = con.bode(sys, omega=None, dB = True, Hz = False, deg = True, Plot = True)

# plt.figure(figsize=(10, 11))
# _ = con.bode(sys, omega=None, dB = True, Hz = True, deg = True, Plot = True)

# # PART 2
# fs = 2*np.pi*50000
# step_size = 1/fs

# t = np.arange(0, 0.01 + step_size, step_size)
# x = np.cos(2*np.pi*100*t) + np.cos(2*np.pi*3024*t) + np.sin(2*np.pi*50000*t)
# plt.figure(figsize=(10, 11))
# plt.subplot(1, 1, 1)
# plt.plot(t, x, "b-")
# plt.grid()
# plt.ylabel('x(t)')
# plt.xlabel('t')
# plt.title('Part 2 Task 1')

# z, p= spsig.bilinear(num, den, fs=fs)
# y = spsig.lfilter(z, p, x)

# plt.figure(figsize=(10, 11))
# plt.subplot(1, 1, 1)
# plt.plot(t, y, "b-")
# plt.grid()
# plt.ylabel('y(t)')
# plt.xlabel('t')
# plt.title('Part 2 Task 4')