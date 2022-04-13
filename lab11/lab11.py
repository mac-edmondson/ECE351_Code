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
# import control as con
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
plt.xlabel('Frequency [pi/sample]')