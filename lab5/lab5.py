################################################################
#                                                              #    
#                                                              #
# Macallyster S. Edmondson                                     #
#                                                              #
# ECE351-53                                                    #
#                                                              #
# Lab #4                                                       #
#                                                              #
# 02/15/2022                                                   #
#                                                              #
# Any other necessary information needed to navigate the file  #
#                                                              #
#                                                              #
#                                                              #
################################################################

import sigsyslib as ss
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as spsig

#GLOBAL
step_size = 1e-6

# PART 1

# 1/2

def h(t) :
    h = 10e3*np.exp(-5000*t)*(np.cos(18584.1*t) - (0.269 * np.sin(18584.1*t)))
    return h

L = 27.0e-3   #H
C = 100.0e-9  #F
R = 1.0e3     #Ohms

system = ([0, L, 0], [R*C*L, L, R])

t = np.arange(0, 1.2e-3 + step_size, step_size)
y1 = h(t)        

tout, y2 = spsig.impulse(system, T=t);

plt.figure(figsize = (10, 11))
plt.subplot(2, 1, 1)
plt.plot(t, y1, "b-")
plt.grid()
plt.ylabel('h(t) [Hand Calc.]')
plt.title('h(t) v. t')

plt.subplot(2, 1, 2)
plt.plot(tout, y2, "r-")
plt.grid()
plt.ylabel('h(t) [Scipy Impulse]')
plt.xlabel('t')
plt.show()

# t2 = np.arange(10, 11 + step_size, step_size)
# tout, y = spsig.impulse(system, T=t2);

# plt.figure(figsize = (10, 11))
# plt.subplot(2, 1, 2)
# plt.plot(tout, y, "r-")
# plt.grid()
# plt.ylim(-1e-6, 1e-6)
# plt.ylabel('H(s)')
# plt.xlabel('t')
# plt.show()


# PART 2

# 1

tout, y = spsig.step(system, T=t)

plt.figure(figsize = (10, 11))
plt.subplot(1, 1, 1)
plt.plot(tout, y, "b-")
plt.grid()
plt.title('(h(t) * u(t)) v. t')
plt.xlabel('t')
plt.ylabel('Step Response of h(t)')
plt.show()