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
step_size = 1e-4

# PART 1

# 1/2

def y(t) :
    h = (1/2 - 1/2 * np.exp(-4*t) + np.exp(-6*t)) * ss.u(t)
    return h

system = ([1, 6, 12], [1, 10, 24])

t = np.arange(0, 2 + step_size, step_size)
y1 = y(t)        

tout, y2 = spsig.step(system, T=t);

plt.figure(figsize = (10, 11))
plt.subplot(2, 1, 1)
plt.plot(t, y1, "b-")
plt.grid()
plt.ylabel('y(t) [Hand Calc.]')
plt.title('y(t) = h(t)*u(t) v. t')

plt.subplot(2, 1, 2)
plt.plot(tout, y2, "r-")
plt.grid()
plt.ylabel('y(t) [Scipy Step]')
plt.xlabel('t')
plt.show()

#3

b = [0, 1, 6, 12]
a = [1, 10, 24, 0]

R, P, K = spsig.residue(b, a)

print("Part 1:")
print("Residue of Poles (same order as presented below): \n", R, "\n\nPoles in ascending order of mag.:\n", P, "\n\nCoefficent of Direct Polynomial Term: \n", K)

#PART 2

#1

b = [0,   0,   0,    0,   0,    0, 25250]
a = [1, 18, 218, 2036, 9085, 25250, 0]

R, P, K = spsig.residue(b, a)

print("Part 2:")
print("Residue of Poles (same order as presented below): \n", R, "\n\nPoles in ascending order of mag.:\n", P, "\n\nCoefficent of Direct Polynomial Term: \n", K)


#2
def cos_mthd(R, P, t):
    y = 0;
    for i in range(0, len(R)):
        k_mag = np.absolute(R[i])
        k_ang = np.angle(R[i])
        alpha = np.real(P[i])
        omega = np.imag(P[i])
        y += k_mag*np.exp(alpha*t)*np.cos(omega*t + k_ang)*ss.u(t)
    return y

def y_s1(t):
    y = (626.375/4)*np.exp(-3*t)*np.sin(4*t-np.deg2rad(156.615))*ss.u(t)
    return y
    
def y_s2(t):
    y = (186.75/10)*np.exp(-1*t)*np.sin(10*t - np.deg2rad(143.723))*ss.u(t)
    return y

def y_t(t):
    y = (1 * ss.u(t)) - (0.215*np.exp(-10*t)*ss.u(t)) + y_s1(t) + y_s2(t)
    return y

#3
system = ([0,   0,   0,    0,   0,  25250], [1, 18, 218, 2036, 9085, 25250])

t = np.arange(0, 4.5 + step_size, step_size)
# y1 = y_t(t) 
y1 = cos_mthd(R, P, t)       

tout, y2 = spsig.step(system, T=t);

plt.figure(figsize = (10, 11))
plt.subplot(2, 1, 1)
plt.plot(t, y1, "b-")
plt.grid()
plt.ylabel('y(t) [Hand Calc.]')
plt.title('y(t) = h(t)*u(t) v. t')

plt.subplot(2, 1, 2)
plt.plot(tout, y2, "r-")
plt.grid()
plt.ylabel('y(t) [Scipy Step]')
plt.xlabel('t')
plt.show()

