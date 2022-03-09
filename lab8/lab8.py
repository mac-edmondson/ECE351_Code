################################################################
#                                                              #    
#                                                              #
# Macallyster S. Edmondson                                     #
#                                                              #
# ECE351-53                                                    #
#                                                              #
# Lab #8                                                       #
#                                                              #
# 03/01/2022                                                   #
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
step_size = 1e-3

# PART 1

#1
def b(k) :
    b_k = (2/(k*np.pi))*(1-np.cos(k*np.pi))
    return b_k

# a_k = 0; Thus is not defined.

print("a_k & b_k for 4 values.")
for i in range(1, 5, 1) :
    print("a_", i, "=", 0, " ; ", "b_", i, "=", b(i))

#2
T = 8;
w0 = 2*np.pi/T
def x(t, N) :
    x = 0
    for k in range(1, N+1, 1) :
        x += b(k) * np.sin(k*w0*t)
    return x

t = np.arange(0, 20 + step_size, step_size)
y1 = x(t, 1)

plt.figure(figsize = (10, 11))
plt.subplot(1, 1, 1)
plt.plot(t, y1, "b-")
plt.grid()
plt.ylabel('x(t) [N = 1]')
plt.xlabel('t')
plt.title('Foureir Series Output for N = 1')

y1 = x(t, 3)

plt.figure(figsize = (10, 11))
plt.subplot(1, 1, 1)
plt.plot(t, y1, "b-")
plt.grid()
plt.ylabel('x(t) [N = 3]')
plt.xlabel('t')
plt.title('Foureir Series Output for N = 3')

y1 = x(t, 15)

plt.figure(figsize = (10, 11))
plt.subplot(1, 1, 1)
plt.plot(t, y1, "b-")
plt.grid()
plt.ylabel('x(t) [N = 15]')
plt.xlabel('t')
plt.title('Foureir Series Output for N = 15')

y1 = x(t, 50)

plt.figure(figsize = (10, 11))
plt.subplot(1, 1, 1)
plt.plot(t, y1, "b-")
plt.grid()
plt.ylabel('x(t) [N = 50]')
plt.xlabel('t')
plt.title('Foureir Series Output for N = 50')

y1 = x(t, 150)

plt.figure(figsize = (10, 11))
plt.subplot(1, 1, 1)
plt.plot(t, y1, "b-")
plt.grid()
plt.ylabel('x(t) [N = 150]')
plt.xlabel('t')
plt.title('Foureir Series Output for N = 150')

y1 = x(t, 1500)

plt.figure(figsize = (10, 11))
plt.subplot(1, 1, 1)
plt.plot(t, y1, "b-")
plt.grid()
plt.ylabel('x(t) [N = 1500]')
plt.xlabel('t')
plt.title('Foureir Series Output for N = 1500')

# print("### PART 1 ###\n")

# # 1
# def G(s) :
#     return (s+9)/((s^2-6*s-16)*(s+4))

# def A(s) :
#     return (s+4)/(s^2 + 4*s + 3)

# def B(s) :
#     return (s^2 + 26*s +168)

# # 2
# zG, pG, kG = spsig.tf2zpk([1, 9], [1, -2, -40, -64])
# zA, pA, kA = spsig.tf2zpk([1, 4], [1, 4, 3])
# # zB, pB, kB = np.roots(1, 26, 168)

# print("G(s): z = ", zG, "p = ", pG)
# print("A(s): z = ", zA, "p = ", pA)

# #3/4 (Open Loop HT = A*G)
# num = spsig.convolve([1, 9], [1, 4])
# den = spsig.convolve([1, -2, -40, -64], [1, 4, 3])
# print("\nOpen Loop Numerator: ", num)
# print("Open Loop Denominator: ", den)

# #5
# tout, y1 = spsig.step(spsig.lti(num, den))
# plt.figure(figsize = (10, 11))
# plt.subplot(1, 1, 1)
# plt.plot(tout, y1, "b-")
# plt.grid()
# plt.ylabel('h(t) [Hand Calc.]')
# plt.title('Open Loop TF Step Response')

# #PART 2
# print("\n### PART 2 ###\n")
# #1/2

# numG = [1, 9]
# denG = [1, -2, -40, -64]

# numA = [1, 4]
# denA = [1, 4, 3]

# B = [1, 26, 168]

# numMain = spsig.convolve(numA, numG)
# denT1 = spsig.convolve(denG, denA)
# denT2 = spsig.convolve(denA, spsig.convolve(B, numG))
# denMain = denT1 + denT2

# #3
# print("Closed Loop Numerator: ", numMain)
# print("Closed Loop Denominator: ", denMain)
# z,p,k = spsig.tf2zpk(numMain, denMain)
# print("G(s): z = ", z)

# #4
# tout, y1 = spsig.step(spsig.lti(numMain, denMain))
# plt.figure(figsize = (10, 11))
# plt.subplot(1, 1, 1)
# plt.plot(tout, y1, "b-")
# plt.grid()
# plt.ylabel('h(t)')
# plt.title('Closed Loop TF Step Response')