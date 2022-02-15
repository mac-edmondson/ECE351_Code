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
step_size = 1e-3

# PART 1

# 1

def h1(t) :
    out = np.exp(-2*t) * (ss.u(t) - ss.u(t-3))
    return out

def h2(t) :
    out = ss.u(t-2) - ss.u(t-6)
    return out

def h3(t) :
    f0 = 0.25 #Hz
    w0 = 2*np.pi*f0
    out = np.cos(w0 * t) * ss.u(t)
    return out

# 2

t = np.arange(-10, 10 + step_size, step_size)
y = h1(t)        

print(len(t))
print(step_size)

plt.figure(figsize = (10, 11))
plt.subplot(3, 1, 1)
plt.plot(t, y, "b-")
plt.grid()
plt.ylabel('h_1(t)')
plt.title('h_1(t), h_2(t), h_3(t) v. t')

y = h2(t)

plt.subplot(3, 1, 2)
plt.plot(t, y, "g-")
plt.grid()
plt.ylabel('h_2(t)')

y = h3(t)

plt.subplot(3, 1, 3)
plt.plot(t, y, "r-")
plt.grid()
plt.ylabel('h_3(t)')
plt.xlabel('t')
plt.show()


# PART 2

# 1

bound = 10
t = np.arange(-10, bound + step_size, step_size)
f1 = h1(t)
f2 = h2(t)
f3 = h3(t)
u = ss.u(t)
f1p = spsig.convolve(f1, u, mode='full')
f2p = spsig.convolve(f2, u, mode='full')
f3p = spsig.convolve(f3, u, mode='full')
t = np.arange(-10 - bound, bound*2 + .5*step_size, step_size)
#print(y[0])       

plt.figure(figsize = (10, 11))
plt.subplot(3, 1, 1)
plt.plot(t, f1p, "b-")
plt.xlim(-20, 20)
plt.grid()
plt.ylabel('h_1(t)')
plt.title('Step response of h_1(t), h_2(t), h_3(t) (Discrete)')

plt.subplot(3, 1, 2)
plt.plot(t, f2p, "g-")
plt.xlim(-20, 20)
plt.grid()
plt.ylabel('f_2 * f_3')

plt.subplot(3, 1, 3)
plt.plot(t, f3p, "r-")
plt.xlim(-20, 20)
plt.grid()
plt.ylabel('f_2 * f_3')

plt.show()

# # 1+

# bound = 20
# t = np.arange(-20, bound + step_size, step_size)
# f1 = h1(t)
# f2 = h2(t)
# f3 = h3(t)
# u = ss.u(t)
# f1p = spsig.convolve(f1, u, mode='full')
# f2p = spsig.convolve(f2, u, mode='full')
# f3p = spsig.convolve(f3, u, mode='full')
# t = np.arange(-20 - bound, bound*2 + step_size*3, step_size)
# #print(y[0])       

# plt.figure(figsize = (10, 11))
# plt.subplot(3, 1, 1)
# plt.plot(t, f1p, "b-")
# plt.xlim(-20, 20)
# plt.grid()
# plt.ylabel('h_1(t)')
# plt.title('Step response of h_1(t), h_2(t), h_3(t) (Discrete)')

# plt.subplot(3, 1, 2)
# plt.plot(t, f2p, "g-")
# plt.xlim(-20, 20)
# plt.grid()
# plt.ylabel('f_2 * f_3')

# plt.subplot(3, 1, 3)
# plt.plot(t, f3p, "r-")
# plt.xlim(-20, 20)
# plt.grid()
# plt.ylabel('f_2 * f_3')

# plt.show()


# 2

t = np.arange(-20, 20 + step_size, step_size)
w0 = 2*np.pi*0.25
f1 = ((1/2) * (np.exp(-2*(t-3)) - 1) * ss.u(t-3)) - ((1/2) * (np.exp(-2*(t)) - 1) * ss.u(t))
f2 = ((t-2) * ss.u(t-2)) - ((t-6) * ss.u(t-6))
f3 = (1/w0) * np.sin(w0 * t) * ss.u(t)

plt.figure(figsize = (10, 11))
plt.subplot(3, 1, 1)
plt.plot(t, f1, "b-")
plt.xlim(-20, 20)
plt.grid()
plt.ylabel('h_1(t)')
plt.title('Step response of h_1(t), h_2(t), h_3(t) (Hand-Calc.)')

plt.subplot(3, 1, 2)
plt.plot(t, f2, "g-")
plt.xlim(-20, 20)
plt.grid()
plt.ylabel('f_2 * f_3')

plt.subplot(3, 1, 3)
plt.plot(t, f3, "r-")
plt.xlim(-20, 20)
plt.grid()
plt.ylabel('f_2 * f_3')

plt.show()