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
# import scipy.signal as spsig

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

# fList = [1, 3, 15, 50, 150, 1500]
fList = [100000]

for i in fList :
    t = np.arange(0, 20 + step_size, step_size)
    y1 = x(t, i)
    
    plt.figure(figsize = (10, 11))
    plt.subplot(1, 1, 1)
    plt.plot(t, y1, "b-")
    plt.grid()
    plt.ylabel('x(t) [N = ' + str(i) + ']')
    plt.xlabel('t')
    plt.title('Fourier Series Output for N = ' + str(i))