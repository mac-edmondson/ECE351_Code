#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 01 18:30:00 2022

@author: puffballjack
"""
################################################################
#                                                              #    
#                                                              #
# Macallyster S. Edmondson                                     #
#                                                              #
# ECE351-53                                                    #
#                                                              #
# Lab #3                                                       #
#                                                              #
# 02/08/2022                                                   #
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


#PART 1

#1

def f1(t) :
    y = ss.u(t - 2) - ss.u(t-9);
    return y;

def f2(t) :
    y = np.exp(-t) * ss.u(t)
    return y

def f3(t) :
    y = (ss.r(t-2) * (ss.u(t-2) - ss.u(t-3))) + (ss.r(4-t) * (ss.u(t-3) - ss.u(t-4)))
    return y

#2

t = np.arange(0, 20 + step_size, step_size)
y = f1(t)        

print(len(t))
print(step_size)

plt.figure(figsize = (10, 11))
plt.subplot(3, 1, 1)
plt.plot(t, y, "b-")
plt.grid()
plt.ylabel('f_1(t)')
plt.title('f_1(t), f_2(t), f_3(t) v. t')

y = f2(t)

plt.subplot(3, 1, 2)
plt.plot(t, y, "g-")
plt.grid()
plt.ylabel('f_2(t)')

y = f3(t)

plt.subplot(3, 1, 3)
plt.plot(t, y, "r-")
plt.grid()
plt.ylabel('f_3(t)')
plt.xlabel('t')
plt.show()

#PART 2

#1
    
def my_conv(f1, f2):
    Nf1 = len(f1)
    Nf2 = len(f2)
    f1Ex = np.append(f1, np.zeros((1, Nf2-1)))
    f2Ex = np.append(f2, np.zeros((1, Nf1-1)))
    result = np.zeros(f1Ex.shape)
    for i in range(Nf2 + Nf1 - 2) :
        result[i] = 0
        for j in range(Nf1):
            if(i-j+1 > 0):
                    try:
                        result[i] += f1Ex[j]*f2Ex[i-j+1]
                    except:
                        print(i, j)
    return result

# def convolve(f1, f2) : #f1 and f2 must be of the same length and have had the same input
# #Output will have same length as f1 and f2
#     if(len(f1) != len(f2)) :
#         print("Error: Functions are not of the same length in convolution function!")
#         return
#     else :
#         out = np.zeros(len(f1))     # Output value for return
#         mit = range(len(f1))               # Maximum iteration value
#         #dt = len(f1)
#         for i in mit :
#             sum = 0
#             if(f1[i] != 0) :
#                 for n in range(i):
#                     sum += (f1[n] * f2[i-n]) #Summation for single point to get magnitude of convolution at i
#                 out[i] = sum
#             else:
#                 out[i] = 0
#         return out
                    
bound = 20
t = np.arange(0, bound + step_size, step_size)
f1 = f1(t)
f2 = f2(t)
f3 = f3(t)
# f1f2p = spsig.convolve(f1, f2, mode='full')
# f2f3p = spsig.convolve(f2, f3, mode='full')
# f1f3p = spsig.convolve(f1, f3, mode='full')
f1f2m = my_conv(f1, f2)
print("1 done")
f2f3m = my_conv(f2, f3)
print("2 done")
f1f3m = my_conv(f1, f3)
print("3 done")
t = np.arange(0, bound*2 + step_size, step_size)
#print(y[0])       

plt.figure(figsize = (10, 11))
plt.subplot(3, 1, 1)
# plt.plot(t, f1f2p, "b-")
plt.plot(t, f1f2m, "b-")
plt.grid()
plt.ylabel('f_1 * f_2')
plt.title('Convolutions of f1, f2, f3')

plt.subplot(3, 1, 2)
# plt.plot(t, f2f3p, "g-")
plt.plot(t, f2f3m, "g-")
plt.grid()
plt.ylabel('f_2 * f_3')

plt.subplot(3, 1, 3)
# plt.plot(t, f1f3p, "r-")
plt.plot(t, f1f3m, "r-")
plt.grid()
plt.ylabel('f_2 * f_3')

plt.show()
        