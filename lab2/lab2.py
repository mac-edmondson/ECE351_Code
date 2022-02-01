#!/usr/bin/env python3
# -*- coding: utf-8 -*-
################################################################
#                                                              #    
#                                                              #
# Macallyster S. Edmondson                                     #
#                                                              #
# ECE351-53                                                     #
#                                                              #
# Lab #2                                                       #
#                                                              #
# 02/01/2022                                                   #
#                                                              #
# Broken into Parts and Tasks as specified in lab handout.     #
#                                                              #
#                                                              #
#                                                              #
################################################################

import numpy as np
import matplotlib.pyplot as plt

# PART 1:

def func1(t) :
    return np.cos(t)

step_size = .1
t = np.arange(0, 10 + step_size, step_size)
y = func1(t)

plt.subplot(1, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('cos(t)')
plt.xlabel('t')
plt.title('cos(t) v. t ; from 0 to 10, inclusive')
plt.show()



# PART 2:

def u(t) :
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0 :
            y[i] = 0
        else:
            y[i] = 1
    
    return y

def r(t) :
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = t[i];

    return y

step_size = .001
t = np.arange(-2, 8 + step_size, step_size)
y = u(t)           

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('u(t)')
plt.title('step and ramp function examples')

y = r(t)

plt.subplot(2, 1, 2)
plt.plot(t, y)
plt.grid()
plt.ylabel('r(t)')
plt.xlabel('t')
plt.show()

def fig2_func(t) :
    y = r(t) - r(t-3) + 5 * u(t-3) - 2 * u(t-6) - 2*r(t - 6)
    return y

step_size = .0001
t = np.arange(-5, 10 + step_size, step_size)
y = fig2_func(t)         

plt.figure(figsize = (10, 7))
plt.subplot(1, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Part 2 - Plot Reproduction')
plt.show()


#PART 3:
  
step_size = 1e-3

#1
t = np.arange(-10, 5 + step_size, step_size)
y = fig2_func(-t)         

plt.figure(figsize = (10, 7))
plt.subplot(1, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('f(-t)')
plt.xlabel('t')
plt.title('Part 3: #1')

#2

t = np.arange(-1, 14 + step_size, step_size)
y = fig2_func(t-4)

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('f(t-4)')
plt.title('Part 3: #2')

t = np.arange(-14, 1 + step_size, step_size)
y = fig2_func(-t-4)

plt.subplot(2, 1, 2)
plt.plot(t, y)
plt.grid()
plt.ylabel('f(-t-4)')
plt.xlabel('t')
plt.show()

#3

t = np.arange(-1, 10*2 + step_size, step_size)
y = fig2_func(t/2)

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('f(t/2)')
plt.title('Part 3: #3')

t = np.arange(-1, 10/2 + step_size, step_size)
y = fig2_func(t*2)

plt.subplot(2, 1, 2)
plt.plot(t, y)
plt.grid()
plt.ylabel('f(2t)')
plt.xlabel('t')
plt.show()

#4

t = np.arange(0, 10 + step_size, step_size)
y = fig2_func(t)

dt = np.diff(t, axis=0)
dy = np.diff(y, axis =0)     
 
plt.figure(figsize = (10, 7))
plt.subplot(1, 1, 1)
plt.plot(t[range(len(dy))], dy/dt)
plt.grid()
plt.ylim(-3, 10)
plt.ylabel('f(t)')
plt.xlabel('t')
plt.title('Part 3: #4')