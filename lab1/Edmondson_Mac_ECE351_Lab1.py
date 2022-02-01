################################################################
#                                                              #    
#                                                              #
# Macallyster S. Edmondson                                     #
#                                                              #
# ECE351-53                                                    #
#                                                              #
# Lab 0                                                        #
#                                                              #
# 01/25/2022                                                   #
#                                                              #
# Any other necessary information needed to navigate the file  #
#                                                              #
#                                                              #
#                                                              #
################################################################

import numpy
import scipy.signal
import time
import cmath

def g(w) : 
    return 1/((4-(w**2))+(w*2j))

def g_ang(w):
    return numpy.degrees(cmath.phase(g(w)));

def g_mag(w):
    return abs(g(w))

def y(F, w, Fa):
    thetaf = Fa;
    thetay = g_ang(w);
    ang = thetaf + thetay
    print(F*g_mag(w), "* cos(", w, "t +", ang, ")")

print(g(5))
y(3,5, -10)
y(5,5, -45)
y(2,2, 178)


#print(g_mag(5))
#print(numpy.degrees(cmath.phase((1+1j))))