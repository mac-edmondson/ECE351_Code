#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 14:37:59 2022

@author: puffballjack
"""

import cmath, numpy

def Y(w) :
    return 1/(complex((2*(1-(w**2))), ((9*w - (w**3)))))

def Y_ang(w):
    return numpy.degrees(cmath.phase(Y(w)));

def Y_mag(w):
    return abs(Y(w))

def y(F, w, Fa):
    thetaf = Fa;
    thetay = Y_ang(w);
    ang = thetaf + thetay
    print(F*Y_mag(w), "* sin(", w, "t +", ang, ")")
    
y(3,1,0)
y(2,3,90)