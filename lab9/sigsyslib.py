#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 12:29:38 2022

@author: puffballjack
"""

# Module used to implement functions for ECE350 Signals and Systems.

import numpy as np

def u(t) :                  # Unit Step Function
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0 :
            y[i] = 0
        else:
            y[i] = 1
    
    return y

def r(t) :                  # Ramp Function
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = t[i];

    return y