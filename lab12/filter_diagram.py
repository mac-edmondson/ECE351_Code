#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:59:03 2022

@author: mac-edmondson
"""

import schemdraw
import schemdraw.elements as elm

# with schemdraw.Drawing() as d:
#     d += elm.Resistor().label('$9.4\: \Omega$')
#     d.push()
#     d += elm.Capacitor().down().label('$3.5\: \mu F$', loc='bottom')
#     d += elm.Line().left().dot(open=True)
#     # d += elm.Ground()
#     # d += elm.Line().dot(open=True)
#     d += elm.Gap().up().label(('-', '$V_{in}$', '+')).dot(open=True)
#     # d += elm.Line().dot(open=True)
#     d.pop()
#     d += elm.Line()
#     # d.push()
#     d += elm.Inductor().down().label('$2.01\: mH$', loc='bottom')
#     d.push()
#     d += elm.Line().left()
#     d.pop()
#     d += elm.Line().right().dot(open=True)
#     # d.pop()
#     # d += elm.Line().dot(open=True)
#     d += elm.Gap().up().label(('-', '$V_{out}$', '+')).dot(open=True)    
#     d += elm.Line().left()
    
with schemdraw.Drawing() as d:
    d += elm.Inductor().label('$2.01\: mH$', loc='top')
    d += elm.Capacitor().label('$3.5\: \mu F$', loc='top')
    d += elm.Resistor().down().label('$9.4\: \Omega$', loc='top')
    d.push()
    d += elm.Line().left()
    d += elm.Line().dot(open=True)
    d += elm.Gap().up().label(('-', '$V_{in}$', '+')).dot(open=True)
    d.pop()
    d += elm.Line().right().dot(open=True)
    d += elm.Gap().up().label(('-', '$V_{out}$', '+')).dot(open=True)
    d += elm.Line().left()