'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)

 Based on the code by Leszek Swirski (https://github.com/LeszekSwirski/singleeyefitter)
'''

from math import cos, sin, pi
import numpy as np
from numpy.linalg import norm
import poly_solver

class Conic(object):
    def __init__(self, a, b, c, d, e, f):
        self.A = a
        self.B = b
        self.C = c
        self.D = d
        self.E = e
        self.F = f

    @classmethod
    def from_ellipse(cls, ellipse):
        eps = 1e-10 # For some reason it doesn't work for some points if we don't do this :(
        ax = cos(ellipse[2]*pi/180 + eps)
        ay = sin(ellipse[2]*pi/180 + eps)

        a2 = ellipse[1][0]*ellipse[1][0]/4
        b2 = ellipse[1][1]*ellipse[1][1]/4

        h = ellipse[0][0]
        k = ellipse[0][1]

        # From: http://www.maa.org/external_archive/joma/Volume8/Kalman/General.html
        a = ax**2 / a2 + ay**2 / b2
        b = 2*ax*ay*(1/a2-1/b2)
        c = ay**2 / a2 + ax**2 / b2
        d = -(2*a*h + k*b)
        e = -(2*c*k + b*h)
        f = a*h**2 + b*h*k + c*k**2 - 1

        params = [a, b, c, d, e, f] # Normalizing the parameters improves the numerical results
        params = np.array(params)/norm(params)

        return cls(params[0], params[1], params[2], params[3], params[4], params[5])
