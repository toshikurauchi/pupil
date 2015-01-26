'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)

 Based on the code by Leszek Swirski (https://github.com/LeszekSwirski/singleeyefitter)
'''

import numpy as np
from numpy.linalg import norm
from poly_solver import solve

class Conicoid(object):

    def __init__(self, a, b, c, f, g, h, u, v, w, d):
        self.A = a
        self.B = b
        self.C = c
        self.F = f
        self.G = g
        self.H = h
        self.U = u
        self.V = v
        self.W = w
        self.D = d

    @classmethod
    def from_conic_and_vertex(cls, conic, vertex):
        # Assumes conic is on the plane z = 0

        alpha = vertex[0,0]
        beta = vertex[1,0]
        gamma = vertex[2,0]
        alpha_sq = alpha*alpha
        beta_sq = beta*beta
        gamma_sq = gamma*gamma

        a = conic.A
        h = conic.B/2
        b = conic.C
        g = conic.D/2
        f = conic.E/2
        d = conic.F

        params = [gamma_sq * a,
                  gamma_sq * b,
                  a*alpha_sq + 2*h*alpha*beta + b*beta_sq + 2*g*alpha + 2*f*beta + d,
                  -gamma * (b*beta + h*alpha + f),
                  -gamma * (h*beta + a*alpha + g),
                  gamma_sq * h,
                  gamma_sq * g,
                  gamma_sq * f,
                  -gamma * (f*beta + g*alpha + d),
                  gamma_sq * d]
        params = np.array(params)/norm(params)

        return cls(params[0], params[1], params[2], params[3], params[4],
                   params[5], params[6], params[7], params[8], params[9])

    def canonical_form(self):
        ''' Get canonical conic form:
             lambda(1) X^2 + lambda(2) Y^2 + lambda(3) Z^2 = mu
        Safaee-Rad 1992 eq (6)
        Done by solving the discriminating cubic (10)
        Lambdas are sorted descending because order of roots doesn't
        matter, and it later eliminates the case of eq (30), where
        lambda(2) > lambda(1)'''

        a = self.A
        b = self.B
        c = self.C
        f = self.F
        g = self.G
        h = self.H
        u = self.U
        v = self.V
        w = self.W
        d = self.D

        lbda = solve(1.,
                     -(a + b + c),
                     (b*c + c*a + a*b - f*f - g*g - h*h),
                     -(a*b*c + 2 * f*g*h - a*f*f - b*g*g - c*h*h))

        assert lbda[0] >= lbda[1], 'lambda1 should be greater or equal to lambda2'
        assert lbda[1] > 0, 'lambda2 should be positive'
        assert lbda[2] < 0, 'lambda3 should be negative'

        return lbda
