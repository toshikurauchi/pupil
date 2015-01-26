'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)

 Based on the code by Leszek Swirski (https://github.com/LeszekSwirski/singleeyefitter)
'''

import math
import numpy as np

class Circle(object):
    def __init__(self, center, normal, radius, valid=True):
        self.center = center
        self.normal = normal
        self.radius = radius
        self.valid = valid

    @classmethod
    def from_params(cls, eye, params):
        if params.radius == 0:
            return cls(None, None, None, False)

        theta = params.theta
        psi = params.psi
        radial = np.array([[math.sin(theta)*math.cos(psi)],
                           [math.cos(theta)],
                           [math.sin(theta)*math.sin(psi)]])

        center = eye.center + eye.radius * radial
        normal = radial
        radius = params.radius
        return cls(center, normal, radius)

class CircleParams(object):
    def __init__(self, theta=0, psi=0, radius=0):
        self.theta = theta
        self.psi = psi
        self.radius = radius
