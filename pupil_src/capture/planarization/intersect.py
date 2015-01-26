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
from math import sqrt

def intersect_line_sphere(line, sphere):
    ''' Finds intersection between given line and sphere. Returns [inter1, inter2, found].
    If there is no intersection found=False and inter1 and inter2 are not
    valid.'''

    assert abs(norm(line.direction) - 1) < 0.0001, 'Line direction is not unit vector'

    v = line.direction
    # Put p at origin
    p = line.origin
    c = sphere.center - p
    r = sphere.radius

    # From Wikipedia :)
    vcvc_cc_rr = v.T.dot(c)[0,0]**2 - c.T.dot(c)[0,0] + r**2
    if vcvc_cc_rr < 0:
        inter = np.array([[-1],[-1],[-1]])
        return [inter, inter, False]

    s1 = v.T.dot(c)[0,0] - sqrt(vcvc_cc_rr)
    s2 = v.T.dot(c)[0,0] + sqrt(vcvc_cc_rr)

    inter1 = p + s1*v
    inter2 = p + s2*v
    return [inter1, inter2, True]

def nearest_intersect_lines(lines):
    ''' Finds the intersection (in a least-squares sense) of multiple lines. '''
    D = len(lines[0].direction)
    A = np.zeros(shape=(D,D))
    b = np.zeros(shape=(D,1))

    for line in lines:
        vi = line.direction
        pi = line.origin

        Ivivi = np.identity(D) - vi.dot(vi.T)

        A = A + Ivivi
        b = b + Ivivi.dot(pi)

    return np.linalg.lstsq(A, b)[0]

