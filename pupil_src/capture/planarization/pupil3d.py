'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.

 Based on the code by Leszek Swirski (https://github.com/LeszekSwirski/singleeyefitter)
----------------------------------------------------------------------------------~(*)
'''

from numpy.linalg import norm
from math import acos, atan2

from line import Line
from circle import Circle, CircleParams
from intersect import intersect_line_sphere
from scipy.constants.constants import psi

class Pupil3D(object):
    def __init__(self, circle, pupil_ellipse, init_valid=True):
        self.circle = circle
        self.ellipse = pupil_ellipse
        self.init_valid = init_valid

    def init_circle_from_model(self, eye_model):
        # Ignore the pupil circle normal, and intersect the pupil circle
        # center projection line with the eyeball sphere
        direction = self.circle.center
        direction = direction/norm(direction)
        line = Line(eye_model.camera_center, direction)
        new_pupil_center, _intersect2, found = intersect_line_sphere(line, eye_model.eye)
        if found:
            # Now that we have 3D positions for the pupil (rather than just a
            # projection line), recalculate the pupil radius at that position.
            pupil_radius_at_1 = self.circle.radius / self.circle.center[2,0]
            new_pupil_radius = pupil_radius_at_1 * new_pupil_center[2,0]

            # Parameterize this new pupil position using spherical coordinates
            center_to_pupil = new_pupil_center - eye_model.eye.center
            r = norm(center_to_pupil)
            theta = acos(center_to_pupil[1,0] / r)
            psi = atan2(center_to_pupil[2,0], center_to_pupil[0,0])
            radius = new_pupil_radius
            self.params = CircleParams(theta, psi, radius)

            # Update pupil circle to match parameters
            self.circle = Circle.from_params(eye_model.eye, self.params)
        else:
            self.circle.valid = False
            self.params = CircleParams()
