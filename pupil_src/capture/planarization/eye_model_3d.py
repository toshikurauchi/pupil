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
import math
#logging
import logging
from matplotlib.patheffects import Normal
logger = logging.getLogger(__name__)

from project import unproject_ellipse, project_point, project_sphere
from distance import point_line_distance
from intersect import nearest_intersect_lines
from sphere import Sphere
from pupil3d import Pupil3D
from line import Line
from circle import Circle

def m_error(point, line, epsilon):
    dist = point_line_distance(point, line)
    if (dist**2 < epsilon**2):
        return dist**2
    return epsilon**2

class EyeModel3D(object):
    def __init__(self, pupil_ellipses, img_origin=(0,0), focal_length=1500):
        self.pupils = pupil_ellipses
        self.pupils3d = []
        self.version = 0
        self.camera_center = np.array([[0],[0],[0]])
        self.focal_length = focal_length
        self.img_origin = img_origin
        self.eye = Sphere(np.array([[-1],[-1],[-1]]), radius=-1, valid=False)

        if self._find_center():
            self._initialize()

    def planarize(self, pupil_ellipse):
        ''' Finds the planarized pupil center by calculating the optical axis
        of the eye using Swirski and Dodgson's method, intersecting it with a
        virtual plane and projecting it to the eye image. '''

        assert self.eye.valid, 'Eye is not valid'

        # Find optical axis
        pupil = self._find_pupil_3d(pupil_ellipse)
        normal = pupil.circle.normal

        # We use the plane at distance 1 from the center of the eye that is
        # perpendicular to the line that connects the center of the eye and the
        # center of projection of the eye camera. The origin of the plane
        # coordinate system is the intersection of this line with the plane.
        normal = normal/normal[2,0]
        return (normal[0,0], normal[1,0]), pupil

    def get_eyeball_image(self):
        e = project_sphere(self.eye, self.focal_length)
        cx = e[0][0] + self.img_origin[0]
        cy = e[0][1] + self.img_origin[1]
        return ((cx,cy),e[1],e[2])

    def get_normal_image(self, normal, center, scale = 2):
        origin = np.array([self.img_origin]).T
        c = project_point(center, self.focal_length) + origin
        n = project_point(normal*scale, self.focal_length)
        end = c+n
        return [(c[0,0],c[1,0]),(end[0,0],end[1,0])]

    def _find_center(self, pupil_radius=1, eye_z=20, use_ransac=True):
        ''' Estimates the center of the eye model as described by
        Swirski and Dodgson (2013).'''
        if len(self.pupils) < 2:
            logger.warning('Need at least two observations')
            return False

        pupil_unprojection_pairs = []
        pupil_gazelines_proj = []
        self.pupil_gazelines_proj = [] # All exposed objects keep the origin at the corner of the image

        for pupil in self.pupils:
            # Get pupil circles (up to depth)
            #
            # Do a per-image unprojection of the pupil ellipse into the two fixed
            # size circles that would project onto it. The size of the circles
            # doesn't matter here, only their center and normal does.
            ellipse = self._fix_ellipse_origin(pupil.ellipse)
            unprojection_pair = unproject_ellipse(ellipse,
                                                  pupil_radius,
                                                  self.focal_length)

            # Get projected circles and gaze vectors
            #
            # Project the circle centers and gaze vectors down back onto the image
            # plane. We're only using them as line parametrizations, so it doesn't
            # matter which of the two centers/gaze vectors we use, as the
            # two gazes are parallel and the centers are co-linear.

            c = unprojection_pair[0].center
            v = unprojection_pair[0].normal

            c_proj = project_point(c, self.focal_length)
            v_proj = project_point(v + c[0:3,0], self.focal_length) - c_proj

            v_proj = v_proj/norm(v_proj)

            pupil_unprojection_pairs.append(unprojection_pair)
            pupil_gazelines_proj.append(Line(c_proj,v_proj))
            self.pupil_gazelines_proj.append(Line(c_proj + np.array([self.img_origin]).T, v_proj))

        # Get eyeball center
        #
        # Find a least-squares 'intersection' (point nearest to all lines) of
        # the projected 2D gaze vectors. Then, unproject that circle onto a
        # point a fixed distance away.
        #
        # For robustness, use RANSAC to eliminate stray gaze lines
        #
        # (This has to be done here because it's used by the pupil circle
        # disambiguation)

        pupil_gazelines_proj = np.array(pupil_gazelines_proj)

        if use_ransac:
            indices = range(len(pupil_gazelines_proj))

            n = 2
            w = 0.3
            p = 0.9999
            k = int(math.ceil(math.log(1 - p) / math.log(1 - w**n)))

            epsilon = 10

            best_inlier_indices = indices
            best_line_distance_error = float('inf')

            for i in range(k):
                index_sample = np.random.choice(indices, size=n, replace=False)
                sample = pupil_gazelines_proj[index_sample]

                sample_center_proj = nearest_intersect_lines(sample)

                is_inlier = lambda line: point_line_distance(sample_center_proj, line) < epsilon
                index_inliers = [i for i in range(len(pupil_gazelines_proj)) if is_inlier(pupil_gazelines_proj[i])]
                inliers = pupil_gazelines_proj[index_inliers]

                if len(inliers) <= w*len(pupil_gazelines_proj):
                    continue

                inlier_center_proj = nearest_intersect_lines(inliers)

                error_fun = lambda line : m_error(inlier_center_proj, line, epsilon)
                line_distance_error = np.sum([error_fun(line) for line in pupil_gazelines_proj])

                if line_distance_error < best_line_distance_error:
                    best_eye_center_proj = inlier_center_proj
                    best_line_distance_error = line_distance_error
                    best_inlier_indices = index_inliers

            msg = 'Inliers: {inl} ({pct}%) = {err}'
            msg = msg.format(inl=len(best_inlier_indices),
                             pct=(100.0*len(best_inlier_indices) / len(pupil_gazelines_proj)),
                             err=best_line_distance_error)
            logger.info(msg)

            for i in range(len(self.pupils)):
                self.pupils[i].init_valid = False
            for i in range(len(best_inlier_indices)):
                self.pupils[best_inlier_indices[i]].init_valid = True

            if len(best_inlier_indices) > 0:
                eye_center_proj = best_eye_center_proj
                valid_eye = True
            else:
                valid_eye = False
        else:
            for pupil in self.pupils:
                pupil.init_valid = True
            eye_center_proj = nearest_intersect_lines(pupil_gazelines_proj)
            valid_eye = True

        if valid_eye:
            center_xy = eye_center_proj * eye_z / self.focal_length
            center = np.array([[center_xy[0,0]], [center_xy[1,0]], [eye_z]])
            self.eye = Sphere(center, radius=1, valid=True)

            # Disambiguate pupil circles using projected eyeball center
            #
            # Assume that the gaze vector points away from the eye center, and
            # so projected gaze points away from projected eye center. Pick the
            # solution which satisfies this assumption
            for i in range(len(self.pupils)):
                pupil_pair = pupil_unprojection_pairs[i]
                line = pupil_gazelines_proj[i]

                c_proj = line.origin
                v_proj = line.direction

                # Check if v_proj going away from est eye center. If it is, then
                # the first circle was correct. Otherwise, take the second one.
                # The two normals will point in opposite directions, so only need
                # to check one.
                if (c_proj - eye_center_proj).T.dot(v_proj)[0,0] >= 0:
                    self.pupils[i].circle = pupil_pair[0]
                else:
                    self.pupils[i].circle = pupil_pair[1]
        else:
            # No inliers, so no eye
            self.eye = Sphere(np.array([[-1],[-1],[-1]]), radius=-1, valid=False)

            # Arbitrarily pick first circle
            for i in range(len(self.pupils)):
                pupil_pair = pupil_unprojection_pairs[i]
                self.pupils[i].circle = pupil_pair[0]

        self.version += 1
        self.pupil_radius = pupil_radius

        return True

    def _initialize(self):
        ''' Estimates the eye radius and then adjusts the model
        (radius and center) to anthropomorphic average radius of 12 mm. '''

        if not self.eye.valid:
            return

        # Find pupil positions on eyeball to get radius
        #
        # For each image, calculate the 'most likely' position of the pupil
        # circle given the eyeball sphere estimate and gaze vector. Re-estimate
        # the gaze vector to be consistent with this position.

        # First estimate of pupil center, used only to get an estimate of eye radius

        eye_radius_acc = 0
        eye_radius_count = 0

        for pupil in self.pupils:
            if not hasattr(pupil,'circle'):
                continue
            if not pupil.init_valid:
                continue

            # Intersect the gaze from the eye center with the pupil circle
            # center projection line (with perfect estimates of gaze, eye
            # center and pupil circle center, these should intersect,
            # otherwise find the nearest point to both lines)

            pupil_center_direction = pupil.circle.center/norm(pupil.circle.center)
            lines = [Line(self.eye.center, pupil.circle.normal),
                     Line(self.camera_center, pupil_center_direction)]

            pupil_center = nearest_intersect_lines(lines)

            distance = norm(pupil_center - self.eye.center)

            eye_radius_acc = eye_radius_acc + distance
            eye_radius_count = eye_radius_count + 1

        # Set the eye radius as the mean distance from pupil centers to eye center
        self.eye.radius = eye_radius_acc / eye_radius_count

        # Second estimate of pupil radius, used to get position of pupil on eye

        for pupil in self.pupils:
            pupil.init_circle_from_model(self)

        # Scale eye to anthropomorphic average radius of 12mm
        scale = 12.0 / self.eye.radius
        self.eye.radius = 12.0
        self.eye.center = self.eye.center * scale
        for pupil in self.pupils:
            pupil.params.radius = pupil.params.radius * scale
            pupil.circle = Circle.from_params(self.eye, pupil.params)

        self.version += 1

    def _find_pupil_3d(self, pupil_ellipse):
        assert self.eye.valid, 'Need to get eye center estimate first (by unprojecting multiple observations)'

        ellipse = self._fix_ellipse_origin(pupil_ellipse)
        center = np.array([[ellipse[0][0]],[ellipse[0][1]],[self.focal_length]]) # We just need the direction to be correct
        radius = 1 # The radius doesn't matter to us
        pupil_circle = Circle(center, np.array([[-1],[-1],[-1]]), radius, valid=False)
        pupil = Pupil3D(pupil_circle, pupil_ellipse)
        pupil.init_circle_from_model(self)
        return pupil

    def _fix_ellipse_origin(self, ellipse):
        # The model assumes that the origin is at the center of the image
        cx = ellipse[0][0] - self.img_origin[0]
        cy = ellipse[0][1] - self.img_origin[1]
        return ((cx,cy),ellipse[1],ellipse[2])
