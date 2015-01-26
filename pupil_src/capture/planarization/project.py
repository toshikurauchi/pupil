'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)

 Based on the code by Leszek Swirski (https://github.com/LeszekSwirski/singleeyefitter)
'''

from conic import Conic
from conicoid import Conicoid
from math import sqrt, copysign
import numpy as np
from numpy.linalg import norm
from circle import Circle

def sign(x):
    return copysign(1,x)

def project_circle(circle, focal_length):
    ''' Calculates the projection of the 3D circle on the image
    plane of the camera with given focal length. '''

    c = circle.center
    n = circle.normal
    r = circle.radius
    f = focal_length

    # Construct cone with circle as base and vertex v = (0,0,0).
    #
    # For the circle,
    #     |p - c|^2 = r^2 where (p-c).n = 0 (i.e. on the circle plane)
    #
    # A cone is basically concentric circles, with center on the line c->v.
    # For any point p, the corresponding circle centre c' is the intersection
    # of the line c->v and the plane through p normal to n. So,
    #
    #     d = ((p - v).n)/(c.n)
    #     c' = d c + v
    #
    # The radius of these circles decreases linearly as you approach 0, so
    #
    #     |p - c'|^2 = (r*|c' - v|/|c - v|)^2
    #
    # Since v = (0,0,0), this simplifies to
    #
    #     |p - (p.n/c.n)c|^2 = (r*|(p.n/c.n)c|/|c|)^2
    #
    #     |(c.n)p - (p.n)c|^2         / p.n \^2
    #     ------------------- = r^2 * | --- |
    #           (c.n)^2               \ c.n /
    #
    #     |(c.n)p - (p.n)c|^2 - r^2 * (p.n)^2 = 0
    #
    # Expanding out p, c and n gives
    #
    #     |(c.n)x - (x*n_x + y*n_y + z*n_z)c_x|^2
    #     |(c.n)y - (x*n_x + y*n_y + z*n_z)c_y|   - r^2 * (x*n_x + y*n_y + z*n_z)^2 = 0
    #     |(c.n)z - (x*n_x + y*n_y + z*n_z)c_z|
    #
    #       ((c.n)x - (x*n_x + y*n_y + z*n_z)c_x)^2
    #     + ((c.n)y - (x*n_x + y*n_y + z*n_z)c_y)^2
    #     + ((c.n)z - (x*n_x + y*n_y + z*n_z)c_z)^2
    #     - r^2 * (x*n_x + y*n_y + z*n_z)^2 = 0
    #
    #       (c.n)^2 x^2 - 2*(c.n)*(x*n_x + y*n_y + z*n_z)*x*c_x + (x*n_x + y*n_y + z*n_z)^2 c_x^2
    #     + (c.n)^2 y^2 - 2*(c.n)*(x*n_x + y*n_y + z*n_z)*y*c_y + (x*n_x + y*n_y + z*n_z)^2 c_y^2
    #     + (c.n)^2 z^2 - 2*(c.n)*(x*n_x + y*n_y + z*n_z)*z*c_z + (x*n_x + y*n_y + z*n_z)^2 c_z^2
    #     - r^2 * (x*n_x + y*n_y + z*n_z)^2 = 0
    #
    #       (c.n)^2 x^2 - 2*(c.n)*c_x*(x*n_x + y*n_y + z*n_z)*x
    #     + (c.n)^2 y^2 - 2*(c.n)*c_y*(x*n_x + y*n_y + z*n_z)*y
    #     + (c.n)^2 z^2 - 2*(c.n)*c_z*(x*n_x + y*n_y + z*n_z)*z
    #     + (x*n_x + y*n_y + z*n_z)^2 * (c_x^2 + c_y^2 + c_z^2 - r^2)
    #
    #       (c.n)^2 x^2 - 2*(c.n)*c_x*(x*n_x + y*n_y + z*n_z)*x
    #     + (c.n)^2 y^2 - 2*(c.n)*c_y*(x*n_x + y*n_y + z*n_z)*y
    #     + (c.n)^2 z^2 - 2*(c.n)*c_z*(x*n_x + y*n_y + z*n_z)*z
    #     + (|c|^2 - r^2) * (n_x^2*x^2 + n_y^2*y^2 + n_z^2*z^2 + 2*n_x*n_y*x*y + 2*n_x*n_z*x*z + 2*n_y*n_z*y*z)
    #
    # Collecting conicoid terms gives
    #
    #       [xyz]^2 : (c.n)^2 - 2*(c.n)*c_[xyz]*n_[xyz] + (|c|^2 - r^2)*n_[xyz]^2
    #    [yzx][zxy] : - 2*(c.n)*c_[yzx]*n_[zxy] - 2*(c.n)*c_[zxy]*n_[yzx] + (|c|^2 - r^2)*2*n_[yzx]*n_[zxy]
    #               : 2*((|c|^2 - r^2)*n_[yzx]*n_[zxy] - (c,n)*(c_[yzx]*n_[zxy] + c_[zxy]*n_[yzx]))
    #         [xyz] : 0
    #             1 : 0

    cn = c.T.dot(n)[0,0]
    c2r2 = c.T.dot(c)[0,0] - r**2

    ABC = cn**2 - 2.0*cn*c*n + c2r2*n**2
    F = 2.0*(c2r2*n[0,1]*n[0,2] - cn*(n[0,1]*c[0,2] + n[0,2]*c[0,1]))
    G = 2.0*(c2r2*n[0,2]*n[0,0] - cn*(n[0,2]*c[0,0] + n[0,0]*c[0,2]))
    H = 2.0*(c2r2*n[0,0]*n[0,1] - cn*(n[0,0]*c[0,1] + n[0,1]*c[0,0]))

    # Then set z=f to get conic which is the result of intersecting the cone with the focal plane
    A = ABC[0,0]      # x^2 (Ax^2)
    B = H             # xy  (Hxy)
    C = ABC[0,1]      # y^2 (By^2)
    D = G*f           # x   (Gxz + Ux, z = f)
    E = F*f           # y   (Fyz + Vy, z = f)
    F = ABC[0,2]*f**2 # 1   (Cz^2 + Wz + D, z = f)

    return Conic(A,B,C,D,E,F)

def unproject_ellipse(ellipse, circle_radius, focal_length):
    ''' Calculates the 3D circle that is projected at the
    ellipse using the method proposed by Safaee-Rad et al. (1992).'''

    # Get cone with base of ellipse and vertex at [0 0 -f]
    # Safaee-Rad 1992 eq (3)
    conic = Conic.from_ellipse(ellipse)
    cam_center_in_ellipse = np.array([[0], [0], [-focal_length]])
    pupil_cone = Conicoid.from_conic_and_vertex(conic, cam_center_in_ellipse)
    return find_circle_in_conicoid(pupil_cone, circle_radius, focal_length)

def find_circle_in_conicoid(conicoid, circle_radius, focal_length, debug=False):
    a = conicoid.A
    b = conicoid.B
    c = conicoid.C
    f = conicoid.F
    g = conicoid.G
    h = conicoid.H
    u = conicoid.U
    v = conicoid.V
    w = conicoid.W
    d = conicoid.D

    lbda = conicoid.canonical_form()
    lbda = np.array([lbda]).T # We use column vectors

    # Now want to calculate l,m,n of the plane
    #     lX + mY + nZ = p
    # which intersects the cone to create a circle.
    # Safaee-Rad 1992 eq (31)
    # [Safaee-Rad 1992 eq (33) comes out of this as a result of lambda(1) == lambda(2)]
    n = sqrt((lbda[1] - lbda[2]) / (lbda[0] - lbda[2]))
    m = 0.0
    l = sqrt((lbda[0] - lbda[1]) / (lbda[0] - lbda[2]))
    # There are two solutions for l, positive and negative, we handle these later

    # Want to calculate T1, the rotation transformation from image
    # space in the canonical conic frame back to image space in the
    # real world

    T1 = np.identity(4)

    # Safaee-Rad 1992 eq (12)
    t1 = (b - lbda)*g - f*h
    t2 = (a - lbda)*f - g*h
    t3 = -(a - lbda)*(t1 / t2) / g - h / g

    T1[1,0:3] = (1 / (1 + (t1 / t2)**2 + t3**2)**.5).T
    T1[0,0:3] = (t1 / t2).T * T1[1,0:3]
    T1[2,0:3] = t3.T * T1[1,0:3]

    # If li,mi,ni follow the left hand rule, flip their signs
    if np.cross(T1[0,0:3],T1[1,0:3]).dot(T1[2,0:3]) < 0:
        T1[0:3,0:3] = -T1[0:3,0:3]

    # Calculate T2, a translation transformation from the canonical
    # conic frame to the image space in the canonical conic frame
    # Safaee-Rad 1992 eq (14)

    T2 = np.identity(4)
    T2[0:3,3] = -np.array([[u,v,w]]).dot(T1[0:3,0:3])/lbda.T

    solutions = [None, None]
    ls = [l, -l]
    for i in range(2):
        l = ls[i]
        # Circle normal in image space (i.e. gaze vector)
        normal = T1[0:3,0:3].dot(np.array([[l],[m],[n]]))

        # Calculate T3, a rotation from a frame where Z is the circle normal
        # to the canonical conic frame
        # Safaee-Rad 1992 eq (19)
        # Want T3 = / -m/sqrt(l*l+m*m) -l*n/sqrt(l*l+m*m) l \
        #           |  l/sqrt(l*l+m*m) -m*n/sqrt(l*l+m*m) m |
        #           \         0           sqrt(l*l+m*m)   n /
        # But m = 0, so this simplifies to
        #    T3 = /       0      -n*l/sqrt(l*l) l \
        #         |  l/sqrt(l*l)        0       0 |
        #         \       0         sqrt(l*l)   n /
        #       = /    0    -n*sgn(l) l \
        #         |  sgn(l)     0     0 |
        #         \    0       |l|    n /

        T3 = np.identity(4)
        if l == 0:
            # Discontinuity of sgn(l), have to handle explicitly
            assert n == 1
            import warnings
            warnings.warn('l == 0')
            T3[0:3,0:3] = np.array([[0, -1, 0],
                                    [1,  0, 0],
                                    [0,  0, 1]])
        else:
            T3[0:3,0:3] = np.array([[0      ,-n*sign(l), l],
                                    [sign(l),     0    , 0],
                                    [0      ,  abs(l)  , n]])

        # Calculate the circle center
        # Safaee-Rad 1992 eq (38), using T3 as defined in (36)
        A = lbda.T.dot(T3[0:3,0]**2)[0]
        B = lbda.T.dot(T3[0:3,0]*T3[0:3,2])[0]
        C = lbda.T.dot(T3[0:3,1]*T3[0:3,2])[0]
        D = lbda.T.dot(T3[0:3,2]**2)[0]

        # Safaee-Rad 1992 eq (41)
        center_in_Xprime = np.ones(shape=(4,1))
        center_in_Xprime[2,0] = A*circle_radius / sqrt(B**2 + C**2 - A*D);
        center_in_Xprime[0,0] = -B / A * center_in_Xprime[2,0]
        center_in_Xprime[1,0] = -C / A * center_in_Xprime[2,0]

        # Safaee-Rad 1992 eq (34)
        T0 = np.identity(4)
        T0[0:3,3] = [0, 0, focal_length]

        # Safaee-Rad 1992 eq (42) using (35)
        center = T0.dot(T1).dot(T2).dot(T3).dot(center_in_Xprime)

        # If z is negative (behind the camera), choose the other
        # solution of eq (41) [maybe there's a way of calculating which
        # solution should be chosen first]

        if center[2,0] < 0:
            center_in_Xprime[0:3,0] = -center_in_Xprime[0:3,0]
            center = T0.dot(T1).dot(T2).dot(T3).dot(center_in_Xprime)

        # Make sure that the gaze vector is toward the camera and is normalized
        if normal.T.dot(center[0:3])[0,0] > 0:
            normal = -normal
        normal = normal/norm(normal)

        # Save the results
        solutions[i] = Circle(center[0:3], normal, circle_radius)
        if debug:
            print 'Lambda:', lbda
            print '[l, m, n] =', [l, m, n]
            print 'T0:', T0
            print 'T1:', T1
            print 'T2:', T2
            print 'T3:', T3
            print 'Center:', center[0:3]
            print 'Normal:', normal
            print 'Radius:', circle_radius

    return solutions

def project_point(point, focal_length):
    return np.reshape(focal_length * point[0:2,0] / point[2,0], (2,1))

def project_sphere(sphere, focal_length):
    # TODO I think this is just an approximation. This is not accurate.
    center = focal_length * sphere.center[0:2,0] / sphere.center[2,0]
    diameter = 2 * focal_length * sphere.radius / sphere.center[2,0]
    theta  = 0
    return ((center[0], center[1]), (diameter, diameter), theta)
