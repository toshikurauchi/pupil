'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)

 Based on the code by Leszek Swirski (https://github.com/LeszekSwirski/singleeyefitter)
'''

from math import sin, cos, acos, sqrt

def solve(*coefs):
    if len(coefs) == 0 or len(coefs) > 4:
        raise TypeError('solve can only solve polynomials of degree 0-3')

    if len(coefs) == 1:
        return solve0(coefs[0])
    if len(coefs) == 2:
        return solve1(coefs[0], coefs[1])
    if len(coefs) == 3:
        return solve2(coefs[0], coefs[1], coefs[2])
    if len(coefs) == 4:
        return solve3(coefs[0], coefs[1], coefs[2], coefs[3])
    return None

def solve0(a):
    '''Solves a = 0'''
    assert a==0, 'No solution'
    return 0

def solve1(a, b):
    '''Solves ax + b = 0'''
    if a == 0:
        return solve0(b)
    return -b/a

def solve2(a, b, c):
    '''Solves ax^2 + bx + c = 0'''
    if a == 0:
        root = solve1(b, c)
        return [root, root]
    # http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-6.pdf
    # Pg 184
    det = b**2 - 4 * a*c
    assert det >= 0, 'No solution'
    if b >= 0:
        q = -0.5*(b + sqrt(det))
    else:
        q = -0.5*(b + -sqrt(det))
    return [q / a, c / q]

def solve3(a, b, c, d):
    '''Solves ax^3 + bx^2 + cx + d = 0'''
    if a == 0:
        root = solve2(b, c, d)
        return [root[0], root[1], root[1]]
    # http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-6.pdf
    # http://web.archive.org/web/20120321013251/http://linus.it.uts.edu.au/~don/pubs/solving.html

    def cbrt(x):
        return x**(1/3.0)

    p = b / a
    q = c / a
    r = d / a

    u = q - p**2 / 3
    v = r - p*q / 3 + 2 * p**3 / 27

    j = 4 * u**3 / 27 + v**2;

    M = float('inf')
    sqrtM = sqrt(M)
    cbrtM = cbrt(M)

    if b == 0 and c == 0:
        return [cbrt(-d), cbrt(-d), cbrt(-d)]
    if abs(p) > 27 * cbrtM:
        return [-p, -p, -p]
    if abs(q) > sqrtM:
        return [-cbrt(v), -cbrt(v), -cbrt(v)]
    if abs(u) > 3 * cbrtM / 4:
        return [cbrt(4)*u / 3, cbrt(4)*u / 3, cbrt(4)*u / 3]
    if j > 0:
        # One real root
        w = sqrt(j)
        if (v > 0):
            y = (u / 3)*cbrt(2 / (w + v)) - cbrt((w + v) / 2) - p / 3
        else:
            y = cbrt((w - v) / 2) - (u / 3)*cbrt(2 / (w - v)) - p / 3
        return [y, y, y]
    # Three real roots
    s = sqrt(-u / 3)
    t = -v / (2 * s**3)
    k = acos(t) / 3

    y1 = 2 * s*cos(k) - p / 3
    y2 = s*(-cos(k) + sqrt(3.)*sin(k)) - p / 3
    y3 = s*(-cos(k) - sqrt(3.)*sin(k)) - p / 3

    return [y1, y2, y3]

