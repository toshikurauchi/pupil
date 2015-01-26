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

def point_line_distance(point, line):
    return norm((line.origin - point) - (line.origin - point).T.dot(line.direction)[0,0]*line.direction)