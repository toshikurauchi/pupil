'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)

 Based on the code by Leszek Swirski (https://github.com/LeszekSwirski/singleeyefitter)
'''


#logging
import logging
logger = logging.getLogger(__name__)

from eye_model_3d import EyeModel3D
from pupil3d import Pupil3D