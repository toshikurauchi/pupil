'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import os
import cv2
import numpy as np
from methods import normalize,denormalize
from gl_utils import draw_gl_point,adjust_gl_view,draw_gl_point_norm,draw_gl_polyline,clear_gl_screen,basic_gl_setup
import OpenGL.GL as gl
from glfw import *
from OpenGL.GLU import gluOrtho2D
import calibrate
from circle_detector import get_canditate_ellipses
from planarization import EyeModel3D, Pupil3D

from ctypes import c_int,c_bool
import atb
import audio

from plugin import Plugin

#logging
import logging
logger = logging.getLogger(__name__)


def draw_circle(pos,r,c):
    pts = cv2.ellipse2Poly(tuple(pos),(r,r),0,0,360,10)
    draw_gl_polyline(pts,c,'Polygon')

def draw_marker(pos):
    pos = int(pos[0]),int(pos[1])
    black = (0.,0.,0.,1.)
    white = (1.,1.,1.,1.)
    s = 2
    for r,c in zip((50,40,30,20,10),(black,white,black,white,black)):
        draw_circle(pos,s*r,c)


# window calbacks
def on_resize(window,w, h):
    active_window = glfwGetCurrentContext()
    glfwMakeContextCurrent(window)
    adjust_gl_view(w,h)
    glfwMakeContextCurrent(active_window)


class Screen_Marker_VOR_Calibration(Plugin):
    """
    Calibrate using a marker on your screen
    We use a ring detector that moves across the screen to 9 sites
    Points are collected at sites not between
    """
    def __init__(self, g_pool, atb_pos=(0,0)):
        Plugin.__init__(self)
        self.g_pool = g_pool
        self.active = False
        self.detected = False
        self.display_pos = (.5, .5)

        self.candidate_ellipses = []
        self.pos = None

        self.show_edges = c_bool(0)
        self.dist_threshold = c_int(5)
        self.area_threshold = c_int(20)

        self.world_size = None

        self._window = None
        self.window_should_close = False
        self.window_should_open = False
        self.fullscreen = c_bool(1)
        self.monitor_idx = c_int(0)
        monitor_handles = glfwGetMonitors()
        self.monitor_names = [glfwGetMonitorName(m) for m in monitor_handles]
        monitor_enum = atb.enum("Monitor",dict(((key,val) for val,key in enumerate(self.monitor_names))))
        #primary_monitor = glfwGetPrimaryMonitor()



        atb_label = "calibrate on screen"
        # Creating an ATB Bar is required. Show at least some info about the Ref_Detector
        self._bar = atb.Bar(name = self.__class__.__name__, label=atb_label,
            help="ref detection parameters", color=(50, 50, 50), alpha=100,
            text='light', position=atb_pos,refresh=.3, size=(300, 90))
        self._bar.add_var("monitor",self.monitor_idx, vtype=monitor_enum)
        self._bar.add_var("fullscreen", self.fullscreen)
        self._bar.add_button("  start calibrating  ", self.start, key='c')

        self._bar.add_var("show edges",self.show_edges, group="Detector Variables")
        self._bar.add_var("area threshold", self.area_threshold ,group="Detector Variables")
        self._bar.add_var("eccetricity threshold", self.dist_threshold, group="Detector Variables" )


    def start(self):
        if self.active:
            return

        audio.say("Starting Calibration")
        logger.info("Starting Calibration")
        self.active = True
        self.ref_list = []
        self.pupil_list = []
        self.pupil_ellipses = []
        self.window_should_open = True
        self.img_origin = None

    def open_window(self):
        if not self._window:
            if self.fullscreen.value:
                monitor = glfwGetMonitors()[self.monitor_idx.value]
                mode = glfwGetVideoMode(monitor)
                height,width= mode[0],mode[1]
            else:
                monitor = None
                height,width= 640,360

            self._window = glfwCreateWindow(height, width, "Calibration", monitor=monitor, share=glfwGetCurrentContext())
            if not self.fullscreen.value:
                glfwSetWindowPos(self._window,200,0)

            on_resize(self._window,height,width)

            #Register callbacks
            glfwSetWindowSizeCallback(self._window,on_resize)
            glfwSetKeyCallback(self._window,self.on_key)
            glfwSetWindowCloseCallback(self._window,self.on_close)

            # gl_state settings
            active_window = glfwGetCurrentContext()
            glfwMakeContextCurrent(self._window)
            basic_gl_setup()
            # refresh speed settings
            glfwSwapInterval(0)

            glfwMakeContextCurrent(active_window)
            self.window_should_open = False


    def on_key(self,window, key, scancode, action, mods):
        if not atb.TwEventKeyboardGLFW(key,int(action == GLFW_PRESS)):
            if action == GLFW_PRESS:
                if key == GLFW_KEY_ESCAPE:
                    self.stop()

    def on_close(self,window=None):
        if self.active:
            self.stop()

    def stop(self):
        audio.say("Stopping Calibration")
        logger.info('Stopping Calibration')
        self.active = False
        self.window_should_close = True

        cal_pt_cloud = calibrate.preprocess_data(self.pupil_list,self.ref_list)

        logger.info("Collected %s data points." %len(cal_pt_cloud))

        if len(cal_pt_cloud) < 20:
            logger.warning("Did not collect enough data.")
            return

        self.g_pool.objs['eye_model3d'] = EyeModel3D(self.pupil_ellipses, self.img_origin)

        logger.info("3D eye model initialized.")

        cal_pt_cloud = np.array(cal_pt_cloud)
        map_fn = calibrate.get_map_from_cloud(cal_pt_cloud,self.world_size)
        self.g_pool.map_pupil = map_fn
        np.save(os.path.join(self.g_pool.user_dir,'cal_pt_cloud.npy'),cal_pt_cloud)

    def close_window(self):
        if self._window:
            glfwDestroyWindow(self._window)
            self._window = None
            self.window_should_close = False


    def update(self,frame,recent_pupil_positions,events):
        if self.window_should_close:
            self.close_window()

        if self.window_should_open:
            self.open_window()

        if self.active:
            img = frame.img
            if self.img_origin is None:
                h,w = img.shape[0], img.shape[1]
                self.img_origin = (w/2,h/2)

            #get world image size for error fitting later.
            if self.world_size is None:
                self.world_size = img.shape[1],img.shape[0]

            #detect the marker
            self.candidate_ellipses = get_canditate_ellipses(img,
                                                            area_threshold=self.area_threshold.value,
                                                            dist_threshold=self.dist_threshold.value,
                                                            min_ring_count=4,
                                                            visual_debug=self.show_edges.value)

            if len(self.candidate_ellipses) > 0:
                self.detected= True
                marker_pos = self.candidate_ellipses[0][0]
                self.pos = normalize(marker_pos,(img.shape[1],img.shape[0]),flip_y=True)
                self.prev_img = img
            elif self.detected: #was detected on the previous frame
                prev_pos = np.array([denormalize(self.pos,(img.shape[1],img.shape[0]),flip_y=True)],dtype=np.float32)
                new_pos, st, err = cv2.calcOpticalFlowPyrLK(self.prev_img, img, prev_pos)
                if st[0]:
                    self.pos = normalize(new_pos[0],(img.shape[1],img.shape[0]),flip_y=True)
                else:
                    self.detected = False
                self.prev_img = img
            else:
                self.detected = False
                self.pos = None #indicate that no reference is detected


            if self.detected:
                ref = {}
                ref["norm_pos"] = self.pos
                ref["timestamp"] = frame.timestamp
                self.ref_list.append(ref)

            last_p_pt = None
            for p_pt in recent_pupil_positions:
                if p_pt['norm_pupil'] is not None:
                    # Store all pupil ellipses for 3D eye model fitting
                    if p_pt['confidence'] > 0.8 and p_pt.has_key('axes'):
                        ellipse = (p_pt['center'], p_pt['axes'], p_pt['angle'])
                        self.pupil_ellipses.append(Pupil3D(None, ellipse, False))
                    # Use only the last position for calibration (because the
                    # eye is moving the previous positions are not related to
                    # the current target position)
                    last_p_pt = p_pt
            if self.detected and last_p_pt is not None:
                self.pupil_list.append(last_p_pt)


    def gl_display(self):
        """
        use gl calls to render
        at least:
            the published position of the reference
        better:
            show the detected postion even if not published
        """

        if self.active and self.detected:
            for e in self.candidate_ellipses:
                pts = cv2.ellipse2Poly( (int(e[0][0]),int(e[0][1])),
                                    (int(e[1][0]/2),int(e[1][1]/2)),
                                    int(e[-1]),0,360,15)
                draw_gl_polyline(pts,(0.,1.,0,1.))
        else:
            pass
        if self._window:
            self.gl_display_in_window()


    def gl_display_in_window(self):
        active_window = glfwGetCurrentContext()
        glfwMakeContextCurrent(self._window)

        clear_gl_screen()

        # Set Matrix unsing gluOrtho2D to include padding for the marker of radius r
        #
        ############################
        #            r             #
        # 0,0##################w,h #
        # #                      # #
        # #                      # #
        #r#                      #r#
        # #                      # #
        # #                      # #
        # 0,h##################w,h #
        #            r             #
        ############################
        r = 60
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        p_window_size = glfwGetWindowSize(self._window)
        # compensate for radius of marker
        gluOrtho2D(-r,p_window_size[0]+r,p_window_size[1]+r, -r) # origin in the top left corner just like the img np-array
        # Switch back to Model View Matrix
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        screen_pos = denormalize(self.display_pos,p_window_size,flip_y=True)

        draw_marker(screen_pos)
        #some feedback on the detection state

        if self.detected:
            draw_gl_point(screen_pos, 5, (0.,1.,0.,1.))
        else:
            draw_gl_point(screen_pos, 5, (1.,0.,0.,1.))

        glfwSwapBuffers(self._window)
        glfwMakeContextCurrent(active_window)


    def cleanup(self):
        """gets called when the plugin get terminated.
           either volunatily or forced.
        """
        if self.active:
            self.stop()
        if self._window:
            self.close_window()
        self._bar.destroy()
