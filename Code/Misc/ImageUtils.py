"""
@file    ImageUtils.py
@author  rohithjayarajan
@date 05/22/2018

Licensed under the
GNU General Public License v3.0
"""

import numpy as np
import cv2
from scipy import interpolate
import math


class ImageUtils:

    def imToFrames(self, videoPath, framesPath):

        vid = cv2.VideoCapture('%s' % (videoPath))
        frameNumber = 1

        while(vid.isOpened()):
            ret, frame = vid.read()

            if ret == True:
                frame = cv2.GaussianBlur(frame, (5, 5), 0.6)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                cv2.imwrite('%s' % (framesPath) + "/%#04d.jpg" %
                            (frameNumber), frame)
                frameNumber += 1

            else:
                break

        vid.release()
        cv2.destroyAllWindows()
