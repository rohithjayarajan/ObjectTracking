"""
@file    NaiveTracker.py
@author  rohithjayarajan
@date 05/22/2018

Licensed under the
GNU General Public License v3.0
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import skimage
from sklearn.cluster import KMeans


class colorProcess:

    def __init__(self, inputImage):
        self.inputImage = inputImage

    def spaceProcess(self, lo1, hi1, lo2, hi2, lo3, hi3):

        lo = np.array([lo1, lo2, lo3])
        hi = np.array([hi1, hi2, hi3])

        mask = cv2.inRange(self.inputImage, lo, hi)
        outputImage = cv2.bitwise_and(
            self.inputImage, self.inputImage, mask=mask)

        return outputImage, mask


def videoManipulation(videoPath, writePath):

    vid = cv2.VideoCapture('%s' % (videoPath))
    out = cv2.VideoWriter(writePath, -1, 29.96, (1920, 1080))
    font = cv2.FONT_HERSHEY_SIMPLEX

    frameNumber = 1

    while(vid.isOpened()):
        ret, frame = vid.read()

        if ret == True:
            frame = cv2.GaussianBlur(frame, (5, 5), 0.6)
            RGBcartImg = frame
            HSVcartImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            RGBprocess = colorProcess(RGBcartImg)
            HSVprocess = colorProcess(HSVcartImg)

            finalOP1 = frame
            finalOP2 = HSVcartImg
            cv2.putText(finalOP2, "HSV space", (125, 125),
                        font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

            RGBdoor, RGBmask = RGBprocess.spaceProcess(
                62, 106, 59, 102, 53, 103)
            HSVdoor, HSVmask = HSVprocess.spaceProcess(
                75, 180, 0, 255, 55, 255)

            finalOP3 = RGBmask
            cv2.putText(finalOP3, "RBG mask", (125, 125),
                        font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            finalOP4 = HSVmask
            cv2.putText(finalOP4, "HSV mask", (125, 125),
                        font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

            mask = cv2.bitwise_and(RGBmask, HSVmask)
            dilateKernel1 = np.ones((7, 7), np.uint8)
            dilatedMask = cv2.dilate(mask, dilateKernel1, iterations=2)

            openKernel = np.ones((40, 40), np.uint8)
            openedMask = cv2.morphologyEx(
                dilatedMask, cv2.MORPH_OPEN, openKernel)

            dilateKernel2 = np.ones((1, 1), np.uint8)
            effMask = cv2.dilate(openedMask, dilateKernel2, iterations=1)

            doorImage = cv2.bitwise_and(RGBcartImg, RGBcartImg, mask=effMask)
            doorImageGray = cv2.cvtColor(doorImage, cv2.COLOR_BGR2GRAY)
            (ret, thresh) = cv2.threshold(doorImageGray,
                                          0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            finalOP5 = thresh
            cv2.putText(finalOP5, "RGB + HSV mask", (125, 125),
                        font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

            _, cnts, hierarchy = cv2.findContours(thresh, 1, 2)
            area = []
            maxArea = 126288
            minArea = 114332
            for c in cnts:
                ar = cv2.contourArea(c)
                if ar > minArea and ar < maxArea:
                    cnt = c
                    break

            x, y, w, h = cv2.boundingRect(cnt)
            doorOP = cv2.rectangle(
                RGBcartImg, (x, y), (x+w, y+h), (0, 0, 255), 2)
            finalOP6 = doorOP
            cv2.putText(finalOP6, "Door", (x, y), font,
                        0.8, (0, 0, 255), 2, cv2.LINE_AA)

            doorMask = np.zeros(RGBcartImg.shape, np.uint8)
            doorMask[y:y+h, x:x+w] = RGBcartImg[y:y+h, x:x+w]

            HLScartImg = cv2.cvtColor(doorMask, cv2.COLOR_BGR2HLS)
            HLSprocess = colorProcess(HLScartImg)
            HLSdoor, HLSmask = HLSprocess.spaceProcess(
                0, 180, 140, 255, 0, 255)
            finalOP7 = HLScartImg
            cv2.putText(finalOP7, "Cart HSV space", (125, 125),
                        font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            finalOP8 = HLSmask
            cv2.putText(finalOP8, "Cart HSV mask", (125, 125),
                        font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

            dilateKernelCart = np.ones((7, 7), np.uint8)
            dilatedMaskCart = cv2.dilate(
                HLSmask, dilateKernelCart, iterations=2)

            # HLSImageGray = cv2.cvtColor(HLSmask, cv2.COLOR_BGR2GRAY)

            (ret, threshCart) = cv2.threshold(dilatedMaskCart,
                                              0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            finalOP9 = threshCart

            _, cntsCart, hierarchy = cv2.findContours(threshCart, 1, 2)
            area1 = []
            maxAreaCart = 3500.00
            minAreaCart = 3000.00
            for cCart in cntsCart:
                arCart = cv2.contourArea(cCart)
                if arCart > minAreaCart and arCart < maxAreaCart:
                    cntCart = cCart
                    areaCart = arCart
                    break

            xCart, yCart, wCart, hCart = cv2.boundingRect(cntCart)
            finalOP10 = cv2.rectangle(
                doorOP, (xCart, yCart), (xCart+wCart, yCart+hCart), (0, 255, 0), 2)
            cv2.putText(finalOP10, "Cart", (xCart, yCart),
                        font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            out.write(finalOP10)

            frameNumber += 1

        else:
            break

    vid.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    videoPath = '/home/rohith/git/ObjectTracking/Data/CartMockupDemo.mp4'
    writePath = '/home/rohith/git/ObjectTracking/Data/trackerOP.avi'

    videoManipulation(videoPath, writePath)
