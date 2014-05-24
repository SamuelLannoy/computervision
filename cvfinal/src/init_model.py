import numpy as np
import cv2
from array import array

'''
Created on 24-mei-2014

@author: samue_000
'''

xPoints = array('f')
yPoints = array('f')

def getModelPoints(image):
    cv2.namedWindow('points')
    cv2.cv.SetMouseCallback('points', mouseCallback, image)
    cv2.imshow('points', image)
    cv2.waitKey(0)
    
    pointsLength = len(xPoints)
    points = np.zeros((pointsLength, 2))
    for i in range(pointsLength):
        points[i,0] = xPoints[i]
        points[i,1] = yPoints[i]
        
    return points

def mouseCallback(event, x, y, flags, param):
    if event == cv2.cv.CV_EVENT_LBUTTONUP:
        xPoints.append(x)
        yPoints.append(y)
        cv2.circle(param, (x,y), 3, 255, thickness=-1)
        cv2.imshow('points', param)