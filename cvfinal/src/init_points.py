import numpy as np
import cv2
from array import array

xPoints = array('f')
yPoints = array('f')
nbModelPoints = 40
counter = nbModelPoints

def getModelPoints(img):
    image = img.copy()
    cv2.namedWindow('points')
    cv2.cv.SetMouseCallback('points', mouseCallback, image)
    cv2.imshow('points', image)
    cv2.waitKey(0)
    
    pointsLength = len(xPoints)
    points = np.zeros((pointsLength, 2))
    for i in range(pointsLength):
        points[i,0] = xPoints[i]
        points[i,1] = yPoints[i]
        
    print points
    return points

def mouseCallback(event, xStacked, y, flags, param):
    global counter
    if event == cv2.cv.CV_EVENT_LBUTTONUP:
        if counter <= 0:
            print 'Enough model points.'
        else:
            xPoints.append(xStacked)
            yPoints.append(y)
            cv2.circle(param, (xStacked,y), 3, 255, thickness=-1)
            cv2.imshow('points', param)
            counter = counter - 1
            print str(counter) + ' to go'
            
