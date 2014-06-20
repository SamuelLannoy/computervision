import numpy as np
import cv2

import radiograph as rg
from array import array

xPoints = array('f')
yPoints = array('f')
nbModelPoints = 40
counter = nbModelPoints
scale = np.float(733)/np.float(rg.cropY[1]-rg.cropY[0])

def getModelPoints(matched):
    return getModelPointsManually(matched)

def getModelPointsManually(matched):
    image = matched.copy()
    cv2.namedWindow('points')
    image = cv2.resize(image, (0,0), fx=scale, fy=scale)
    
    cv2.cv.SetMouseCallback('points', mouseCallback, image)
    
    cv2.imshow('points', image)
    cv2.waitKey(0)
    
    pointsLength = len(xPoints)
    points = np.zeros((pointsLength, 2))
    for i in range(pointsLength):
        points[i,0] = xPoints[i]/scale
        points[i,1] = yPoints[i]/scale
        
    return points

def mouseCallback(event, x, y, flags, param):
    global counter
    
    debugFB = False
    
    if event == cv2.cv.CV_EVENT_LBUTTONUP:
        if debugFB : print 'DB: clicked on (' + str(x) + ', ' + str(y) + ')'
        if counter <= 0:
            print 'Enough model points.'
        else:
            xPoints.append(x)
            yPoints.append(y)
            cv2.circle(param, (x,y), 3, 255, thickness=-1)
            cv2.imshow('points', param)
            counter = counter - 1
            print str(counter) + ' to go'
            
def getModelPointsAutomatically(matched):
    image = matched.copy()
    
    y_min = np.argmin(np.average(image, 1))
    line = np.int32([np.array([[0,y_min],[image.shape[1]-1, y_min]])])
    
    cv2.polylines(image, line, False, 255)
    
    cv2.imshow('result', image)
    cv2.waitKey(0)
    
'''
MAIN PROGRAM
'''
if __name__ == '__main__':
    for i in range(30):
        image = rg.readRadioGraph(i)
        print str(i+1)
        getModelPointsAutomatically(image)

