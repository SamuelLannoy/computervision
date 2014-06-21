import numpy as np
import cv2
from array import array

import radiograph as rg
import landmarks as lm
import procrustes as procru
import match_template as mt
import main

xPoints = array('f')
yPoints = array('f')
nbModelPoints = 40
counter = nbModelPoints

'''
Returns initial points (Point x Dim) for the given image with manually clikcing them.
'''
def getModelPointsManually(matched, toothId):
    image = matched.copy()
    image = cv2.resize(image, (0,0), fx=main.windowscale, fy=main.windowscale)
    
    cv2.namedWindow('points')
    cv2.cv.SetMouseCallback('points', mouseCallback, image)
    
    cv2.imshow('points', image)
    
    print 'Click initial points for tooth #' + str(toothId)
    
    cv2.waitKey(0)
    
    pointsLength = len(xPoints)
    points = np.zeros((pointsLength, 2))
    for i in range(pointsLength):
        points[i,0] = xPoints[i]/main.windowscale
        points[i,1] = yPoints[i]/main.windowscale
        
    return points

'''
Catches mouse click for manual initialization.
'''
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

'''
Returns initial points (Point x Dim) for the given image with automatic searching.
'''            
def getModelPointsAutomatically(image, toothId):
    chosenScr = 0
    for templId in range(14):
        template = mt.createTemplate(rg.readRadioGraph(templId), lm.readLandmarksOfPerson(templId, 40))
        x, y, scr = mt.matchTemplate(image, template[0])
        if scr > chosenScr:
            chosenScr = scr
            chosenX = x
            chosenY = y
            chosenTemplate = template
    '''
    combined = image.copy()
    combined[x:x+template[0].shape[0], y:y+template[0].shape[1]] = template[0]
    cv2.circle(combined, (y,x), 3, 255, thickness=-1)
    cv2.imshow('combined', combined)
    '''
    init_points = procru.translateMatrixForPerson(chosenTemplate[1][toothId,:,:], np.array([[chosenX],[chosenY]]))
    '''
    init_image = image.copy()
    cv2.polylines(init_image, np.int32([init_points]), True, 255)
    cv2.imshow('init_image',init_image)
    cv2.waitKey(0)
    '''
    return init_points
    
    
'''
Finds the middle of the teeth by finding the darkest horizontal line in the image.
'''
def getVerticalMiddleOfTeeth(image):
    y_min = np.argmin(np.average(image, 1))
    line = np.int32([np.array([[0,y_min],[image.shape[1]-1, y_min]])])
    
    cv2.polylines(image, line, False, 255)
    
    cv2.imshow('result', image)
    cv2.waitKey(0)

'''
Returns initial points (Point x Dim) for the given image.
'''
def getModelPoints(image, toothId):
    auto = True
    if auto : return getModelPointsAutomatically(image, toothId)
    else :    return getModelPointsManually(image, toothId)

'''
MAIN PROGRAM
'''
if __name__ == '__main__':
    for i in range(30):
        image = rg.readRadioGraph(i)
        print str(i+1)
        getVerticalMiddleOfTeeth(image)
