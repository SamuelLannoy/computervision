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
def getModelPointsManually(personId, toothId):
    image = rg.readRadioGraph(personId)
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
        
    print points
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
Returns initial points (Point x Tooth x Dim) for the given image with automatic searching.
'''            
def getModelPointsAutomatically(personId):
    image = rg.readRadioGraph(personId)
    sumScrs = 0
    avgTeeth = np.zeros((main.nbLandmarks, main.toothIds.shape[0], 2))
    corrLM = np.zeros((main.nbLandmarks, main.toothIds.shape[0], 2))
    
    for templId in range(14):
        # get the template
        templImg, templLM = mt.getTemplate(templId, main.toothIds)
        # match the template
        (x,y), scale, scr = mt.matchTemplate(image, templImg)
        # translate the template landmarks
        for i in range(main.toothIds.shape[0]):
            toothIdx = main.toothIds[i]
            corrLM[:,toothIdx,:] = procru.scaleMatrixForPerson(templLM[:,toothIdx,:], scale)
            corrLM[:,toothIdx,:] = procru.translateMatrixForPerson(corrLM[:,toothIdx,:], np.array([[x],[y]]))
        # add the translated landmarks to the average tooth with weight = scr
        avgTeeth = avgTeeth + scr**main.templ_scr_loyalty*corrLM
        # update the sum of all scores
        sumScrs = sumScrs + scr**main.templ_scr_loyalty
        
    avgTeeth = avgTeeth/sumScrs

    for i in range(8):
        cv2.polylines(image, np.int32([avgTeeth[:,i,:]]), True, 255)
    main.showScaled(image, 0.6, 'automatic initialisation', True)
        
    return avgTeeth

'''
Return initial points (Point x Tooth x Dim) for the given image with automatic hierarchical searching.
'''
def getModelPointsHierarchically(personId):
    debugMBB = True
    image = rg.readRadioGraph(personId)
    
    # Choice of search area around the upper resp. lower inscisors within (preprocessed) image
    ## these coordinates are in the not-preprocessed images, format Part x {UpperLeft, LowerRight} x Dim
    partSearchAreas = np.array([[[1190,550], [1830,1120]],
                                [[1210,880], [1810,1400]]])
    '''
    partSearchAreas = np.array([[[523,527], [1478,1570]],
                                [[889,913], [1478,1535]]])
    '''
    ## correct for preprocessing of images
    partSearchAreas[:,:,0] = partSearchAreas[:,:,0] - np.ones_like(partSearchAreas[:,:,0])*rg.cropX[0]
    partSearchAreas[:,:,1] = partSearchAreas[:,:,1] - np.ones_like(partSearchAreas[:,:,1])*rg.cropY[0]
    
    if debugMBB:
        debugImage = image.copy()
        cv2.rectangle(debugImage, (partSearchAreas[0,0,0], partSearchAreas[0,0,1]), (partSearchAreas[0,1,0], partSearchAreas[0,1,1]), 255)
        cv2.rectangle(debugImage, (partSearchAreas[1,0,0], partSearchAreas[1,0,1]), (partSearchAreas[1,1,0], partSearchAreas[1,1,1]), 255)
        #main.showScaled(debugImage, 0.6, 'debugImage', True)
    
    # 1 FIND BOUNDING BOXES OF PARTS IN SEARCH AREA
    if debugMBB:
        sumUpScrs = 0
        sumLowScrs = 0
        '''
        avgUpTeeth = np.zeros((main.nbLandmarks, 4, 2))
        avgLowTeeth = np.zeros((main.nbLandmarks, 4, 2))
        '''
    ## get search images (image cropped to resp. search area)
    upperImage = image.copy()[partSearchAreas[0,0,1]:partSearchAreas[0,1,1], partSearchAreas[0,0,0]:partSearchAreas[0,1,0]]
    lowerImage = image.copy()[partSearchAreas[1,0,1]:partSearchAreas[1,1,1], partSearchAreas[1,0,0]:partSearchAreas[1,1,0]]
    
    upMBBUL = np.zeros((2,))
    upMBBLR = np.zeros((2,))
    lowMBBUL = np.zeros((2,))
    lowMBBLR = np.zeros((2,))
    
    for templId in range(14):
        ## get templates for these parts
        upTemplImg, upTemplLMs = mt.getTemplate(templId, np.array(range(4)), Delta=0)
        lowTemplImg, lowTemplLMs = mt.getTemplate(templId, np.array(range(4,8)), Delta=0)
        ## match templates
        (upX,upY), upScale, upScore = mt.matchTemplate(upperImage, upTemplImg)
        (lowX,lowY), lowScale, lowScore = mt.matchTemplate(lowerImage, lowTemplImg)
        ## avoid deviding by zero
        if upScore == 0 : upScore = 1
        if lowScore == 0 : lowScore = 1
        ## calculate matched bounding box (MBB) corners, relative to the preprocessed image (image)
        upMBBUL = upMBBUL + (upScore**main.templ_scr_loyalty)*( np.array([upX,upY]) + partSearchAreas[0,0,:] )
        upMBBLR = upMBBLR + (upScore**main.templ_scr_loyalty)*( np.array([upX,upY]) + partSearchAreas[0,0,:]
                                                                          + np.array([upTemplImg.shape[1]*upScale,
                                                                                      upTemplImg.shape[0]*upScale])
                                                               )
        lowMBBUL = lowMBBUL + (lowScore**main.templ_scr_loyalty)*( np.array([lowX,lowY]) + partSearchAreas[1,0,:] )
        lowMBBLR = lowMBBLR + (lowScore**main.templ_scr_loyalty)*( np.array([lowX,lowY]) + partSearchAreas[1,0,:]
                                                                             + np.array([lowTemplImg.shape[1]*lowScale,
                                                                                         lowTemplImg.shape[0]*lowScale])
                                                                  )
            
        # update the sum of all scores
        sumUpScrs = sumUpScrs + upScore**main.templ_scr_loyalty
        sumLowScrs = sumLowScrs + lowScore**main.templ_scr_loyalty

    ## calculate average MBB and correct for search area
    upMBBUL = np.int32(upMBBUL/sumUpScrs)
    upMBBLR = np.int32(upMBBLR/sumUpScrs)
    lowMBBUL = np.int32(lowMBBUL/sumLowScrs)
    lowMBBLR = np.int32(lowMBBLR/sumLowScrs)
    
    #if debugMBB:
        #cv2.rectangle(debugImage, (upMBBUL[0], upMBBUL[1] ), (upMBBLR[0], upMBBLR[1] ), 255)
        #cv2.rectangle(debugImage, (lowMBBUL[0],lowMBBUL[1]), (lowMBBLR[0],lowMBBLR[1]), 255)
    
    #2 FIND LOCATION OF TEETH IN FOUND BOUNDING BOXES
    
    # Choise of search space for each inscisor
    ## the numbers in this table are the relative, horizontal borders of
    ##   the search area within the matched bounding box of the resp. part, format Tooth x {LeftBorder, RightBorder}
    searchBoxBorders = np.array([[0.0,0.3], [0.2,0.6], [0.4,0.8], [0.7,1.0],
                                 [0.0,0.3], [0.2,0.6], [0.4,0.8], [0.7,1.0]])
    
    searchBoxBorders = np.array([[0.0,0.25], [0.25,0.5], [0.5,0.75], [0.75,1.0],
                                 [0.0,0.25], [0.25,0.5], [0.5,0.75], [0.75,1.0]])
    
    ## widths of the parts
    upDeltaX = upMBBLR[0]-upMBBUL[0]
    lowDeltaX = lowMBBLR[0]-lowMBBUL[0]
    
    print upDeltaX, lowDeltaX
    
    ## Choice of extra space to add around the searchboxes
    extraSpace = 30
    
    searchBoxes = np.zeros((2,4,2,2)) # Part x ToothInPart x {UpperLeft, LowerRight} x Dim
    ## calculate search boxes for teeth
    for toothIdx in range(4):
        searchBoxes[0,toothIdx,0,0] = upMBBUL[0] + upDeltaX*searchBoxBorders[toothIdx,0] - extraSpace # upper part, upper left x
        searchBoxes[0,toothIdx,0,1] = upMBBUL[1] - extraSpace # upper part, upper left y
        searchBoxes[0,toothIdx,1,0] = upMBBUL[0] + upDeltaX*searchBoxBorders[toothIdx,1] + extraSpace # upper part, lower right x
        searchBoxes[0,toothIdx,1,1] = upMBBLR[1] + extraSpace # upper part, lower right y
        cv2.rectangle(debugImage, (np.int(searchBoxes[0,toothIdx,0,0]), np.int(searchBoxes[0,toothIdx,0,1])),
                                  (np.int(searchBoxes[0,toothIdx,1,0]), np.int(searchBoxes[0,toothIdx,1,1])), 255)
    for toothIdx in range(4,8):
        searchBoxes[1,toothIdx%4,0,0] = lowMBBUL[0] + lowDeltaX*searchBoxBorders[toothIdx,0] - extraSpace # lower part, upper left x
        searchBoxes[1,toothIdx%4,0,1] = lowMBBUL[1] - extraSpace # lower part, upper left y
        searchBoxes[1,toothIdx%4,1,0] = lowMBBUL[0] + lowDeltaX*searchBoxBorders[toothIdx,1] + extraSpace # lower part, lower right x
        searchBoxes[1,toothIdx%4,1,1] = lowMBBLR[1] + extraSpace # lower part, lower right y
        cv2.rectangle(debugImage, (np.int(searchBoxes[1,toothIdx%4,0,0]), np.int(searchBoxes[1,toothIdx%4,0,1])),
                                  (np.int(searchBoxes[1,toothIdx%4,1,0]), np.int(searchBoxes[1,toothIdx%4,1,1])), 255)
    
    main.showScaled(debugImage, 0.6, 'debugImage', True)
        
        
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
MAIN PROGRAM
'''
if __name__ == '__main__':
    for i in range(14):
        #getModelPointsAutomatically(i)
        getModelPointsHierarchically(i)
        print i
