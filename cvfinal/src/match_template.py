import cv2
import numpy as np

import radiograph as rg
import landmarks as lm
import procrustes as procru

'''
A template is a tuple (img,lms) with img is the image of the
template, and lms are the (translated) landmarks of the template.
'''

'''
Creates a template from the given image and landmarks (LM x Tooth x Dim).

The given image is presumed to be preprocessed, ie. no extra preprocessing will happen in this function.
The given landmarks are presumed to be in the coordinate system of the image.
'''
def createTemplate(image, landmarks):
    minX = landmarks[:,:,0].min()-20
    maxX = landmarks[:,:,0].max()+20
    minY = landmarks[:,:,1].min()-20
    maxY = landmarks[:,:,1].max()+20
    
    image = image[minY:maxY,minX:maxX]
    
    for toothId in range(8):
        landmarks[:,toothId,:] = procru.translateMatrixForPerson(landmarks[:,toothId,:], np.array([[-minX],[-minY]]))
    
    return image, landmarks

def testCreateTemplate():
    for personId in range(14):
        print str(personId+1)
        image, landmarks = createTemplate(cv2.imread(rg.radiographPath(personId, True),0), lm.readLandmarksOfPerson(personId, 40))
        for toothId in range(8):
            cv2.polylines(image, np.int32([landmarks[toothId,:,:]]), True, 255)
        cv2.imshow('result', image)
        cv2.waitKey(0)

'''
Returns a triplet (Delta_X, Delta_Y, score) which is the best matching location of the template image in the given image
and the score of that match.
'''
def matchTemplate(image, templImage):
    # TODO: scaling and rotation?
    matches = cv2.matchTemplate(image, templImage, cv2.cv.CV_TM_CCORR_NORMED)
    '''
    cv2.imshow('matches', matches)
    cv2.waitKey(0)
    '''
    _, maxVal, _, maxLoc = cv2.minMaxLoc(matches)
    return maxLoc[0], maxLoc[1], maxVal

def testMatchTemplate():
    for personId in range(14,30):
        image = rg.readRadioGraph(personId)
        
        chosenScr = 0
        for templId in range(14):
            template = createTemplate(rg.readRadioGraph(templId), lm.readLandmarksOfPerson(templId, 40))
            x, y, scr = matchTemplate(image, template[0])
            if scr > chosenScr:
                chosenScr = scr
                chosenX = x
                chosenY = y
                chosenTemplate = template
                chosenTemplId = templId
                
        print (personId, chosenTemplId)
        
        combined = image.copy()
        combined[x:x+template[0].shape[0], y:y+template[0].shape[1]] = template[0]
        cv2.circle(combined, (y,x), 3, 255, thickness=-1)
        cv2.imshow('combined', combined)
        
        init_image = image.copy()
        for toothId in range(8):
            init_points = procru.translateMatrixForPerson(chosenTemplate[1][toothId,:,:], np.array([[chosenX],[chosenY]]))
            cv2.polylines(init_image, np.int32([init_points]), True, 255)
        cv2.imshow('init_image',init_image)
        cv2.waitKey(0)

if __name__ == '__main__':
    testMatchTemplate()
    
    
    '''
    personId = 6
    imgToMatch = cv2.imread(main.radiographPath(personId, True),0)
    template = cv2.imread('../data/Templates/01u.png',0)
    matched = cv2.matchTemplate(imgToMatch, template, cv2.cv.CV_TM_SQDIFF_NORMED)
    print cv2.minMaxLoc(matched) 
    cv2.imshow('matched', matched)
    cv2.waitKey(0)
    '''