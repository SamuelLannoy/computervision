import cv2
import numpy as np

import main
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
    minX = landmarks[:,:,0].min()-10
    maxX = landmarks[:,:,0].max()+10
    minY = landmarks[:,:,1].min()-10
    maxY = landmarks[:,:,1].max()+10
    
    image = image[minY:maxY,minX:maxX]
    
    for toothId in range(8):
        landmarks[:,toothId,:] = procru.translateMatrixForPerson(landmarks[:,toothId,:], np.array([[-minX],[-minY]]))
    
    return image, landmarks

def testCreateTemplate():
    for personId in range(14):
        print str(personId+1)
        image, landmarks = createTemplate(cv2.imread(rg.radiographPath(personId, True),0), lm.readLandmarksOfPerson(personId))
        for toothId in range(main.toothIds.shape[0]):
            cv2.polylines(image, np.int32([landmarks[:,toothId,:]]), True, 255)
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

if __name__ == '__main__':
    testCreateTemplate()
    
    
    '''
    personId = 6
    imgToMatch = cv2.imread(main.radiographPath(personId, True),0)
    template = cv2.imread('../data/Templates/01u.png',0)
    img = cv2.matchTemplate(imgToMatch, template, cv2.cv.CV_TM_SQDIFF_NORMED)
    print cv2.minMaxLoc(img) 
    cv2.imshow('img', img)
    cv2.waitKey(0)
    '''
