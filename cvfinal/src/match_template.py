import cv2
import numpy as np
import main
import procrustes as procru

'''
A template is a tuple (img,lms) with img is the image of the
template, and lms are the (translated) landmarks of the template.
'''

'''
Creates a template from the given image and landmarks (Tooth x LM x Dim).

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
        landmarks[toothId,:,:] = procru.translateMatrixForPerson(landmarks[toothId,:,:], np.array([[-minX],[-minY]]))
    
    return image, landmarks

def testCreateTemplate():
    for personId in range(14):
        print str(personId+1)
        image, landmarks = createTemplate(cv2.imread(main.radiographPath(personId, True),0), main.readLandmarksOfPerson(personId, 40))
        for toothId in range(8):
            cv2.polylines(image, np.int32([landmarks[toothId,:,:]]), True, 255)
        cv2.imshow('result', image)
        cv2.waitKey(0)

if __name__ == '__main__':
    testCreateTemplate()
    
    '''
    personId = 6
    imgToMatch = cv2.imread(main.radiographPath(personId, True),0)
    template = cv2.imread('../data/Templates/01u.png',0)
    matched = cv2.matchTemplate(imgToMatch, template, cv2.cv.CV_TM_SQDIFF_NORMED)
    print cv2.minMaxLoc(matched) 
    cv2.imshow('matched', matched)
    cv2.waitKey(0)
    '''