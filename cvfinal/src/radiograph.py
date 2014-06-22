import numpy as np
import cv2

cropX = (1100,1900)
cropY = (450,1550)

def preprocess(img):
    cropped = img[cropY[0]:cropY[1],cropX[0]:cropX[1]]
    denoised = cv2.fastNlMeansDenoising(cropped) #cv2.bilateralFilter(cropped, 9, 150, 150)  cv2.fastNlMeansDenoising(cropped) #h=5 
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8,8)); #16x16
    return clahe.apply(denoised)

def sharpen(img):
    kernel = np.array([[0,0,0], [0,2,0], [0,0,0]]) - (1.0/9.0) * np.array([[1,1,1], [1,1,1], [1,1,1]])
    return cv2.filter2D(img, -1, kernel)

'''
Returns a path to the radiograph for the given personId (counting from 0).
'''
def radiographPath(personId, preprocessed):
    path = '../data/Radiographs/'
    if personId < 9 :
        path = path + '0' + str(personId+1)
    elif personId < 14:
        path = path + str(personId+1)
    else:
        path = path + 'extra/' + str(personId+1)
    
    if preprocessed: return path + 'p.tif'
    else: return path + '.tif'

'''
Reads radiographs for the given personIds (counting from 0)
'''
def readRadiographs(personIds):
    xDim = cropX[1] - cropX[0]
    yDim = cropY[1] - cropY[0]
    
    images = np.zeros((personIds.shape[0], yDim, xDim))
    for i in range(personIds.shape[0]):
        # if already preprocessed images are available, turn flag to true
        preprocessed = True
        path = radiographPath(personIds[i], preprocessed)
        if preprocessed : images[i] = cv2.imread(path,0)
        else : images[i] = preprocess(cv2.imread(path,0))
    
    return images

def readRadioGraph(personId):
    # if already preprocessed images are available, turn flag to true
    preprocessed = True
    image = cv2.imread(radiographPath(personId, preprocessed),0)
    if preprocessed : return image
    else : return preprocess(image)

'''
Updates the preprocessed radiographs on disk for the given personIds (counting from 0).
'''
def updatePreprocessedRadioGraphs(personIds):
    for personId in personIds:
        matched = cv2.imread(radiographPath(personId, False),0)
        matched = preprocess(matched)
        cv2.imwrite(radiographPath(personId, True), matched)
        #cv2.imshow('preprocessed', matched)
        print 'Image ' + str(personId+1) + ' is processed'
        #cv2.waitKey(0)

'''
MAIN PROGRAM
'''
if __name__ == '__main__':
    updatePreprocessedRadioGraphs(range(30))
    
'''
OLD

if __name__ == '__main__':    
    # THIS ROUTINE LETS YOU PREPROCESS IMAGES
    for personId in range(1,15):
    #personId = 1
        img = cv2.imread('../data/Radiographs/' + ("0" + str(personId) if personId < 10 else str(personId)) + '.tif',0)
        img = preprocess(img)
        cv2.imwrite('../data/Radiographs/' + ("0" + str(personId) if personId < 10 else str(personId)) + 'p.tif', img)
        print 'Image ' + str(personId) + ' is processed!'
        #cv2.imshow('preprocessed', img)
'''
