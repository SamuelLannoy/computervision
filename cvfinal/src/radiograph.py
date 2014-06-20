import numpy as np
import cv2

cropX = (1100,1900)
cropY = (450,1550)

def preprocess(matched):
    cropped = matched[cropY[0]:cropY[1],cropX[0]:cropX[1]]
    denoised = cv2.fastNlMeansDenoising(cropped)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8));
    return clahe.apply(denoised)

def sharpen(matched):
    kernel = np.array([[0,0,0], [0,1.25,0], [0,0,0]]) - (1/9) * np.array([[1,1,1], [1,1,1], [1,1,1]])
    return cv2.filter2D(matched, -1, kernel)

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