import numpy as np
import cv2

cropX = (1100,1900)
cropY = (550,1450)

def preprocess(matched):
    cropped = matched[cropY[0]:cropY[1],cropX[0]:cropX[1]]
    denoised = cv2.fastNlMeansDenoising(cropped)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8));
    return clahe.apply(denoised)

def sharpen(matched):
    kernel = np.array([[0,0,0], [0,1.25,0], [0,0,0]]) - (1/9) * np.array([[1,1,1], [1,1,1], [1,1,1]])
    return cv2.filter2D(matched, -1, kernel)

'''
MAIN PROGRAM
'''
if __name__ == '__main__':    
    # THIS ROUTINE LET YOU PREPROCESS IMAGES
    for personId in range(1,15):
        matched = cv2.imread('../data/Radiographs/' + ("0" + str(personId) if personId < 10 else str(personId)) + '.tif',0)
        matched = preprocess(matched)
        cv2.imwrite('../data/Radiographs/' + ("0" + str(personId) if personId < 10 else str(personId)) + 'p.tif', matched)
        #cv2.imshow('preprocessed', matched)
        print 'Image ' + str(personId) + ' is processed!'
        #cv2.waitKey(0)


