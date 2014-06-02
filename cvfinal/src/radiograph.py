'''
Created on 17-mei-2014

@author: samue_000
'''

import numpy as np
import cv2

cropX = (1100,1900)
cropY = (550,1450)

def preprocess(img):
    cropped = img[cropY[0]:cropY[1],cropX[0]:cropX[1]]
    denoised = cv2.fastNlMeansDenoising(cropped)
    clahe = cv2.createCLAHE(clipLimit=20, tileGridSize=(8,8));
    return clahe.apply(denoised)

def sharpen(img):
    kernel = np.array([[0,0,0], [0,1.25,0], [0,0,0]]) - (1/9) * np.array([[1,1,1], [1,1,1], [1,1,1]])
    return cv2.filter2D(img, -1, kernel)

'''
MAIN PROGRAM
'''
if __name__ == '__main__':
    #cv2.imshow('original', img)
    #cv2.imshow('preprocessed', preprocess(img))
    #cv2.waitKey(0)
    
    # THIS ROUTINE LET YOU PREPROCESS IMAGES
    number = '01'
    img = cv2.imread('../data/Radiographs/' + number + '.tif',0)
    img = preprocess(img)
    cv2.imwrite('../data/Radiographs/' + number + 'p2.tif', img)
    cv2.imshow('preprocessed', img)
    print 'Image ' + number + ' is processed!'
    cv2.waitKey(0)


