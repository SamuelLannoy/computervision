'''
Created on 17-mei-2014

@author: samue_000
'''

import numpy as np
import cv2

def preprocess(img):
    cropped = img[400:1600,1000:2100]
    denoised = cv2.fastNlMeansDenoising(cropped)
    clahe = cv2.createCLAHE(clipLimit=5);
    return clahe.apply(denoised)

def sharpen(img):
    kernel = np.array([[0,0,0], [0,1.25,0], [0,0,0]]) - (1/9) * np.array([[1,1,1], [1,1,1], [1,1,1]])
    return cv2.filter2D(img, -1, kernel)

'''
Everything starting here is just a test that can be run
'''
img = cv2.imread('../data/Radiographs/07.tif',0)
#cv2.imshow('original', img)
cv2.imshow('preprocessed', preprocess(img))
cv2.waitKey(0)