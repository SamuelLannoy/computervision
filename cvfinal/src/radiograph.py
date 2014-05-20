'''
Created on 17-mei-2014

@author: samue_000
'''

import numpy as np
import cv2

def preprocess(img):
    clahe = cv2.createCLAHE();
    return clahe.apply(img)

def sharpen(img):
    kernel = np.array([[0,0,0], [0,2,0], [0,0,0]]) - (1/9) * np.array([[1,1,1], [1,1,1], [1,1,1]])
    return cv2.filter2D(img, -1, kernel)

'''
Everything starting here is just a test that can be run
'''
img = cv2.imread('../data/Radiographs/01.tif',0)
cv2.imshow('original', img)
cv2.imshow('preprocessed', preprocess(img))
#cv2.imshow('sharpened', sharpen(preprocess(img)))
cv2.waitKey(0)