import numpy as np
import cv2

cropX = (1100,1900)
cropY = (550,1450)

def preprocess(img):
    cropped = img[cropY[0]:cropY[1],cropX[0]:cropX[1]]
    denoised = cv2.fastNlMeansDenoising(cropped) #cv2.bilateralFilter(cropped, 9, 150, 150)  cv2.fastNlMeansDenoising(cropped) #h=5 
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8,8)); #16x16
    return clahe.apply(denoised)

def sharpen(img):
    kernel = np.array([[0,0,0], [0,2,0], [0,0,0]]) - (1.0/9.0) * np.array([[1,1,1], [1,1,1], [1,1,1]])
    return cv2.filter2D(img, -1, kernel)

'''
MAIN PROGRAM
'''
if __name__ == '__main__':    
    # THIS ROUTINE LETS YOU PREPROCESS IMAGES
    for personId in range(1,15):
    #personId = 1
        img = cv2.imread('../data/Radiographs/' + ("0" + str(personId) if personId < 10 else str(personId)) + '.tif',0)
        img = preprocess(img)
        cv2.imwrite('../data/Radiographs/' + ("0" + str(personId) if personId < 10 else str(personId)) + 'p.tif', img)
        print 'Image ' + str(personId) + ' is processed!'
        #cv2.imshow('preprocessed', img)
        #cv2.waitKey(0)


