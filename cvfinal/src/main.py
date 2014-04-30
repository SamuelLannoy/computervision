import numpy as np
import cv2

def readLandmarks(toothId, nbPersons, nbLandmarks):
    landmarks = np.zeros((nbLandmarks,nbPersons))
    for personId in range(1, nbPersons+1):
        f = open('../data/Landmarks/original/landmarks' + str(personId) + '-' + str(toothId) + '.txt', 'r')
        for landmarkId in range(nbLandmarks): 
            landmarks[landmarkId, personId-1] = float(f.readline())
    return landmarks
    
if __name__ == '__main__':
    print readLandmarks(1, 14, 80)