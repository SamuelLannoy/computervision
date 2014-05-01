import cv2
import numpy as np
import procrustes

'''
Creates a  tooth matrix from the landmarks in the files.
A tooth matrix is a Landmark x Person x Dimension matrix,
with dimension = 0 -> x and dimension = 1 -> y
'''
def readLandmarks(toothId, nbPersons, nbLandmarks):
    landmarks = np.zeros((nbLandmarks, nbPersons, 2))
    for personId in range(1, nbPersons+1):
        f = open('../data/Landmarks/original/landmarks' 
                 + str(personId) + '-' + str(toothId) + '.txt', 'r')
        for landmarkId in range(nbLandmarks/2): 
            landmarks[landmarkId, personId-1, 0] = float(f.readline())
            landmarks[landmarkId, personId-1, 1] = float(f.readline())
    return landmarks
    
if __name__ == '__main__':
    lm = readLandmarks(1, 14, 80)
    print procrustes.procrustesMatrix(lm)
    
