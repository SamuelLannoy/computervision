import cv2
import numpy as np

'''
Takes toothmatrix, and returns a similar one with, for every person, the landmarks in coordinates with an origin in the mean of those landmarks.
'''
def procrustesTranslateMatrix(matrix):
    nbP = matrix.shape[1]
    ret = np.zeros(matrix.shape)
    for p in range(nbP):
        ret[:,p,0] = procrustesTranslateVector(matrix[:,p,0])
        ret[:,p,1] = procrustesTranslateVector(matrix[:,p,1])
    return ret

'''
Takes numpy vector with the values for one persons, different x- or y values of landmarks, and returns a similar vector with the delta of each
entry to the average over the landmarks.
'''    
def procrustesTranslateVector(vector):
    nbL = vector.shape[0]
    avg = np.mean(vector)
    
    ret = np.zeros(vector.shape)
    for l in range(nbL):
        ret[l] = vector[l] - avg
    return ret

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
    print readLandmarks(1, 14, 80)
