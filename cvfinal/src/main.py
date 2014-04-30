import cv2
import numpy as np

'''
Takes numpy matrix with landmarks in the rows (xyxyxy...) and persons in the columns, and returns a similar matrix with the delta of
each entry to the average over the persons.
'''
def procrustesTranslateMatrix(matrix):
    L = matrix.shape[0]
    ret = np.zeros(matrix.shape)
    for i in range(L):
        ret[i,:] = procrustesTranslateVector(matrix[i,:])
    return ret

'''
Takes numpy vector with the values for different persons and one landmark, and returns a similar vector with the delta of
each entry to the average over the persons.
'''    
def procrustesTranslateVector(vector):
    som = 0
    P = vector.shape[0]
    for j in range(P):
        som = sum + vector[j]
    avg = som/P
    ret = np.zeros(vector.shape)
    for j in range(P):
        ret[j] = vector[j] - avg
    return ret


def readLandmarks(toothId, nbPersons, nbLandmarks):
    landmarks = np.zeros((nbLandmarks,nbPersons))
    for personId in range(1, nbPersons+1):
        f = open('../data/Landmarks/original/landmarks' + str(personId) + '-' + str(toothId) + '.txt', 'r')
        for landmarkId in range(nbLandmarks): 
            landmarks[landmarkId, personId-1] = float(f.readline())
    return landmarks
    
if __name__ == '__main__':
    print readLandmarks(1, 14, 80)
