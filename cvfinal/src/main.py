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
        
    return ret #TODO: methode gebruikt?

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
Takes toothmatrix, and returns a similar one with, for every person, the landmarks in coordinates scaled so
that the standard deviation of those landmarks becomes 1 (in both directions).
'''
def procrustesScaleMatrix(matrix):
    nbP = matrix.shape[1]
    ret = np.zeros(matrix.shape)
    for p in range(nbP):
        ret[:,p,0] = procrustesScaleVector(matrix[:,p,0])
        ret[:,p,1] = procrustesScaleVector(matrix[:,p,1])
        
    return ret

'''
Takes numpy vector with the values for one persons, different x- or y values of landmarks, and returns a similar vector so that the standard deviation
of the values becomes 1.
'''    
def procrustesScaleVector(vector):
    nbL = vector.shape[0]
    std = np.std(vector)
    
    ret = np.zeros(vector.shape)
    for l in range(nbL):
        ret[l] = vector[l]/std
    return ret

def distance(matrix1,matrix2):
    return 0.5 #TODO

def alignWith(matrix,mean):
    return matrix #TODO

def newMean(itmat, firstMean):
    return firstMean #TODO: berekenen, alignen met firstMean en schalen

def procrustesMatrix(matrix):
    # translate to gravity
    itmat = procrustesTranslateMatrix(matrix)
    # choose one example, scale it
    firstMean = np.zeros(matrix[:,0,:].shape)
    firstMean[:,0,0] = procrustesScaleVector(itmat[:,0,0])
    firstMean[:,0,1] = procrustesScaleVector(itmat[:,0,1])
    # loop
    converged = False
    newmean = firstMean
    while not converged:
        oldmean = newmean
        # align all schapes
        itmat = alignWith(itmat, oldmean)
        # calculate newmean
        newmean = newMean(itmat, firstMean)  
        # update converged
        converged = distance(oldmean, newmean) < 0.5
    return itmat

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
    print procrustesMatrix(readLandmarks(1, 14, 80))
    
