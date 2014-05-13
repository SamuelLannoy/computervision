import numpy as np
import plot_teeth as pt

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

def getEntrywiseProduct(array1,array2):
    result = np.zeros(array1.shape)
    
    for i in range(array1.size):
        result[i] = array1[i]*array2[i]
    
    return result

def getSmallestSSDAngle(mean, toAlign):
    wy = getEntrywiseProduct(toAlign[0], mean[1])
    zx = getEntrywiseProduct(toAlign[1], mean[0])
    wy_zx = wy - zx
    num = wy_zx.sum()
    
    wx = getEntrywiseProduct(toAlign[0], mean[0])
    zy = getEntrywiseProduct(toAlign[1], mean[1])
    wx_zy = wx + zy
    den = wx_zy.sum()
    
    return np.arctan2(num,den)

'''
Rotates the teeth in the given matrix so that they have a smallest SSD to the mean.
'''
def procrustesRotateMatrix(matrix,mean):
    rotated = np.zeros(matrix.shape)
    
    for person in range(matrix.shape[1]):
        theta = getSmallestSSDAngle(mean, matrix[:,person,:])
        for landmark in range(matrix.shape[0]):
            rotated[landmark,person,0] = matrix[landmark,person,0]*np.cos(theta) - matrix[landmark,person,1]*np.sin(theta)
            rotated[landmark,person,1] = matrix[landmark,person,0]*np.sin(theta) + matrix[landmark,person,1]*np.cos(theta)
    
    return rotated

def distance(matrix1,matrix2):
    return (matrix2-matrix1).max()

def alignWith(matrix,mean):
    scaled = procrustesScaleMatrix(matrix)
    rotated = procrustesRotateMatrix(scaled,mean)
    return rotated

def estimateMean(itmat, firstMean):
    nbL = itmat.shape[0]
    newMatrix = np.zeros((nbL,1,2))
    
    # calculate mean (as toothmatrix with one person)
    for l in range(nbL):
        avgX = np.average(itmat[l,:,0])
        avgY = np.average(itmat[l,:,1])
        newMatrix[l,0,0] = avgX
        newMatrix[l,0,1] = avgY
    
    # align new mean with first mean
    newMatrix = alignWith(newMatrix, firstMean)
    
    # return new mean as 2D matrix
    newMean = newMatrix[:,0,:]
    return newMean

def procrustesMatrix(matrix):    
    # translate to gravity
    itmat = procrustesTranslateMatrix(matrix)
    
    pt.plotTeeth(itmat)
    pt.show()
    
    # choose one example, scale it
    firstMean = np.zeros(matrix[:,0,:].shape)
    firstMean[:,0] = procrustesScaleVector(itmat[:,0,0])
    firstMean[:,1] = procrustesScaleVector(itmat[:,0,1])
    
    # loop
    converged = False
    newmean = firstMean
    #it = 0
    while not converged:
        oldmean = newmean
        # align all schapes
        itmat = alignWith(itmat, oldmean)
        # calculate newmean
        newmean = estimateMean(itmat, firstMean)  
        # update converged
        d = distance(oldmean, newmean)
        converged = d < 1.0e-10 # or it > 1000
        
        #it = it + 1
        
    pt.plotTeeth(itmat)
    pt.show()
        
    return itmat

