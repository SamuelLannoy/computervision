import numpy as np
import plot_teeth as pt

'''
Takes toothmatrix (LM x Pers x Dim), and returns a similar one with, for every person, the landmarks in coordinates with an
origin in the mean of those landmarks. Also returns a matrix with translation components for every person in the rows (Pers x Dim).
'''
def procrustesTranslateMatrix(matrix):
    nbP = matrix.shape[1]
    ret = np.zeros(matrix.shape)
    trans = np.zeros((matrix.shape[1],2))
    for p in range(nbP):
        (ret[:,p,0], trans[p,0]) = procrustesTranslateVector(matrix[:,p,0])
        (ret[:,p,1], trans[p,1]) = procrustesTranslateVector(matrix[:,p,1])
        
    return ret, trans

'''
Takes numpy vector with the values for one person, different x- or y values of landmarks, and returns a similar vector with the delta of each
entry to the average over the landmarks. Also returns the translated distance (= minus this average).
'''    
def procrustesTranslateVector(vector):
    nbL = vector.shape[0]
    avg = np.mean(vector)
    
    ret = np.zeros(vector.shape)
    for l in range(nbL):
        ret[l] = vector[l] - avg
    return ret, -avg

'''
Takes toothmatrix (LM x Pers x Dim), and returns a similar one with, for every person, the landmarks in coordinates scaled so
that the standard deviation of those landmarks becomes 1 (in both directions). Also returns a matrix with the scaling components
for every person in the rows (Pers x Dim)
'''
def procrustesScaleMatrix(matrix):
    nbP = matrix.shape[1]
    ret = np.zeros(matrix.shape)
    scales = np.zeros((matrix.shape[1],2))
    for p in range(nbP):
        (ret[:,p,0], scales[p,0]) = procrustesScaleVector(matrix[:,p,0])
        (ret[:,p,1], scales[p,1]) = procrustesScaleVector(matrix[:,p,1])
        
    return ret, scales

'''
Takes numpy vector with the values for one persons, different x- or y values of landmarks, and returns a similar vector so that the standard deviation
of the values becomes 1. Also returns scale.
'''    
def procrustesScaleVector(vector):
    nbL = vector.shape[0]
    std = np.std(vector)
    
    ret = np.zeros(vector.shape)
    for l in range(nbL):
        ret[l] = vector[l]/std
    
    return ret, 1.0/std

def getEntrywiseProduct(array1,array2):
    result = np.zeros(array1.shape)
    
    for i in range(array1.size):
        result[i] = array1[i]*array2[i]
    
    return result

'''
Returns the angle over which the given tooth toAlign (LM x Dim) has to be rotated so that its landmarks have a minimal
Sum of Squared Distances (SSD) to the given tooth target (LM x Dim).
'''
def getSmallestSSDAngle(target, toAlign):
    wy = getEntrywiseProduct(toAlign[0], target[1])
    zx = getEntrywiseProduct(toAlign[1], target[0])
    wy_zx = wy - zx
    num = wy_zx.sum()
    
    wx = getEntrywiseProduct(toAlign[0], target[0])
    zy = getEntrywiseProduct(toAlign[1], target[1])
    wx_zy = wx + zy
    den = wx_zy.sum()
    
    return np.arctan2(num,den)

'''
Rotates the teeth in the given matrix (LM x Pers x Dim) so that they have a smallest SSD to the given tooth (LM x Dim).
Also returns the rotations for every person in the rows (Pers x Angle).
'''
def procrustesRotateMatrix(matrix,target):
    rotated = np.zeros(matrix.shape)
    thetas = np.zeros(matrix.shape[1])
    
    for p in range(matrix.shape[1]):
        thetas[p] = getSmallestSSDAngle(target, matrix[:,p,:])
        for l in range(matrix.shape[0]):
            rotated[l,p,0] = matrix[l,p,0]*np.cos(thetas[p]) - matrix[l,p,1]*np.sin(thetas[p])
            rotated[l,p,1] = matrix[l,p,0]*np.sin(thetas[p]) + matrix[l,p,1]*np.cos(thetas[p])
    
    return rotated, thetas

def distance(matrix1,matrix2):
    return (matrix2-matrix1).max()

'''
Scales and rotates the teeth in the given matrix (LM x Pers x Dim) to match the given target (LM x Dim).
'''
def alignWith(matrix,target):
    scaled = procrustesScaleMatrix(matrix)[0]
    rotated = procrustesRotateMatrix(scaled,target)[0]
    return rotated

'''
Returns the mean tooth of the given teeth in itmat (LM x Pers), aligned to the given target (LM x Dim).
'''
def estimateMean(itmat, target):
    nbL = itmat.shape[0]
    newMatrix = np.zeros((nbL,1,2))
    
    # calculate mean (as toothmatrix with one person) (step 5)
    for l in range(nbL):
        avgX = np.average(itmat[l,:,0])
        avgY = np.average(itmat[l,:,1])
        newMatrix[l,0,0] = avgX
        newMatrix[l,0,1] = avgY
    
    # align new mean with first mean (step 6)
    newMatrix = alignWith(newMatrix, target)
    
    # return new mean as 2D matrix
    newMean = newMatrix[:,0,:]
    return newMean

'''
Translates, scales and rotates the tooth for different persons in
the given toothMatrix to a common framework (common for this tooth
and these persons). maxIts limits the number of iterations (zero
represents infinity).

Protocol 4: Aligning a Set of Shapes (Appendix A, page 21)
'''
def procrustesMatrix(matrix, maxIts):    
    # translate to gravity (step 1)
    itmat = procrustesTranslateMatrix(matrix)[0]
    
    # choose one example, scale it (step 2 and 3)
    firstMean = np.zeros(matrix[:,0,:].shape)
    firstMean[:,0] = procrustesScaleVector(itmat[:,0,0])[0]
    firstMean[:,1] = procrustesScaleVector(itmat[:,0,1])[0]
    
    # loop
    converged = False
    newmean = firstMean
    it = 1
    while not converged:
        oldmean = newmean
        # align all schapes (step 4)
        itmat = alignWith(itmat, oldmean)
        # calculate newmean (step 5 and 6)
        newmean = estimateMean(itmat, firstMean)
        
        # update converged (step 7)
        d = distance(oldmean, newmean)
        converged = d < 1.0e-10 or it == maxIts
        
        it = it + 1
        
    return itmat

