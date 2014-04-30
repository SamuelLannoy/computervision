import numpy as np

'''
Created on 28-apr.-2014

@author: Samuel Lannoy, Vital D'haveloose
'''

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


if __name__ == '__main__':
    pass

