import numpy as np
import radiograph as rg

'''
Returns a matrix of the form LM x Pers x Dim, representing a given tooth for the given persons.
All Id's counting from 0
'''
def readLandmarksOfTooth(toothId, personIds, nbLandmarks):
    landmarks = np.zeros((nbLandmarks, personIds.shape[0], 2))
    
    for i in range(personIds.shape[0]):
        f = open('../data/Landmarks/original/landmarks' 
                 + str(personIds[i]+1) + '-' + str(toothId+1) + '.txt', 'r')
        for landmarkId in range(nbLandmarks):
            landmarks[landmarkId, i, 0] = float(f.readline()) - rg.cropX[0]
            landmarks[landmarkId, i, 1] = float(f.readline()) - rg.cropY[0]
    return landmarks

'''
Returns a matrix of the form Tooth x LM x Dim, representing the teeth of a given person.
'''
def readLandmarksOfPerson(personId, nbLandmarks):
    landmarks = np.zeros((8,nbLandmarks,2))
    for toothId in range(8):
        landmarks[toothId,:,:] = readLandmarksOfTooth(toothId,np.array([personId]),nbLandmarks)[:,0,:]
    return landmarks