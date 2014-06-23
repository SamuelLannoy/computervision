import numpy as np
import radiograph as rg
import main

'''
Returns a matrix of the form LM x Pers x Dim, representing a given tooth for the given persons.

Translates the landmarks to match the cropped images (preprocessing)

All Id's counting from 0
'''
def readLandmarksOfTooth(toothId, personIds, corrPreProc = True):
    landmarks = np.zeros((main.nbLandmarks, personIds.shape[0], 2))
    
    for i in range(personIds.shape[0]):
        f = open('../data/Landmarks/original/landmarks' 
                 + str(personIds[i]+1) + '-' + str(toothId+1) + '.txt', 'r')
        for landmarkId in range(main.nbLandmarks):
            landmarks[landmarkId, i, 0] = float(f.readline())
            landmarks[landmarkId, i, 1] = float(f.readline())
            
        if corrPreProc :
            for landmarkId in range(main.nbLandmarks):
                landmarks[landmarkId, i, 0] = landmarks[landmarkId, i, 0] - rg.cropX[0]
                landmarks[landmarkId, i, 1] = landmarks[landmarkId, i, 1] - rg.cropY[0]
    return landmarks

'''
Returns a matrix of the form LM x Tooth x Dim, representing all the teeth of a given person.

Translates the landmarks to match the cropped images (preprocessing)

All Id's counting from 0
'''
def readLandmarksOfPerson(personId, corrPreProc = True):
    return readLandmarksOfPersonAndTeeth(personId, main.toothIds, corrPreProc)

'''
Returns a matrix of the form LM x Tooth x Dim, representing the given teeth of a given person.

Translates the landmarks to match the cropped images (preprocessing)

All Id's counting from 0
'''
def readLandmarksOfPersonAndTeeth(personId, toothIds, corrPreProc = True):
    landmarks = np.zeros((main.nbLandmarks, toothIds.shape[0], 2))
    for i in range(toothIds.shape[0]):
        toothId = toothIds[i]
        landmarks[:,i,:] = readLandmarksOfTooth(toothId,np.array([personId]), corrPreProc)[:,0,:]
    return landmarks

def printBoundryLandmarks():
    upLeft = 10000
    upRight = 0
    upTop = 10000
    upBot = 0
    lowLeft = 10000
    lowRight = 0
    lowTop = 10000
    lowBot = 0
    
    for personId in range(14):
        upLMs = readLandmarksOfPersonAndTeeth(personId, np.array(range(4)), corrPreProc = False)
        lowLMs = readLandmarksOfPersonAndTeeth(personId, np.array(range(4,8)), corrPreProc = False)
        
        upLeft = min(upLeft, np.min(upLMs[:,0]))
        upRight = max(upRight, np.max(upLMs[:,0]))
        upTop = min(upTop, np.min(upLMs[:,1]))
        upBot = max(upBot, np.max(upLMs[:,1]))
        
        lowLeft = min(lowLeft, np.min(lowLMs[:,0]))
        lowRight = max(lowRight, np.max(lowLMs[:,0]))
        lowTop = min(lowTop, np.min(lowLMs[:,1]))
        lowBot = max(lowBot, np.max(lowLMs[:,1]))
    
    print 'up left: ' + str(upLeft)
    print 'up right: ' + str(upRight)
    print 'up top: ' + str(upTop)
    print 'up bottom: ' + str(upBot)
    
    print 'low left: ' + str(lowLeft)
    print 'low right: ' + str(lowRight)
    print 'low top: ' + str(lowTop)
    print 'low bottom: ' + str(lowBot)
        
        
        
'''
MAIN PROGRAM
'''
if __name__ == '__main__':
    printBoundryLandmarks()

            