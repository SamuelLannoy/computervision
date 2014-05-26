import cv2
import numpy as np
import procrustes
import plot_teeth as pt
import profile
import radiograph as rg
import init_points

'''
Creates a tooth matrix from the landmarks in the files.
A tooth matrix is a Landmark x Person x Dimension matrix,
with dimension = 0 -> x and dimension = 1 -> y
'''
def readLandmarks(toothId, nbPersons, nbLandmarks):
    landmarks = np.zeros((nbLandmarks, nbPersons, 2))
    for personId in range(1, nbPersons+1):
        f = open('../data/Landmarks/original/landmarks' 
                 + str(personId) + '-' + str(toothId) + '.txt', 'r')
        for landmarkId in range(nbLandmarks): 
            landmarks[landmarkId, personId-1, 0] = float(f.readline())
            landmarks[landmarkId, personId-1, 1] = float(f.readline())
    return landmarks

'''
Creates an image matrix from the radiographs in the files.
The resulting matrix is a Person x yDim x xDim matrix.
'''
def readImages(nbPersons):
    xDim = rg.cropX[1] - rg.cropX[0]
    yDim = rg.cropY[1] - rg.cropY[0]
    images = np.zeros((nbPersons, yDim, xDim))
    for personId in range(0,nbPersons):
        images[personId] = rg.preprocess(cv2.imread('../data/Radiographs/' + ("0" + str(personId+1) if personId+1 < 10 else str(personId+1)) + '.tif',0))
    return images

'''
Returns the image to fit from the given person id (15..30)
'''
def readImageToFit(personId):
    return rg.preprocess(cv2.imread('../data/Radiographs/extra/' + str(personId) + '.tif',0))

def fitToothModelToImage(landmarks, images, imageToFit):
    return

'''
landmarks is LM x Person x Dim
'''
def stackPoints(landmarks):
    return np.vstack((landmarks[:,:,0],landmarks[:,:,1])) 

'''
landmarks is LM * Dim
'''
def stackPointsForPerson(landmarks):
    return np.vstack((np.transpose([landmarks[:,0]]),np.transpose([landmarks[:,1]]))) 

'''
stackedLandmarks is 2*LM
'''
def unstackPointsForPerson(stackedLandmarks):
    columnLength = stackedLandmarks.shape[0]/2
    return np.column_stack((stackedLandmarks[0:columnLength], stackedLandmarks[columnLength:2*columnLength]))
    
if __name__ == '__main__':
    landmarks = readLandmarks(1, 14, 40)
    images = readImages(14)
    imageToFit = readImageToFit(17)
    
    processedLandmarks = procrustes.procrustesMatrix(landmarks,0) 
    pcMean, pcEigv = cv2.PCACompute(np.transpose(stackPoints(processedLandmarks)))
    
    x = xStriped = np.transpose(pcMean)
    P = np.transpose(pcEigv) # normalized
    
    '''
    covar, mean = cv2.calcCovarMatrix(stackPoints(processedLandmarks), cv2.cv.CV_COVAR_NORMAL | cv2.cv.CV_COVAR_COLS)
    eVal = np.linalg.eigvalsh(covar)
    
    print xStriped - mean
    print eVal
    '''
    
    b = np.zeros_like(xStriped) # Protocol 1: step 1
    
    Y = np.array([[ 284.,    3.],
                 [ 279.,   13.],
                 [ 276.,   24.],
                 [ 270.,   41.],
                 [ 267.,   51.],
                 [ 263.,   64.],
                 [ 261.,   74.],
                 [ 259.,   85.],
                 [ 256.,   97.],
                 [ 253.,  113.],
                 [ 251.,  121.],
                 [ 247.,  135.],
                 [ 243.,  146.],
                 [ 239.,  158.],
                 [ 239.,  172.],
                 [ 241.,  183.],
                 [ 238.,  191.],
                 [ 236.,  209.],
                 [ 234.,  222.],
                 [ 248.,  235.],
                 [ 255.,  247.],
                 [ 271.,  252.],
                 [ 284.,  253.],
                 [ 293.,  245.],
                 [ 296.,  233.],
                 [ 298.,  219.],
                 [ 298.,  202.],
                 [ 299.,  190.],
                 [ 305.,  171.],
                 [ 313.,  153.],
                 [ 313.,  141.],
                 [ 319.,  131.],
                 [ 321.,  117.],
                 [ 323.,  105.],
                 [ 327.,   89.],
                 [ 328.,   72.],
                 [ 330.,   64.],
                 [ 332.,   57.],
                 [ 332.,   50.],
                 [ 333.,   45.]]) # Protocol 1: current found points
                 
    #Y = init_points.getModelPoints(imageToFit)
    
    
    while(1):
        # Protocol 1: step 3 & 4
        y, _ = procrustes.procrustesTranslateMatrixForPerson(Y)
        y, _ = procrustes.procrustesScaleMatrixForPerson(y)
        y, theta = procrustes.procrustesRotateMatrixForPerson(y, unstackPointsForPerson(x))
        
        # Protocol 1: step 5
        yStacked = stackPointsForPerson(y) 
        #yStacked = yStacked/np.dot(np.transpose(yStacked), xStriped) 
        
        # Protocol 1: step 6
        b = np.dot(np.transpose(P), (yStacked - xStriped))
        
        # Reconstruction
        x = xStriped + np.dot(P, b)
    
        pt.plotTooth(unstackPointsForPerson(x))
        pt.show()

    
