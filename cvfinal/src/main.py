import cv2
import numpy as np
import procrustes
import plot_teeth as pt
import profile
import radiograph as rg
import init_points

'''
Creates a tooth matrix from the landmarks in the files.
A tooth matrix is a Landmark xStacked Person xStacked Dimension matrix,
with dimension = 0 -> xStacked and dimension = 1 -> y
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
The resulting matrix is a Person xStacked yDim xStacked xDim matrix.
'''
def readImages(nbPersons):
    xDim = rg.cropX[1] - rg.cropX[0]
    yDim = rg.cropY[1] - rg.cropY[0]
    images = np.zeros((nbPersons, yDim, xDim))
    for personId in range(0,nbPersons):
        images[personId] = rg.preprocess(cv2.imread('../data/Radiographs/' + ("0" + str(personId+1) if personId+1 < 10 else str(personId+1)) + '.tif',0))
    return images

def readData(toothId, nbPersons, nbLandmarks):
    xDim = rg.cropX[1] - rg.cropX[0]
    yDim = rg.cropY[1] - rg.cropY[0]
    
    images = np.zeros((nbPersons, yDim, xDim))
    landmarks = np.zeros((nbLandmarks, nbPersons, 2))
    for personId in range(0,nbPersons):
        #images[personId] = rg.preprocess(cv2.imread('../data/Radiographs/' + ("0" + str(personId+1) if personId+1 < 10 else str(personId+1)) + '.tif',0))
        images[personId] = rg.preprocess(cv2.imread('../data/Segmentations/' + ("0" + str(personId+1) if personId+1 < 10 else str(personId+1)) + '-0.png',0))
        f = open('../data/Landmarks/original/landmarks' 
                 + str(personId+1) + '-' + str(toothId) + '.txt', 'r')
        for landmarkId in range(nbLandmarks): 
            landmarks[landmarkId, personId, 0] = float(f.readline()) - rg.cropX[0]
            landmarks[landmarkId, personId, 1] = float(f.readline()) - rg.cropY[0]
            
    return images, landmarks
        

'''
Returns the image to fit from the given person id (15..30)
'''
def readImageToFit(personId):
    #return rg.preprocess(cv2.imread('../data/Radiographs/0' + str(personId) + '.tif',0))
    return rg.preprocess(cv2.imread('../data/Segmentations/0' + str(personId) + '-0.png',0))

def fitToothModelToImage(landmarks, images, imageToFit):
    return

'''
landmarks is LM xStacked Person xStacked Dim
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
    nModel = 30
    nSample = 50
    
    images, landmarks = readData(1, 5, 40)
    imageToFit = readImageToFit(1)
    
    processedLandmarks = procrustes.procrustesMatrix(landmarks,0)
    pcMean, pcEigv = cv2.PCACompute(np.transpose(stackPoints(processedLandmarks)))
    
    xStacked = xStriped = np.transpose(pcMean)
    P = np.transpose(pcEigv) # normalized
    
    '''
    covar, mean = cv2.calcCovarMatrix(stackPoints(processedLandmarks), cv2.cv.CV_COVAR_NORMAL | cv2.cv.CV_COVAR_COLS)
    eVal = np.linalg.eigvalsh(covar)
    
    print xStriped - mean
    print eVal
    '''
    
    # Protocol 1: step 1
    b = np.zeros_like(xStriped) 

    # Protocol 1: current found points
    #Y = init_points.getModelPoints(imageToFit)
    #Y = landmarks[:,0,:]
    Y = np.array([[ 275.,  197.],
                 [ 267.,  205.],
                 [ 259.,  216.],
                 [ 254.,  228.],
                 [ 248.,  240.],
                 [ 244.,  255.],
                 [ 238.,  266.],
                 [ 235.,  278.],
                 [ 229.,  289.],
                 [ 225.,  303.],
                 [ 222.,  317.],
                 [ 219.,  329.],
                 [ 214.,  345.],
                 [ 211.,  358.],
                 [ 208.,  369.],
                 [ 204.,  387.],
                 [ 204.,  401.],
                 [ 206.,  416.],
                 [ 210.,  430.],
                 [ 219.,  438.],
                 [ 233.,  442.],
                 [ 250.,  446.],
                 [ 262.,  447.],
                 [ 275.,  447.],
                 [ 284.,  441.],
                 [ 289.,  430.],
                 [ 295.,  414.],
                 [ 298.,  396.],
                 [ 298.,  379.],
                 [ 297.,  354.],
                 [ 300.,  335.],
                 [ 304.,  315.],
                 [ 307.,  294.],
                 [ 313.,  273.],
                 [ 313.,  262.],
                 [ 312.,  246.],
                 [ 308.,  229.],
                 [ 302.,  216.],
                 [ 299.,  205.],
                 [ 292.,  199.]])
    
    directions = profile.getDirections(landmarks)
    model = profile.getModel(images, landmarks, directions, nModel)
    
    while(1):
        # Protocol 1: step 3 & 4
        y, _ = procrustes.procrustesTranslateMatrixForPerson(Y)
        y, _ = procrustes.procrustesScaleMatrixForPerson(y)
        y, _ = procrustes.procrustesRotateMatrixForPerson(y, unstackPointsForPerson(xStacked))
        
        drawImage = imageToFit.copy()
        cv2.polylines(drawImage, np.int32([Y]), True, 255) # cv2.cv.Scalar(0,0,255)
        cv2.imshow('draw', drawImage)
        cv2.waitKey(0)
        
        # Protocol 1: step 5 --> NOT NEEDED??
        # yStacked = stackPointsForPerson(y) 
        # yStacked = yStacked/np.dot(np.transpose(yStacked), xStriped)
        
        # Protocol 1: step 6
        b = np.dot(np.transpose(P), (stackPointsForPerson(y) - xStriped))
        
        # Reconstruction / Protocol 1: step 2
        xStacked = xStriped + np.dot(P, b)
        
        # Plot projection on the model
        #pt.plotTooth(unstackPointsForPerson(xStacked))
        #pt.plotTooth(y)
        #pt.show()
        
        # Create new model
        Y = profile.getNewModelPoints(imageToFit, Y, model, nSample)

    
