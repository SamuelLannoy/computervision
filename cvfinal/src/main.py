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

Creates an image matrix from the radiographs in the files.
The resulting matrix is a Person xStacked yDim xStacked xDim matrix.
'''
def readData(toothId, nbPersons, nbLandmarks):
    xDim = rg.cropX[1] - rg.cropX[0]
    yDim = rg.cropY[1] - rg.cropY[0]
    
    images = np.zeros((nbPersons, yDim, xDim))
    landmarks = np.zeros((nbLandmarks, nbPersons, 2))
    for personId in range(0,nbPersons):
        images[personId] = rg.preprocess(cv2.imread('../data/Radiographs/' + ("0" + str(personId+1) if personId+1 < 10 else str(personId+1)) + '.tif',0))
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
    return rg.preprocess(cv2.imread('../data/Radiographs/0' + str(personId) + '.tif',0))

'''
Does nothing at the moment
'''
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
    
'''
MAIN PROGRAM
'''
if __name__ == '__main__':
    # Choice of profile length (2n+1)
    nModel = 25
    nSample = 40
    
    # Read data (images and landmarks)
    images, landmarks = readData(1, 5, 40)
    imageToFit = readImageToFit(1)
    
    # Number of modes
    nbModes = landmarks.shape[1]
    
    # Initialization of mean vector (xStriped), covariance matrix, x-vector, x-striped-vector (mean), eigenvectors (P) and eigenvalues.
    processedLandmarks = procrustes.procrustesMatrix(landmarks,0)
    pcMean, pcEigv = cv2.PCACompute(np.transpose(stackPoints(processedLandmarks)))
    xStacked = xStriped = np.transpose(pcMean)
    P = np.transpose(pcEigv[:nbModes]) # normalized
    covar, _ = cv2.calcCovarMatrix(stackPoints(processedLandmarks), cv2.cv.CV_COVAR_SCRAMBLED | cv2.cv.CV_COVAR_SCALE | cv2.cv.CV_COVAR_COLS)    
    eigval = np.sort(np.linalg.eigvals(covar), kind='mergesort')[::-1][:nbModes] # pick t larges eigenvalues
    
    # Initialization of the initial points
    X = init_points.getModelPoints(imageToFit)
    
    # Draw the initial points
    drawImage = imageToFit.copy()
    cv2.polylines(drawImage, np.int32([X]), True, 255)
    cv2.imshow('draw', drawImage)
    cv2.waitKey(0)
    
    # Initialize the model
    directions = profile.getDirections(landmarks)
    model = profile.getModel(images, landmarks, directions, nModel)
    
    ## Protocol 1: step 1 (initialize shape parameters)
    b = np.zeros((nbModes,1))

    while(1):        
        # Protocol 2: step 1 (examine region around each point to find best nearby match)
        Y = profile.getNewModelPoints(imageToFit, X, model, nSample)
        
        # Protocol 2: step 3 = protocol 1
        
        ## Protocol 1: step 2 (generate model point positions)
        xStacked = xStriped + np.dot(P, b)

        ## Protocol 1: step 3 & 4 (project Y into the model coordinate frame)
        y, translation = procrustes.procrustesTranslateMatrixForPerson(Y)
        y, scale = procrustes.procrustesScaleMatrixForPerson(y)
        y, rotation = procrustes.procrustesRotateMatrixForPerson(y, unstackPointsForPerson(xStacked))
        
        ## Protocol 1: step 5 --> NOT NEEDED??
        #yStacked = stackPointsForPerson(y) 
        #yStacked = yStacked/np.dot(np.transpose(yStacked), xStriped)
        
        ## Protocol 1: step 6 (update model parameters)
        b = np.dot(np.transpose(P), (stackPointsForPerson(y) - xStriped))
        
        # Protocol 2: step 3 (apply constraints to b)
        for i in range(b.shape[0]):
            if np.abs(b[i,0]) < 3*np.sqrt(eigval[i]):
                b[i,0] = np.sign(b[i,0]) * 3*np.sqrt(eigval[i])
        
        x = unstackPointsForPerson(xStriped + np.dot(P, b))
        X = procrustes.rotateMatrixForPerson(x, -rotation)
        X = procrustes.scaleMatrixForPerson(X, 1/scale)
        X = procrustes.translateMatrixForPerson(X, -translation)
        
        print b
        
        # Draw the model on the radiograph
        drawImage = imageToFit.copy()
        cv2.polylines(drawImage, np.int32([X]), True, 255)
        cv2.imshow('draw', drawImage)
        cv2.waitKey(0)
        
        # Plot projection on the model
        pt.plotTooth(x)
        pt.plotTooth(y)
        pt.plotTooth(unstackPointsForPerson(xStriped))
        pt.show()

    
