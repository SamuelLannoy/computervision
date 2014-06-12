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
        #images[personId] = cv2.imread('../data/Radiographs/' + ("0" + str(personId+1) if personId+1 < 10 else str(personId+1)) + 'p.tif',0)
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
Aligns the given shapes with each other.
Returning scale [0] and rotation [1].

x1 and x2 are LM * Dim
'''
def alignShapes(x1, x2):
    x1Stacked = stackPointsForPerson(x1)
    x2Stacked = stackPointsForPerson(x2)
    
    x1_norm_sq = np.linalg.norm(x1Stacked, 2)**2
    
    a = np.dot(np.transpose(x1Stacked), x2Stacked)[0][0]/x1_norm_sq
    
    b = 0
    for i in range(x1.shape[0]):
        b += x1[i,0]*x2[i,1] - x1[i,1]*x2[i,0]
    b /= x1_norm_sq
    
    return np.sqrt(a**2 + b**2), np.arctan2(b, a)
    
'''
MAIN PROGRAM
'''
if __name__ == '__main__':
    # Choice of profile length (2n+1)
    nModel = 20
    nSample = 30
    
    # Read data (images and landmarks)
    images, landmarks = readData(1, 14, 40)
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
    prev_b = b

    stop = False
    it = 1
    while(not stop): 
        
        # Protocol 2: step 1 (examine region around each point to find best nearby match)
        Y = profile.getNewModelPoints(imageToFit, X, model, nSample)
        
        # Protocol 2: step 3 = protocol 1
        
        ## Protocol 1: step 2 (generate model point positions)
        xStacked = xStriped + np.dot(P, b)

        ## Protocol 1: step 3 & 4 (project Y into the model coordinate frame)
        y, translation = procrustes.procrustesTranslateMatrixForPerson(Y)
        scale, rotation = alignShapes(unstackPointsForPerson(xStacked), y)
        scale = 1/scale
        rotation = -rotation
        
        y = procrustes.scaleMatrixForPerson(y, scale)
        y = procrustes.rotateMatrixForPerson(y, rotation)
        
        ## Protocol 1: step 5 --> NOT NEEDED??
        #yStacked = stackPointsForPerson(y) 
        #yStacked = yStacked/np.dot(np.transpose(yStacked), xStriped)
        #y = unstackPointsForPerson(yStacked)
        
        ## Protocol 1: step 6 (update model parameters)
        b = np.dot(np.transpose(P), (stackPointsForPerson(y) - xStriped))
        
        # Protocol 2: step 3 (apply constraints to b)
        for i in range(b.shape[0]):
            if np.abs(b[i,0]) > 3*np.sqrt(eigval[i]):
                b[i,0] = np.sign(b[i,0]) * 3*np.sqrt(eigval[i])
        
        # Calculate difference with previous result
        # and stop iterating after a certain treshold
        differences = prev_b - b
        prev_b = b
        
        stop = True
        for diff in differences:
            if np.abs(diff) > 0.01:
                stop = False
                
        it += 1
        
    print "Number of iterations: " + str(it)
    
    # Draw the model on the radiograph
    x = unstackPointsForPerson(xStriped + np.dot(P, b))
    X = procrustes.rotateMatrixForPerson(x, -rotation)
    X = procrustes.scaleMatrixForPerson(X, 1/scale)
    X = procrustes.translateMatrixForPerson(X, -translation)
    
    drawImage = imageToFit.copy()
    cv2.polylines(drawImage, np.int32([X]), True, 255)
    cv2.imshow('draw', drawImage)
    cv2.waitKey(0)
            
    # Plot projection on the model
    #pt.plotTooth(x)
    #pt.plotTooth(y)
    #pt.plotTooth(unstackPointsForPerson(xStriped))
    #pt.show()
