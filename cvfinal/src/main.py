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
        #images[personId] = rg.preprocess(cv2.imread('../data/Radiographs/' + ("0" + str(personId+1) if personId+1 < 10 else str(personId+1)) + '.tif',0))
        images[personId] = cv2.imread('../data/Radiographs/' + ("0" + str(personId+1) if personId+1 < 10 else str(personId+1)) + 'p.tif',0)
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
    nModel = 15
    nSample = 40
    
    # Read data (images and landmarks)
    images, landmarks = readData(2, 14, 40)
    imageToFit = readImageToFit(2)
    
    # Number of modes
    nbModes = 7 #landmarks.shape[1]
    
    # Initialization of mean vector (xStriped), covariance matrix, x-vector, x-striped-vector (mean), eigenvectors (P) and eigenvalues.
    processedLandmarks = procrustes.procrustesMatrix(landmarks,0)
    pcMean, pcEigv = cv2.PCACompute(np.transpose(stackPoints(processedLandmarks)))
    xStacked = xStriped = np.transpose(pcMean)
    P = np.transpose(pcEigv[:nbModes]) # normalized
    covar, _ = cv2.calcCovarMatrix(stackPoints(processedLandmarks), cv2.cv.CV_COVAR_SCRAMBLED | cv2.cv.CV_COVAR_SCALE | cv2.cv.CV_COVAR_COLS)    
    eigval = np.sort(np.linalg.eigvals(covar), kind='mergesort')[::-1][:nbModes] # pick t largest eigenvalues
        
    # Initialization of the initial points
    X = init_points.getModelPoints(imageToFit)
    '''X = np.array([[ 318., 149.],
                 [ 313., 171.],
                 [ 310., 186.],
                 [ 305., 218.],
                 [ 306., 223.],
                 [ 309., 231.],
                 [ 311., 256.],
                 [ 305., 274.],
                 [ 298., 293.],
                 [ 296., 309.],
                 [ 293., 332.],
                 [ 293., 345.],
                 [ 295., 363.],
                 [ 296., 382.],
                 [ 295., 397.],
                 [ 295., 416.],
                 [ 310., 432.],
                 [ 336., 438.],
                 [ 350., 440.],
                 [ 375., 444.],
                 [ 386., 435.],
                 [ 391., 418.],
                 [ 397., 398.],
                 [ 401., 377.],
                 [ 400., 353.],
                 [ 399., 348.],
                 [ 396., 333.],
                 [ 396., 312.],
                 [ 396., 298.],
                 [ 395., 277.],
                 [ 392., 259.],
                 [ 390., 241.],
                 [ 387., 228.],
                 [ 381., 207.],
                 [ 379., 194.],
                 [ 379., 179.],
                 [ 374., 161.],
                 [ 373., 157.],
                 [ 361., 151.],
                 [ 334., 148.]])'''
    '''X = np.array([[ 311., 135.],
                 [ 311., 151.],
                 [ 313., 168.],
                 [ 314., 183.],
                 [ 315., 204.],
                 [ 313., 222.],
                 [ 310., 242.],
                 [ 304., 268.],
                 [ 304., 287.],
                 [ 299., 307.],
                 [ 295., 328.],
                 [ 293., 351.],
                 [ 293., 377.],
                 [ 293., 399.],
                 [ 299., 417.],
                 [ 310., 431.],
                 [ 319., 440.],
                 [ 335., 448.],
                 [ 357., 448.],
                 [ 377., 447.],
                 [ 389., 437.],
                 [ 397., 419.],
                 [ 402., 397.],
                 [ 404., 376.],
                 [ 403., 359.],
                 [ 401., 340.],
                 [ 396., 315.],
                 [ 393., 296.],
                 [ 386., 270.],
                 [ 379., 250.],
                 [ 376., 236.],
                 [ 375., 212.],
                 [ 375., 195.],
                 [ 377., 180.],
                 [ 377., 166.],
                 [ 376., 150.],
                 [ 368., 137.],
                 [ 354., 128.],
                 [ 339., 124.],
                 [ 323., 126.]])'''
    '''X =np.array([[ 311., 118.],
                 [ 311., 133.],
                 [ 311., 145.],
                 [ 311., 168.],
                 [ 312., 183.],
                 [ 311., 198.],
                 [ 315., 217.],
                 [ 313., 230.],
                 [ 313., 251.],
                 [ 308., 269.],
                 [ 308., 280.],
                 [ 312., 289.],
                 [ 306., 302.],
                 [ 302., 318.],
                 [ 299., 341.],
                 [ 298., 357.],
                 [ 300., 377.],
                 [ 300., 399.],
                 [ 307., 414.],
                 [ 316., 419.],
                 [ 345., 420.],
                 [ 352., 411.],
                 [ 360., 395.],
                 [ 363., 375.],
                 [ 368., 345.],
                 [ 370., 322.],
                 [ 371., 298.],
                 [ 372., 280.],
                 [ 374., 258.],
                 [ 378., 239.],
                 [ 380., 224.],
                 [ 380., 201.],
                 [ 382., 189.],
                 [ 382., 169.],
                 [ 380., 155.],
                 [ 380., 140.],
                 [ 379., 131.],
                 [ 369., 120.],
                 [ 350., 117.],
                 [ 335., 113.]])'''
    ''''X = np.array([[ 440., 219.],
         [ 435., 237.],
         [ 434., 254.],
         [ 435., 262.],
         [ 436., 275.],
         [ 439., 295.],
         [ 441., 306.],
         [ 440., 325.],
         [ 440., 344.],
         [ 438., 360.],
         [ 437., 375.],
         [ 437., 394.],
         [ 436., 407.],
         [ 436., 414.],
         [ 436., 432.],
         [ 438., 437.],
         [ 444., 444.],
         [ 456., 449.],
         [ 468., 449.],
         [ 483., 449.],
         [ 493., 449.],
         [ 505., 443.],
         [ 513., 432.],
         [ 517., 422.],
         [ 520., 414.],
         [ 520., 408.],
         [ 523., 393.],
         [ 525., 373.],
         [ 527., 351.],
         [ 528., 334.],
         [ 529., 318.],
         [ 529., 297.],
         [ 527., 278.],
         [ 516., 259.],
         [ 504., 243.],
         [ 499., 224.],
         [ 485., 217.],
         [ 472., 214.],
         [ 461., 216.],
         [ 452., 217.]])'''
    
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
        translation = -translation

        y = procrustes.rotateMatrixForPerson(y, -rotation)
        y = procrustes.scaleMatrixForPerson(y, 1/scale)
        
        ## Protocol 1: step 5 --> NOT NEEDED??
        #yStacked = stackPointsForPerson(y) 
        #yStacked = yStacked/np.dot(np.transpose(yStacked), xStriped)
        #y = unstackPointsForPerson(yStacked)
        
        ## Protocol 1: step 6 (update model parameters)
        b = np.dot(np.transpose(P), (stackPointsForPerson(y) - xStriped))
        
        # Protocol 2: step 3 (apply constraints to b)
        #mahalonobis = np.sqrt(np.sum(b**2)/np.sum(eigval))
        #if mahalonobis > 3.0:
        #   b = b*(3.0/mahalonobis)
        
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
    X = procrustes.scaleMatrixForPerson(x, scale)
    X = procrustes.rotateMatrixForPerson(X, rotation)
    X = procrustes.translateMatrixForPerson(X, translation)
    
    drawImage = imageToFit.copy()
    cv2.polylines(drawImage, np.int32([X]), True, 255)
    cv2.imshow('draw', drawImage)
    cv2.waitKey(0)
            
    # Plot projection on the model
    pt.plotTooth(x)
    pt.plotTooth(y)
    pt.plotTooth(unstackPointsForPerson(xStriped))
    pt.show()

