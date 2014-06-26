import cv2
import numpy as np

import radiograph as rg
import landmarks as lm

import init_points as ip
import procrustes
import profile
import plot_teeth as pt

# Choice of technical parameters
debugFB  = True
windowscale = 1.0 #np.float(733)/np.float(rg.cropY[1]-rg.cropY[0])

# Choice of profile length (2n+1)
nModel = 15
nSample = 40

# Choice whether the initial points are generated automatically
autoInitPoints = True

# Choice of template score loyalty (for template matching in init_points) (higher means higher loyalty of the
#  returned average tooth to templates with a better score
#  0 -> unweighted average
#  high value -> take template with highest score as initial points
templ_scr_loyalty = 2

# Choice of parameters (all Id's count from 0)
nbLandmarks = 40
trainingPersonIds = np.array(range(14))
personToFitIds = range(14,30)
toothIds = np.array(range(8))

'''
landmarks is LM x Pers x Dim

output is LM*Dim x Pers
'''
def stackPoints(landmarks):
    return np.vstack((landmarks[:,:,0],landmarks[:,:,1])) 

'''
landmarks is LM x Dim

output is LM*Dim
'''
def stackPointsForPerson(landmarks):
    return np.vstack((np.transpose([landmarks[:,0]]),np.transpose([landmarks[:,1]]))) 

'''
stackedLandmarks is LM*Dim

output is LM x Dim
'''
def unstackPointsForPerson(stackedLandmarks):
    columnLength = stackedLandmarks.shape[0]/2
    return np.column_stack((stackedLandmarks[0:columnLength], stackedLandmarks[columnLength:2*columnLength]))

'''
Copies the given image and shows a scaled version of the copy.
'''
def showScaled(image, scale, name, wait):
    showed = image.copy()
    showed = cv2.resize(showed, (0,0), fx=scale, fy=scale)
    cv2.imshow(name, showed)
    if wait : cv2.waitKey(0)
    
'''
Plot the variations of the PCA model
'''
def plotVariations(toothId=0, nbModes=15):
    # Read data (images and landmarks)
    landmarks = lm.readLandmarksOfTooth(toothId, trainingPersonIds)
    # Initialization of mean vector (xStriped), covariance matrix, x-vector, x-striped-vector (mean), eigenvectors (P) and eigenvalues.
    processedLandmarks = procrustes.procrustesMatrix(landmarks,100)
    mean, eigvec = cv2.PCACompute(np.transpose(stackPoints(processedLandmarks)))
    # Calculate the eigenvalues and sort them
    covar, _ = cv2.calcCovarMatrix(stackPoints(processedLandmarks), cv2.cv.CV_COVAR_SCRAMBLED | cv2.cv.CV_COVAR_SCALE | cv2.cv.CV_COVAR_COLS)    
    eigval = np.sort(np.linalg.eigvals(covar), kind='mergesort')[::-1]
    
    for mode in range(nbModes):
        for var in [-3,0,3]:
            tooth = mean + var*(eigval[mode]**0.5)*eigvec[mode]
            tooth = unstackPointsForPerson(np.transpose(tooth))
            # plot the tooth
            pt.plotTooth(tooth)
        # show the variations
        pt.show()
            
'''
Find the incisors in the radiographs.
'''
def findSegments():
    # load training radiographs    
    images = rg.readRadiographs(trainingPersonIds)
    if debugFB : print 'DB: Training radiographs loaded'

    for personToFitId in personToFitIds:
        # load radiograph to determine segments for
        imageToFit = rg.readRadioGraph(personToFitId)
        if debugFB : print 'DB: Radiograph to fit (#' + str(personToFitId+1) + ') loaded'
        
        contourImage = imageToFit.copy()
        segmentsImage = np.zeros_like(np.array(imageToFit))
        
        if autoInitPoints : init_points = ip.getModelPointsAutoWhole(personToFitId)

        for i in range(toothIds.shape[0]):
            toothId = toothIds[i]
            
            # Read data (images and landmarks)
            landmarks = lm.readLandmarksOfTooth(toothId, trainingPersonIds)
            if debugFB : print '   DB: Landmarks loaded for tooth #' + str(toothId+1)
            
            # Initialization of mean vector (xStriped), covariance matrix, x-vector, x-striped-vector (mean), eigenvectors (P) and eigenvalues.
            processedLandmarks = procrustes.procrustesMatrix(landmarks,100)
            if debugFB : print '   DB: Procrustes ready for tooth #' + str(toothId+1)
            pcMean, pcEigv = cv2.PCACompute(np.transpose(stackPoints(processedLandmarks)))
            if debugFB : print '   DB: PCA ready for tooth #' + str(toothId+1)
            xStacked = xStriped = np.transpose(pcMean)        
            
            covar, _ = cv2.calcCovarMatrix(stackPoints(processedLandmarks), cv2.cv.CV_COVAR_SCRAMBLED | cv2.cv.CV_COVAR_SCALE | cv2.cv.CV_COVAR_COLS)    
            eigval = np.sort(np.linalg.eigvals(covar), kind='mergesort')[::-1]
            
            # Number of modes
            coverage = 0
            nbModes = 0
            eigval_total = np.sum(eigval)
            for value in eigval:
                coverage += value/eigval_total
                nbModes += 1
                if coverage >= 0.99:
                    break
            
            P = np.transpose(pcEigv[:nbModes]) # normalized
            eigval = eigval[:nbModes]

            # Initialization of the initial points
            if autoInitPoints : X = init_points[:,toothId,:]
            else : X = ip.getModelPointsManually(personToFitId, toothId)
            
            # Draw the initial points
            initialImage = imageToFit.copy()
            cv2.polylines(initialImage, np.int32([X]), True, 255, thickness = 2)
            #showScaled(initialImage, windowscale, 'initial', False)
            
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
                scale, rotation = procrustes.alignShapes(unstackPointsForPerson(xStacked), y)
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
                # and stop iterating after a certain threshold
                differences = prev_b - b
                prev_b = b
                
                stop = True
                for diff in differences:
                    if np.abs(diff) > 0.01:
                        stop = False
                        
                it += 1
                
            if debugFB : print '   DB: Tooth #' + str(toothId+1) + ' finished in ' + str(it) + ' iterations.'
            
            # Draw the model on the radiograph
            x = unstackPointsForPerson(xStriped + np.dot(P, b))
            X = procrustes.rotateMatrixForPerson(x, rotation)
            X = procrustes.scaleMatrixForPerson(X, scale)
            X = procrustes.translateMatrixForPerson(X, translation)
            
            cv2.polylines(contourImage, np.int32([X]), True, 255, thickness = 2)
            #showScaled(contourImage, windowscale, 'contours', True)
            
            cv2.fillPoly(segmentsImage, np.int32([[X]]), 128)
            #showScaled(segmentsImage, windowscale, 'segments', True)
            
        #cv2.imwrite('C:/Users/samue_000/Desktop/' + str(personToFitId+1) + 'i.jpg', initialImage)
        #cv2.imwrite('C:/Users/samue_000/Desktop/' + str(personToFitId+1) + 'c.jpg', contourImage)
        #cv2.imwrite('C:/Users/samue_000/Desktop/' + str(personToFitId+1) + 's.jpg', segmentsImage)
        showScaled(contourImage, windowscale, 'contours', True)
            
        print 'DB: Radiograph #' + str(personToFitId+1) + ' is segmented.'
  
'''
MAIN PROGRAM
'''
if __name__ == '__main__':
    #plotVariations()
    findSegments()
