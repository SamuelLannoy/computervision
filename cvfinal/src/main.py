import cv2
import numpy as np
import procrustes
import plot_teeth
import profile
import radiograph as rg
import init_model

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
    for personId in range(nbPersons):
        images[personId] = rg.preprocess(cv2.imread('../data/Radiographs/' + ("0" + str(personId+1) if personId < 10 else str(personId+1)) + '.tif',0))
    return images

'''
Returns the image to fit from the given person id (15..30)
'''
def readImageToFit(personId):
    return rg.preprocess(cv2.imread('../data/Radiographs/extra/' + str(personId) + '.tif',0))

def fitToothModelToImage(landmarks, images, imageToFit):
    return
    
if __name__ == '__main__':
    #landmarks = readLandmarks(1, 2, 40)
    #images = readImages(2)
    imageToFit = readImageToFit(15)
    
    print init_model.getModelPoints(imageToFit)
    
    '''processedLandmarks = procrustes.procrustesMatrix(landmarks,0)
    stackedProcessedLandmarks = np.vstack((processedLandmarks[:,:,0],processedLandmarks[:,:,1])) 
    pcLandmarks = cv2.PCACompute(np.transpose(stackedProcessedLandmarks))
    pcLandmarksMean = pcLandmarks[0][0]
    pcLandmarksEig = pcLandmarks[1]'''
    
    '''
    PLOT MEAN TOOTH
    pcLandmarksMeanX = pcLandmarksMean[0:len(pcLandmarksMean)/2]
    pcLandmarkMeanY = pcLandmarksMean[len(pcLandmarksMean)/2:len(pcLandmarksMean)]
    plot_teeth.plotToothXY(pcLandmarksMeanX, pcLandmarkMeanY)
    plot_teeth.show()
    '''

    
    
