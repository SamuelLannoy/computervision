import cv2
import numpy as np
import procrustes
import plot_teeth

'''
Creates a  tooth matrix from the landmarks in the files.
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
    
if __name__ == '__main__':
    landmarks = readLandmarks(1, 14, 40)
    processedLandmarks = procrustes.procrustesMatrix(landmarks)
    stackedProcessedLandmarks = np.vstack((processedLandmarks[:,:,0],processedLandmarks[:,:,1])) 
    pcLandmarks = cv2.PCACompute(np.transpose(stackedProcessedLandmarks))
    pcLandmarksMean = pcLandmarks[0][0]
    pcLandmarksEig = pcLandmarks[1]
    
    pcLandmarksMeanX = pcLandmarksMean[0:len(pcLandmarksMean)/2]
    pcLandmarkMeanY = pcLandmarksMean[len(pcLandmarksMean)/2:len(pcLandmarksMean)]
    plot_teeth.plotToothXY(pcLandmarksMeanX, pcLandmarkMeanY)
    plot_teeth.show()

    
    
