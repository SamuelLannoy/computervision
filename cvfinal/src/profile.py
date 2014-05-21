import numpy as np
import cv2
import main

'''
Created on 20-mei-2014

@author: vital.dhaveloose
'''

'''
Returns the mean of the profiles of the images (g striped), and the covariance matrix (S_g).
'''
def getModel(imgs, points, N):
    profiles = np.zeros((points.shape[0], points.shape[1], 2*N+1), dtype=np.int)
    means = np.zeros((points.shape[0], 2*N+1), dtype=np.int)
    
    for p in range(points.shape[1]):
        directions = getDirections(points[:,p,:])
        for l in range(points.shape[0]):
            profiles[l,p,:] = getProfile(imgs[p], points[l,p,:], directions[l,:], N)
    means = np.mean(profiles, axis=1)
    return means

def getDirections(points):
    dirs = np.zeros(points.shape[0],2)
    for i in range(dirs.shape[0]):
        dirs[i] = np.array([1,0])
    return dirs

'''
Returns an array of 2*N+1 grey values that represent the
profile in the given image, at the given point in the given direction.
'''
def getProfile(img, point, direction, N):
    (xs_profile, ys_profile) = getProfilePixels(point, direction, N)
    return img[ys_profile, xs_profile]
    
'''
direction is a unit vector that goes what we call Right
'''
def getProfilePixels(point, direction, N):
    delta_x = 1.0/np.abs(direction[0])
    delta_y = 1.0/np.abs(direction[1])
    
    t_next_x = delta_x/2
    t_next_y = delta_y/2
    
    profile_xs = np.zeros(2*N+1, dtype=np.int)
    profile_ys = np.zeros(2*N+1, dtype=np.int)
    
    curr_cell_L = point
    curr_cell_R = point
    
    profile_xs[N] = point[0]
    profile_ys[N] = point[1]
    
    for it in range(1,N+1):
        if t_next_x < t_next_y :
            # nu x-traversie
            curr_cell_R = (curr_cell_R[0] + 1, curr_cell_R[1])
            curr_cell_L = (curr_cell_L[0] - 1, curr_cell_L[1])
            t_next_x += delta_x
        elif t_next_y < t_next_x:
            # nu y-traversie
            curr_cell_R = (curr_cell_R[0], curr_cell_R[1] + 1)
            curr_cell_L = (curr_cell_L[0], curr_cell_L[1] - 1)
            t_next_y += delta_y
        else:
            #traversie door snijpunt tussen pixels
            curr_cell_R = (curr_cell_R[0] + 1, curr_cell_R[1])
            curr_cell_L = (curr_cell_L[0] - 1, curr_cell_L[1])
            t_next_x += delta_x
            curr_cell_R = (curr_cell_R[0], curr_cell_R[1] + 1)
            curr_cell_L = (curr_cell_L[0], curr_cell_L[1] - 1)
            t_next_y += delta_y
            
        profile_xs[N+it] = curr_cell_R[0]
        profile_xs[N-it] = curr_cell_L[0]
        profile_ys[N+it] = curr_cell_R[1]
        profile_ys[N-it] = curr_cell_L[1]
    
    return profile_xs, profile_ys
            
            
'''
Returns an integer that indicates the translation of the profileToMatch
to the learnedProfile so that the Mahalanobis distance becomes minimal.
'''
#def matchProfile(learnedProfile, profileToMatch):
    
img1 = cv2.imread('../data/Radiographs/01.tif',0)
img2 = cv2.imread('../data/Radiographs/02.tif',0)
imgs = np.array([img1,img2])
points = main.readLandmarks(1, 2, 40)
#img = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,12,13,14]])
#print getProfile(img, (1,1), (np.sqrt(3.0)/2.0, 0.5), 10)
print getModel(imgs, points, 10)
    