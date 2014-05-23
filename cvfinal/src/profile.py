import numpy as np
import cv2
import main

'''
Returns the mean of the profiles of the images (g striped), and the covariance matrix (S_g).

points is LM x Pers x Dim
'''
def getModel(imgs, points, n):
    profiles = np.zeros((points.shape[0], points.shape[1], 2*n+1))
    
    for p in range(points.shape[1]):
        profiles[:,p,:] = getProfilesForPerson(imgs[p], points[:,p,:], n)
    
    covars = np.zeros((points.shape[0], 2*n+1, 2*n+1))
    means = np.zeros((points.shape[0], 2*n+1))
    
    for l in range(points.shape[0]):
        [covars[l], means[l]] = cv2.calcCovarMatrix(profiles[l,:,:]*1.0, cv2.cv.CV_COVAR_NORMAL | cv2.cv.CV_COVAR_ROWS)
        
    return covars, means

'''
Returns the directions for a certain person,
given his model points (landmarks).

points is LM x Dim
'''
def getDirectionsForPerson(points):
    dirs = np.zeros(points.shape)
    for l in range(points.shape[0]):
        dirs[l,1] = 1.0
    return dirs
    #TODO: real implementation (bisectrice? loodrecht op vorig stuk?)

'''
Returns the profiles for a person, given all his model points (landmarks) and
the number of pixels that should be sampled at each side.

points is LM x Dim
'''
def getProfilesForPerson(img, points, n):
    directions = getDirectionsForPerson(points)
    profiles = np.zeros((points.shape[0], 2*n+1))
    for l in range(points.shape[0]):
        profiles[l,:] = getProfileForPersonAndLandmark(img, points[l,:], directions[l,:], n)
    return profiles

'''
Returns an array of 2*N+1 grey values that represent the
profile in the given image, at the given point in the given direction.

point is two-tuple
'''
def getProfileForPersonAndLandmark(img, point, direction, n):
    (xs_profile, ys_profile) = getProfilePixels(point, direction, n)
    prof = img[ys_profile, xs_profile]
    norm = np.linalg.norm(prof, 1)
    return prof*1.0/norm # typing problem: int <> float

'''
Returns the profile pixels given the middle point, the direction of the profile and
the number of pixels that should be returned on the left and right of the given point.

direction is a unit vector that goes what we call Right
'''
def getProfilePixels(point, direction, n):
    delta_x = 1.0/np.abs(direction[0])
    delta_y = 1.0/np.abs(direction[1])
    
    t_next_x = delta_x/2
    t_next_y = delta_y/2
    
    profile_xs = np.zeros(2*n+1, dtype=np.int)
    profile_ys = np.zeros(2*n+1, dtype=np.int)
    
    curr_cell_L = point
    curr_cell_R = point
    
    profile_xs[n] = point[0]
    profile_ys[n] = point[1]
    
    for it in range(1,n+1):
        if t_next_x < t_next_y :
            # x-traversion
            curr_cell_R = (curr_cell_R[0] + 1, curr_cell_R[1])
            curr_cell_L = (curr_cell_L[0] - 1, curr_cell_L[1])
            t_next_x += delta_x
        elif t_next_y < t_next_x:
            # y-traversion
            curr_cell_R = (curr_cell_R[0], curr_cell_R[1] + 1)
            curr_cell_L = (curr_cell_L[0], curr_cell_L[1] - 1)
            t_next_y += delta_y
        else:
            # traversion through intersection of pixels
            curr_cell_R = (curr_cell_R[0] + 1, curr_cell_R[1])
            curr_cell_L = (curr_cell_L[0] - 1, curr_cell_L[1])
            t_next_x += delta_x
            curr_cell_R = (curr_cell_R[0], curr_cell_R[1] + 1)
            curr_cell_L = (curr_cell_L[0], curr_cell_L[1] - 1)
            t_next_y += delta_y
            
        profile_xs[n+it] = curr_cell_R[0]
        profile_xs[n-it] = curr_cell_L[0]
        profile_ys[n+it] = curr_cell_R[1]
        profile_ys[n-it] = curr_cell_L[1]
    
    return profile_xs, profile_ys
            
            
'''
Returns an integer that indicates the translation of the profileToMatch
to the learnedProfile so that the Mahalanobis distance becomes minimal.
'''
def matchProfiles(model, profiles):
    for l in range(profiles.shape[0]):
        covar = model[0][l]
        mean = model[1][l]
        icovar = np.linalg.pinv(covar)
        
        m = (profiles.shape[1]-1)/2
        n = (model[1].shape[1]-1)/2
        
        dist = np.zeros(2*(m-n)+1)
        for t in range(2*(m-n)+1):
            tMean = translateMean(mean, t, m)
            tIcovar = translateCovar(icovar, t, m)
            dist[t] = cv2.Mahalanobis(profiles[l], tMean, tIcovar)
    
    print dist
    return np.argmin(dist)
            
'''
Translates the given mean into a larger vector with dimension 2*m+1,
with distance t from the right.
'''
def translateMean(mean, t, m):
    return np.concatenate((np.zeros(t), mean, np.zeros(2*m+1-t-mean.shape[0])))

'''
Translates the given covariance matrix into a larger matrix with dimension (2*m+1, 2*m+1),
with distance t from the right and the top.
'''
def translateCovar(covar, t, m):
    ret = np.zeros((2*m+1, 2*m+1))
    ret[t:t+covar.shape[0], t:t+covar.shape[1]] = covar #OLD: ret[t:2*m+1-t-covar.shape[0]] = covar
    return ret



'''
MAIN PROGRAM
'''    
img1 = cv2.imread('../data/Radiographs/01.tif',0)
img2 = cv2.imread('../data/Radiographs/02.tif',0)
img3 = cv2.imread('../data/Radiographs/03.tif',0)
imgs = np.array([img1,img2])
points = main.readLandmarks(1, 2, 40)
#img = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,12,13,14]])
#print getProfileForPersonAndLandmark(img, (1,1), (np.sqrt(3.0)/2.0, 0.5), 10)
covars, means = getModel(imgs, points, 2)
print matchProfiles((covars, means), getProfilesForPerson(img3, points[:,1,:], 6))
