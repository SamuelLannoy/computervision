import numpy as np
import cv2
import main
import matplotlib.pyplot as ppl
import plot_teeth as pt

np.set_printoptions(threshold='nan')

'''
Returns the mean of the profiles of the images (g striped), and the covariance matrix (S_g).

points is LM x Pers x Dim
'''
def getModel(imgs, points, directions, n):
    profiles = np.zeros((points.shape[0], points.shape[1], 2*n+1))
    
    for p in range(points.shape[1]):
        profiles[:,p,:] = getProfilesForPerson(imgs[p], points[:,p,:], directions[:,p,:], n)
        
    covars = np.zeros((points.shape[0], 2*n+1, 2*n+1))
    means = np.zeros((points.shape[0], 2*n+1))
    
    for l in range(points.shape[0]):
        [covars[l], means[l]] = cv2.calcCovarMatrix(profiles[l,:,:]*1.0, cv2.cv.CV_COVAR_NORMAL | cv2.cv.CV_COVAR_ROWS)
        #covars[l] = np.cov(profiles[l,:,:]*1.0, rowvar=0)
        #means[l] = np.mean(profiles[l,:,:]*1.0, axis=0)

    return covars, means

'''
Returns the directions for a certain person,
given his model points (landmarks).

points is LM x Dim
'''
def getDirectionsForPerson(points):
    dirs = np.zeros(points.shape)
    
    middle = np.average(points, 0)
    
    for l in range(0, points.shape[0]):        
        if l == 0:
            dir1 = [points[-1][0]-points[0][0], points[-1][1]-points[0][1]]
        else:
            dir1 = [points[l-1][0]-points[l][0], points[l-1][1]-points[l][1]]
        
        if l == points.shape[0]-1:
            dir2 = [points[0][0]-points[-1][0], points[0][1]-points[-1][1]]
        else:
            dir2 = [points[l+1][0]-points[l][0], points[l+1][1]-points[l][1]]
        
        dir1 = dir1/np.linalg.norm(dir1, 2)
        dir2 = dir2/np.linalg.norm(dir2, 2)
        dirnorm = np.linalg.norm(dir1 + dir2, 2)
        
        if dirnorm != 0:
            dirs[l] = (dir1 + dir2) / dirnorm
        else:
            dirs[l] = [-dir1[1], dir1[0]]
        
        toMiddle = middle - points[l]
        if np.dot(toMiddle, np.transpose(dirs[l])) < 0:
            dirs[l] = -dirs[l]
            
    return dirs

'''
points is LM x Pers x Dim
'''
def getDirections(points):
    dirs = np.zeros_like(points)
    for p in range(points.shape[1]):
        dirs[:,p,:] = getDirectionsForPerson(points[:,p,:])
    return dirs

'''
Returns the profiles for a person, given all his model points (landmarks),
directions and the number of pixels that should be sampled at each side.

points is LM x Dim
'''
def getProfilesForPerson(img, points, directions, n):
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
        if t_next_x < t_next_y:
            # x-traversion
            if direction[0] < 0:
                curr_cell_R = (curr_cell_R[0] - 1, curr_cell_R[1])
                curr_cell_L = (curr_cell_L[0] + 1, curr_cell_L[1])
            else:
                curr_cell_R = (curr_cell_R[0] + 1, curr_cell_R[1])
                curr_cell_L = (curr_cell_L[0] - 1, curr_cell_L[1])
            t_next_x += delta_x
        elif t_next_y < t_next_x:
            # y-traversion
            if direction[1] < 0:
                curr_cell_R = (curr_cell_R[0], curr_cell_R[1] - 1)
                curr_cell_L = (curr_cell_L[0], curr_cell_L[1] + 1)  
            else:
                curr_cell_R = (curr_cell_R[0], curr_cell_R[1] + 1)
                curr_cell_L = (curr_cell_L[0], curr_cell_L[1] - 1)
            t_next_y += delta_y
        else:
            # traversion through intersection of pixels
            if direction[0] < 0:
                curr_cell_R = (curr_cell_R[0] - 1, curr_cell_R[1])
                curr_cell_L = (curr_cell_L[0] + 1, curr_cell_L[1])
            else:
                curr_cell_R = (curr_cell_R[0] + 1, curr_cell_R[1])
                curr_cell_L = (curr_cell_L[0] - 1, curr_cell_L[1])
            t_next_x += delta_x
            if direction[1] < 0:
                curr_cell_R = (curr_cell_R[0], curr_cell_R[1] - 1)
                curr_cell_L = (curr_cell_L[0], curr_cell_L[1] + 1)  
            else:
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
    tProfiles = np.zeros(profiles.shape[0])
    for l in range(profiles.shape[0]):
        covar = model[0][l]
        mean = model[1][l]
        icovar = cv2.invert(covar, flags=cv2.DECOMP_SVD)[1]
        
        m = (profiles.shape[1]-1)/2
        n = (model[1].shape[1]-1)/2
        
        dist = np.zeros(2*(m-n)+1)
        for t in range(2*(m-n)+1):
            tMean = translateMean(mean, t, m)
            tIcovar = translateCovar(icovar, t, m)
            dist[t] = cv2.Mahalanobis(profiles[l], tMean, tIcovar)
        tProfiles[l] = m-n-np.argmin(dist)
    
    ''' PROFILE VISUALISATION
    ppl.vlines(np.arange(mean.shape[0]), np.zeros_like(mean), mean)
    ppl.show()
    ppl.vlines(np.arange(profiles[20].shape[0]), np.zeros_like(profiles[20]), profiles[20])
    ppl.show()
    '''
    
    print tProfiles
    return tProfiles
            
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
    ret[t:t+covar.shape[0], t:t+covar.shape[1]] = covar
    return ret

'''
Translates the old model to a new model point
given the translation in number of pixels
'''
def getNewModelPoint(point, direction, n, tProfile):
    xs_profile, ys_profile = getProfilePixels(point, direction, n)
    index = n-tProfile
    return xs_profile[index], ys_profile[index]

def getNewModelPoints(imageToFit, points, model, n):
    directions = getDirectionsForPerson(points)
    profiles = getProfilesForPerson(imageToFit, points, directions, n)
    tProfiles = matchProfiles(model, profiles)
    
    newPoints = np.zeros_like(points)
    for i in range(points.shape[0]):
        newPoints[i] = getNewModelPoint(points[i], directions[i], n, tProfiles[i])
        
    return newPoints

'''
MAIN PROGRAM
'''    
if __name__ == '__main__':
    #img1 = cv2.imread('../data/Radiographs/01.tif',0)
    #img2 = cv2.imread('../data/Radiographs/02.tif',0)
    #img3 = cv2.imread('../data/Radiographs/03.tif',0)
    images, points = main.readData(1, 5, 40)
    dirs = getDirections(points)
    #print getProfileForPersonAndLandmark(img, (1,1), (np.sqrt(3.0)/2.0, 0.5), 10)
    covars, means = getModel(images, points, dirs, 2)
    #print matchProfiles((covars, means), getProfilesForPerson(img3, points[:,1,:], directions[:,1,:], 6))
    #pt.plotTooth(points[:,1,:])
    #pt.plotTooth(points[:,1,:] + 5*dirs[:,1,:])
    #pt.show()

