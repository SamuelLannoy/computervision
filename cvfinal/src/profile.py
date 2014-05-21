import numpy as np

'''
Created on 20-mei-2014

@author: vital.dhaveloose
'''

'''
Returns the mean of the profiles of the images (g striped), and the covariance matrix (S_g).
'''
#def getModel(imgs, point, direction, N):


'''
Returns an array of 2*N+1 grey values that represent the
profile in the given image, at the given point in the given direction.
'''
#def getProfile(img, point, direction, N):
    
'''
direction is a unit vector that goes what we call Right
'''
def getProfilePixels(point, direction, N):
    delta_x = 1.0/np.abs(direction[0])
    delta_y = 1.0/np.abs(direction[1])
    
    t_next_x = delta_x/2
    t_next_y = delta_y/2
    
    profile_xs = np.zeros(2*N+1)
    profile_ys = np.zeros(2*N+1)
    
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
    
print getProfilePixels((100,100), (np.sqrt(3.0)/2.0, 0.5), 10)

    