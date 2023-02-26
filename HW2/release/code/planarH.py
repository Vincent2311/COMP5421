import numpy as np
import cv2
from BRIEF import briefLite, briefMatch

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    # TO DO ...
    p2_hmg = np.stack((p2[0], p2[1], np.ones(p2.shape[1])), axis = 1)

    first = np.empty((2*p2_hmg.shape[0],3))
    first[0::2] = -p2_hmg
    first[1::2] = 0

    second = np.empty((2*p2_hmg.shape[0],3))
    second[0::2] = 0
    second[1::2] = -p2_hmg

    third = np.empty((2*p2_hmg.shape[0],3))
    third[0::2] = p2_hmg*(np.transpose([p1[0]]))
    third[1::2] = p2_hmg*(np.transpose([p1[1]]))

    a_matrix = np.hstack((first, second, third))
    _, _, vh = np.linalg.svd(a_matrix)
    H2to1 = vh[-1, :].reshape((3, 3))
    return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    ###########################
    # TO DO ...
    max_inlier = 0
    bestH = np.empty((3,3))
    p1 = locs1[matches[:,0]]
    p2 = locs2[matches[:,1]]
    p1 = np.delete(p1, 2, 1) 
    p2 = np.delete(p2, 2, 1) 
    for _ in range(0,num_iter):
        sampled = np.random.randint(p1.shape[0], size=4)
        H = computeH(np.transpose(p1[sampled]),np.transpose(p2[sampled]))
        
        p1_hmg = np.ones((p1.shape[0],3))
        p1_hmg[:,:-1] = p1
        p2_hmg = np.ones((p2.shape[0],3))
        p2_hmg[:,:-1] = p2
        proj_locs2 = np.dot(H,np.transpose(p2_hmg))
        proj_locs2 = proj_locs2/proj_locs2[2,:] 

        loss = np.sum((np.transpose(p1_hmg) - proj_locs2) ** 2, axis=0) ** 0.5
        inliers = len(loss[loss <= tol])
        if inliers > max_inlier:
            max_inlier = inliers
            bestH = H

    return bestH
        
    

if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    ransacH(matches, locs1, locs2, num_iter=5000, tol=2)

