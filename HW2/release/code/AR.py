import numpy as np
import cv2

def compute_extrinsics(K, H):
    '''
    INPUTS:
        K - a 3 x 3 matrix containing camera intrinsic parameters
        H - a 3 x 3 matrix for estimated homography
    OUTPUTS:
        R - extrinsic parameters for relative 3D rotation 
        t - extrinsic parameters for translation
    '''
    new_H = np.matmul(np.linalg.inv(K),H)
    U,L, VT = np.linalg.svd(new_H)
    R = np.empty((3,3))
    R[:,:2] = U @ np.array([[1,0],[0,1],[0,0]]) @ VT
    R[:,2] = np.cross(R[:,0],R[:,1])
    if np.linalg.det(R) < 0:
        R[:,2] = R[:,2] * -1
    scale = np.sum(np.concatenate(new_H/R[:,:2]))/6
    t = R[:,2] / scale
    
    return R,t

def project_extrinsics(K, W, R, t):
    pass