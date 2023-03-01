import numpy as np
import cv2
from planarH import computeH
import matplotlib.pyplot as plt

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
    U,L, Vh = np.linalg.svd(new_H[:,:2])
    R = np.empty((3,3))
    R[:,:2] = U @ np.array([[1,0],[0,1],[0,0]]) @ Vh
    R[:,2] = np.cross(R[:,0],R[:,1])
    if round(np.linalg.det(R)) == -1:
        R[:,2] = R[:,2] * -1
    scale = np.sum(np.concatenate(new_H[:,:2]/R[:,:2]))/6
    t = new_H[:,2] / scale
    
    return R,t

def project_extrinsics(K, W, R, t):
    extrinsic_matrix = np.empty((3,4))
    extrinsic_matrix[:,:3] = R
    extrinsic_matrix[:,3] = t
    x = np.matmul(np.matmul(K,extrinsic_matrix),W)
    x = x / x[2,:]
    return x

if __name__ == "__main__":
    im = cv2.imread('../data/prince_book.jpeg')
    p1 = np.array([[483, 1704, 2175, 67], [810, 781, 2217, 2286]])
    p2 = np.array([[0.0, 18.2, 18.2, 0.0], [0.0, 0.0, 26.0, 26.0]])
    H = computeH(p1,p2)
    K = np.array([[3043.72, 0.0, 1196.00], [0.0, 3043.72, 1604.00], [0.0, 0.0, 1.0]])
    R, t = compute_extrinsics(K,H)
    
    
    W = np.loadtxt('../data/sphere.txt')
   
    o_coord_2d = [800, 1400, 1] 
    o_coord_3d = np.matmul(np.linalg.inv(H), o_coord_2d)
    o_coord_3d = o_coord_3d/o_coord_3d[2]
    o_coord_3d[2] = 6.8581/2 
    W[0] = W[0] + o_coord_3d[0]
    W[1] = W[1] + o_coord_3d[1]
    W = np.vstack((W, np.ones((1, len(W[0])))))
    X = project_extrinsics(K, W, R, t)
    X = X.astype(int)

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.imshow(im)
    plt.plot(X[0, :], X[1, :], 'y', linewidth=1, markersize=1)
    plt.show()
    