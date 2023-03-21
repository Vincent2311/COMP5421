"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import helper
import matplotlib.pyplot as plt

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    T = np.array([[1/M,0,0],[0,1/M,0],[0,0,1]])
    pts1_hmg = np.hstack((pts1,np.ones((pts1.shape[0],1)))).T
    pts2_hmg = np.hstack((pts2,np.ones((pts2.shape[0],1)))).T
    pts1_nor = np.matmul(T,pts1_hmg).T
    pts2_nor = np.matmul(T,pts2_hmg).T
    A1 = np.tile(pts1_nor,3)
    A2 = np.hstack((np.tile(pts2_nor[:,0][:,None],3),np.tile(pts2_nor[:,1][:,None],3),np.tile(pts2_nor[:,2][:,None],3)))
    A = A1 * A2
    u, s, vh = np.linalg.svd(A)
    F = vh[-1, :].reshape((3, 3))
    F = helper.refineF(F,pts1_nor[:,0:2],pts2_nor[:,0:2])
    F = T.T @ F @ T
    
    return F 


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    # Replace pass by your implementation
    T = np.array([[1/M,0,0],[0,1/M,0],[0,0,1]])
    pts1_hmg = np.hstack((pts1,np.ones((pts1.shape[0],1)))).T
    pts2_hmg = np.hstack((pts2,np.ones((pts2.shape[0],1)))).T
    pts1_nor = np.matmul(T,pts1_hmg).T
    pts2_nor = np.matmul(T,pts2_hmg).T
    A1 = np.tile(pts1_nor,3)
    A2 = np.hstack((np.tile(pts2_nor[:,0][:,None],3),np.tile(pts2_nor[:,1][:,None],3),np.tile(pts2_nor[:,2][:,None],3)))
    A = A1 * A2
    u, s, vh = np.linalg.svd(A)

    F1 = vh[-1, :].reshape((3, 3))
    F2 = vh[-2, :].reshape((3, 3))
    constraint = lambda x: np.linalg.det(x * F1 + (1-x)*F2)
    a0 = constraint(0)
    a2 = (constraint(-1) + constraint(1)) / 2 -a0
    a1 = 2*(constraint(1)-constraint(-1))/3 - (constraint(2)-constraint(-2))/12
    a3 = constraint(1) - a2 - a1 - a0 

    roots = np.roots([a3, a2, a1, a0])
    Farray = [root* F1 + (1-root) * F2 for root in roots]
    Farray = [helper.refineF(F,pts1_nor[:,0:2],pts2_nor[:,0:2]) for F in Farray]
    Farray = [T.T @ F @ T for F in Farray]
    return Farray


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    E = K2.T @ F @ K1
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    c1_1 = C1[0, :][:, None]
    c1_2 = C1[1, :][:, None]
    c1_3 = C1[2, :][:, None]

    c2_1 = C2[0, :][:, None]
    c2_2 = C2[1, :][:, None]
    c2_3 = C2[2, :][:, None]
    A = np.empty((4,4))
    P = np.empty((pts1.shape[0],4)) 
    for i in range(0,pts1.shape[0]):
        A[0,:] = np.reshape(c1_1 - c1_3 * pts1[i][0],(4,))
        A[1,:] = np.reshape(c1_2 - c1_3 * pts1[i][1],(4,))
        A[2,:] = np.reshape(c2_1 - c2_3 * pts2[i][0],(4,))
        A[3,:] = np.reshape(c2_2 - c2_3 * pts2[i][1],(4,))
        u, s, vh = np.linalg.svd(A)
        P[i] = vh[-1,:].reshape(1,4)
        P[i] = P[i] / P[i,-1]

    proj_p1 = (C1 @ P.T).T
    proj_p2 = (C2 @ P.T).T
    proj_p1 = proj_p1 / proj_p1[:,-1][:,None]
    proj_p1 = proj_p1[:,0:2]
    proj_p2 = proj_p2 / proj_p2[:,-1][:,None]
    proj_p2 = proj_p2[:,0:2]

    err = np.sum((proj_p1 - pts1)**2) + np.sum((proj_p2 - pts2)**2)
    return P[:,0:3] , err


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    pass

'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M):
    # Replace pass by your implementation
    pass

'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    pass

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    pass

'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass


    

    
