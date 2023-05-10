import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image

	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    threshold = 0.005

    x_coor = np.arange(0,It.shape[1])
    y_coor = np.arange(0,It.shape[0])
    x_coor,y_coor = np.meshgrid(x_coor,y_coor)
    It1_spline = RectBivariateSpline(np.arange(It1.shape[0]),np.arange(It1.shape[1]),It1)
    It_spline = RectBivariateSpline(np.arange(It.shape[0]),np.arange(It.shape[1]),It)
    It_template = It_spline.ev(y_coor,x_coor)

    p = M.flatten()
    delta_p = np.array((6,))
    
    y_gradient = It1_spline.ev(y_coor,x_coor,dx = 1,dy = 0).flatten()
    x_gradient = It1_spline.ev(y_coor,x_coor,dx = 0,dy=1).flatten()

    A = np.empty((x_gradient.shape[0],6))
    A[:,0] = x_coor.flatten() * x_gradient
    A[:,1] = y_coor.flatten() * x_gradient
    A[:,2] = x_gradient
    A[:,3] = x_coor.flatten() * y_gradient
    A[:,4] = y_coor.flatten() * y_gradient
    A[:,5] = y_gradient
    
    while np.sum(delta_p**2) > threshold:
        x_coor_wrapped = x_coor *p[0] + y_coor*p[1] + p[2]
        y_coor_wrapped = x_coor *p[3] + y_coor*p[4] + p[5]
        mask = (x_coor_wrapped >=0) &(x_coor_wrapped < It1.shape[1]) & (y_coor_wrapped >=0) &(y_coor_wrapped < It1.shape[0])
        x_coor_wrapped = x_coor_wrapped[mask]
        y_coor_wrapped = y_coor_wrapped[mask]
        It1_wrapped =  It1_spline.ev(y_coor_wrapped,x_coor_wrapped)

        A_ = A[mask.flatten()]
        b = It1_wrapped.flatten() - It_template[mask].flatten() 

        delta_p = np.linalg.inv(A_.T @ A_) @ A_.T @ b
        
        M = np.vstack((np.reshape(p, (2, 3)), np.array([[0, 0, 1]])))
        delta_M = np.vstack((np.reshape(delta_p, (2, 3)), np.array([[0, 0, 1]])))
        delta_M[0, 0] += 1
        delta_M[1, 1] += 1
        M = np.dot(M, np.linalg.inv(delta_M))

        p = M[:2, :].flatten()
        
    M = M[:2, :]
    return M