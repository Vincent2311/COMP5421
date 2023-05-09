import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeBasis(It, It1, rect, bases):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	bases: [n, m, k] where nxm is the size of the template.
	# Output:
	#	p: movement vector [dp_x, dp_y]

    # Put your implementation here
    threshold = 0.0001 

    x_min, y_min, x_max, y_max = rect[0], rect[1], rect[2], rect[3]
    rect_It1_spline = RectBivariateSpline(np.arange(It1.shape[0]),np.arange(It1.shape[1]),It1)
    rect_It_spline = RectBivariateSpline(np.arange(It.shape[0]),np.arange(It.shape[1]),It)

    p = np.zeros(2)
    delta_p = np.array((2,))
    B = []
    for i in range(bases.shape[2]):
        B.append(bases[:, :, i].flatten())
    B = np.transpose(np.array(B))
    B_ = B@B.T

    x = np.arange(x_min, x_max + 0.1)
    y = np.arange(y_min, y_max + 0.1)
    X, Y = np.meshgrid(x, y)
    It_rect = rect_It_spline.ev(Y, X)

    while np.sum(delta_p**2) > threshold:
        x_ = np.arange(x_min + p[0], x_max + 0.5 + p[0])
        y_ = np.arange(y_min + p[1], y_max + 0.5 + p[1])
        X_, Y_ = np.meshgrid(x_, y_)
        It1_rect = rect_It1_spline.ev(Y_, X_)        

        b = It_rect.flatten() - It1_rect.flatten()

        x_gradient = rect_It1_spline.ev(Y_, X_, dx=0, dy=1).flatten()
        y_gradient = rect_It1_spline.ev(Y_, X_, dx=1, dy=0).flatten()
        
        A = np.zeros((x_gradient.shape[0], 2))
        A[:, 0] = x_gradient
        A[:, 1] = y_gradient

        A = A - B_@A
        b = b-B_@b 

        delta_p = np.linalg.inv(A.T @ A) @ A.T @ b
        p += delta_p
    
    return p

    
