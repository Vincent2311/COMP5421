import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]
	
    # Put your implementation here
    threshold = 0.005 
    x_coor = np.arange(rect[0][0],rect[2][0])
    y_coor = np.arange(rect[1][0],rect[3][0])
    x_coor,y_coor = np.meshgrid(x_coor,y_coor) 
    rect_It1_spline = RectBivariateSpline(np.arange(It1.shape[0]),np.arange(It1.shape[1]),It1)
    rect_It_spline = RectBivariateSpline(np.arange(It.shape[0]),np.arange(It.shape[1]),It)

    It_rect = rect_It_spline.ev(y_coor,x_coor)

    p = p0
    delta_p = np.array((2,))
    while np.sum(delta_p**2) > threshold:
        x_coor_wrapped = p[0] + x_coor
        y_coor_wrapped = p[1] + y_coor
        It1_wrapped = rect_It1_spline.ev(y_coor_wrapped,x_coor_wrapped)
        b = It1_wrapped.flatten() - It_rect.flatten()

        y_gradient = rect_It1_spline.ev(y_coor_wrapped,x_coor_wrapped,dx = 1,dy = 0).flatten()
        x_gradient = rect_It1_spline.ev(y_coor_wrapped,x_coor_wrapped,dx = 0,dy=1).flatten()

        A = np.empty((x_gradient.shape[0],2))
        A[:,0] = x_gradient
        A[:,1] = y_gradient

        delta_p = np.linalg.inv(A.T @ A) @ A.T @ b
        p += delta_p
    
    return p
