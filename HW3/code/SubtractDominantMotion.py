import numpy as np
import scipy.ndimage.morphology
from LucasKanadeAffine import LucasKanadeAffine
from scipy.interpolate import RectBivariateSpline

def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
    # put your implementation here
    
    mask = np.ones(image1.shape, dtype=bool)
    M = LucasKanadeAffine(image1,image2)
    
    x_coor = np.arange(0,image1.shape[1])
    y_coor = np.arange(0,image1.shape[0])
    x_coor,y_coor = np.meshgrid(x_coor,y_coor)
    It1_spline = RectBivariateSpline(np.arange(image2.shape[0]),np.arange(image2.shape[1]),image2)
    It_spline = RectBivariateSpline(np.arange(image1.shape[0]),np.arange(image1.shape[1]),image1)
    It_template = It_spline.ev(y_coor,x_coor)
    
    x_coor_wrapped = x_coor *M[0,0] + y_coor*M[0,1] + M[0,2]
    y_coor_wrapped = x_coor *M[1,0] + y_coor*M[1,1] + M[1,2]

    invalid = (x_coor_wrapped < 0) | (x_coor_wrapped >= image2.shape[1]) | (y_coor_wrapped <0) | (y_coor_wrapped >= image2.shape[0])
    It1_wrapped =  It1_spline.ev(y_coor_wrapped,x_coor_wrapped)
    It1_wrapped[invalid] = 0
    It_template[invalid] = 0

    diff = It1_wrapped - It_template
    threshold = 0.1
    idx = diff > threshold
    mask[idx] = 1
    mask = scipy.ndimage.morphology.binary_dilation(mask).astype(mask.dtype)

    return mask
