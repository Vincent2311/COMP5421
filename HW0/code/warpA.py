import numpy as np
def warp(im, A, output_shape):
    """ Warps (h,w) image im using affine (3,3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_size.
    Uses nearest neighbor interpolation."""
    
    inv_A = np.linalg.inv(A)
    output = np.empty((output_shape[0], output_shape[1]))
    
    coords_x, coords_y = np.indices(output_shape)
    coords = np.stack((coords_x.ravel(), coords_y.ravel(), np.ones(coords_y.size)))
    coords_vec = np.reshape(coords, (3, output_shape[0]*output_shape[1]))
    coords_vec = np.round(inv_A.dot(coords_vec)).astype(int)
    coords_vec[0][coords_vec[0] >= output_shape[0]] = 0
    coords_vec[1][coords_vec[1] >= output_shape[1]] = 0
    
    output = np.where(coords_vec[0] > 0, im[coords_vec[0], coords_vec[1]], 0) 
    output = np.where(coords_vec[1] > 0, output, 0)
    output = output.reshape(output_shape[0], output_shape[1]) 
   
    return output