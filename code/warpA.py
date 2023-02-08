import numpy as np
def warp(im, A, output_shape):
    """ Warps (h,w) image im using affine (3,3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_size.
    Uses nearest neighbor interpolation."""
    invA = np.linalg.inv(A)
    output = np.ones(output_shape,np.double)
    out_h, out_w = output_shape[0], output_shape[1]

    input_h, input_w = im.shape[0], im.shape[1]
    out_x = np.linspace(0,out_h-1,out_h)
    out_y = np.linspace(0,out_w-1,out_w)
    input_x = np.linspace(0,input_h-1,input_h)
    input_y = np.linspace(0,input_w-1,input_w)
    output_xv, output_yv = np.meshgrid(out_x,out_y)
    input_xv, input_yv = np.meshgrid(input_x,input_y)
    ones_arr = np.ones_like(output_xv)
    index_output = np.stack((output_xv,output_yv,ones_arr),axis=-1)
    index_input = np.stack((input_xv,input_yv,ones_arr),axis=-1)

    print(invA.shape)
    invA = np.tile(invA,(index_output.shape[0],1))
    print(invA.shape)
    print(index_output.shape)
    index = np.rint(np.matmul(invA,index_output))
    index[index[0] < 0 or index[0] >= im.shape[0] or index[1] < 0 or index[1] >= im.shape[1]] = (0,0)
    output[index != (0,0)] = im[index]
    # for x in range(0,out_h):
    #     for y in range(0,out_w):
    #         index = np.rint(np.matmul(invA,np.array([x,y,1])))
    #         index_x = int(index[0])
    #         index_y = int(index[1])
    #         if index_x < 0 or index_x >= im.shape[0] or index_y < 0 or index_y >= im.shape[1]:
    #             output[x][y] = 0
    #         else:
    #             output[x][y] = im[index_x][index_y]

    return output
