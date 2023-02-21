import numpy as np
import cv2

def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    DoG_pyramid = []
    ################
    # TO DO ...
    # compute DoG_pyramid here
    top_level = levels[-1]
    for i in levels:
        if i == levels[0]:
            continue 
        DoG_pyramid.append(gaussian_pyramid[:,:,i+1] - gaussian_pyramid[:,:,i])
    DoG_pyramid = np.stack(DoG_pyramid,axis=-1)
    DoG_levels = levels[1:]
    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    principal_curvature = None
    ##################
    # TO DO ...
    # Compute principal curvature here
    D_xx = cv2.Sobel(DoG_pyramid,ddepth=-1,dx=2,dy=0,ksize=3)
    D_yy = cv2.Sobel(DoG_pyramid,ddepth=-1,dx=0,dy=2,ksize=3)
    D_xy = cv2.Sobel(DoG_pyramid,ddepth=-1,dx=1,dy=1,ksize=3)

    principal_curvature = np.empty_like(DoG_pyramid)
    trace = D_xx + D_yy
    det = D_xx*D_yy - D_xy**2
    principal_curvature = trace**2/det
    return principal_curvature

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = None
    ##############
    #  TO DO ...
    # Compute locsDoG here
    X = []; Y = []; Z = []
    for level in DoG_levels:
        stacked = np.dstack((DoG_pyramid[1:-1, 1:-1, level],
                             DoG_pyramid[1:-1, :-2, level],
                             DoG_pyramid[:-2, 1:-1, level],
                             DoG_pyramid[1:-1, 2:, level],
                             DoG_pyramid[2:, 1:-1, level],
                             DoG_pyramid[:-2, 2:, level],
                             DoG_pyramid[2:, :-2, level],
                             DoG_pyramid[2:, 2:, level],
                             DoG_pyramid[:-2, :-2, level]))
        if level == DoG_levels[0]:
            stacked = np.dstack((stacked, DoG_pyramid[1:-1, 1:-1, level+1]))
        elif level == DoG_levels[-1]:
            stacked = np.dstack((stacked, DoG_pyramid[1:-1, 1:-1, level-1]))
        else:
            stacked = np.dstack((stacked, DoG_pyramid[1:-1, 1:-1, level-1],
                                          DoG_pyramid[1:-1, 1:-1, level+1]))

        max_stacked = np.amax(stacked, axis = 2)
        min_stacked = np.amin(stacked, axis = 2)
        keypoint_matrix_max = np.where(DoG_pyramid[1:-1, 1:-1, level] == max_stacked,
                                        DoG_pyramid[1:-1, 1:-1, level], 0)
        keypoint_matrix_min = np.where(DoG_pyramid[1:-1, 1:-1, level] == min_stacked,
                                        DoG_pyramid[1:-1, 1:-1, level], 0)
        keypoint_matrix = keypoint_matrix_max + keypoint_matrix_min

        keypoint_matrix = np.where(keypoint_matrix > th_contrast, keypoint_matrix, 0)
        keypoint_matrix = np.where(principal_curvature[1:-1, 1:-1, level] < th_r, keypoint_matrix, 0)

        x, y = np.nonzero(keypoint_matrix)
        z = np.full(len(x), level)
        X.extend(x+1); Y.extend(y+1); Z.extend(z)

    locsDoG = np.stack([np.asarray(Y), np.asarray(X), np.asarray(Z)], axis = -1)
    return locsDoG
   

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, gauss_pyramid here
    gauss_pyramid = createGaussianPyramid(im,sigma0,k,levels)
    DoG_pyramid,DoG_levels = createDoGPyramid(gauss_pyramid,levels)
    principal_curvature = computePrincipalCurvature(DoG_pyramid)
    locsDoG = getLocalExtrema(DoG_pyramid,DoG_levels,principal_curvature,th_contrast,th_r)
    return locsDoG, gauss_pyramid







if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
    displayPyramid(im_pyr)
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    displayPyramid(DoG_pyr)
    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    #displayPyramid(pc_curvature)
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)
    position = []
    for point in locsDoG:
        position.append((int(point[0]),int(point[1])))
    position = np.array(position)
    cv_kpts1 = [cv2.KeyPoint(int(position[i][0]), int(position[i][1]), 1)
                        for i in range(position.shape[0])]
    output = np.copy(im)
    cv2.drawKeypoints(im,cv_kpts1,output,(0,255,0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('../results/detected_keypoints.jpg', output)



