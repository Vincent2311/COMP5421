import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches

def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    #######################################
    # TO DO ...
    size = (3*im1.shape[0],im1.shape[1]//5*4)
    pano_im = cv2.warpPerspective(im1, np.identity(3), size)
    pano_im = np.maximum(pano_im,cv2.warpPerspective(im2, H2to1, size))

    # save the image and matrix
    np.save('../results/q6_1.npy',H2to1)
    cv2.imwrite('../results/6_1.jpg', pano_im)
    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    ######################################
    # TO DO ...
    im2_H, im2_W, _ = im2.shape
    width = 3*im1.shape[1]
    proj_corner = np.dot(H2to1,np.array([[0,0,1],[im2_W,0,1],[0,im2_H,1],[im2_W,im2_H,1]]).T).T
    proj_corner = proj_corner.T
    proj_corner = (proj_corner/proj_corner[2,:]).T
    x_min = min(proj_corner[0,0],0,proj_corner[2,0]) 
    x_max = max(proj_corner[1,0],im2_W,proj_corner[3,0])
    y_min = min(proj_corner[0,1],0,proj_corner[1,1])
    y_max = max(proj_corner[2,1],im2_H,proj_corner[3,1])

    r = width/(x_max - x_min)
    height = round(r*(y_max - y_min))
    out_size = (width,height)

    M =np.array([[r,0,-r*x_min],[0,r,-r*y_min],[0,0,1]])
    warp_im1 = cv2.warpPerspective(im1, M, out_size)
    warp_im2 = cv2.warpPerspective(im2, np.matmul(M,H2to1), out_size)
    pano_im = np.maximum(warp_im1,warp_im2)

    cv2.imwrite('../results/q6_2 pan.jpg', pano_im)
    return pano_im

def generatePanorama(im1, im2):
    '''
    Generate and save panorama of im1 and im2.
    INPUT
        im1 and im2 - two images for stitching
    OUTPUT
        Blends img1 and warped img2 (with no clipping)
        and saves the panorama image.
    '''

    ######################################
    # TO DO ...
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    pano_im = imageStitching_noClip(im1, im2, H2to1)
    cv2.imwrite('../results/q6_3.jpg', pano_im)

if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    generatePanorama(im1,im2)
    # print(im1.shape)
    # locs1, desc1 = briefLite(im1)
    # locs2, desc2 = briefLite(im2)
    # matches = briefMatch(desc1, desc2)
    # # plotMatches(im1,im2,matches,locs1,locs2)
    # H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    # pano_im = imageStitching_noClip(im1, im2, H2to1)
    # #pano_im = imageStitching(im1, im2, H2to1)
    # cv2.imwrite('../results/panoImg.png', pano_im)
    # cv2.imshow('panoramas', pano_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()