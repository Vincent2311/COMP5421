import numpy as np
import cv2
import os
import BRIEF
import matplotlib.pyplot as plt


if __name__ == '__main__':
    compareX, compareY = BRIEF.makeTestPattern()

    im = cv2.imread('../data/model_chickenbroth.jpg')
    rows,cols,_ = im.shape
    center = (cols/2, rows/2)  
    locs1, desc1 = BRIEF.briefLite(im)
    count = []

    for i in range(0,36):
        angle = i*10
        M = cv2.getRotationMatrix2D(center, angle, 1)
        im_rot = cv2.warpAffine(src=im,M=M,dsize=(rows,cols),borderValue=(255, 255, 255))
        locs2, desc2 = BRIEF.briefLite(im_rot)
        matches = BRIEF.briefMatch(desc1, desc2)
        count.append(len(matches))
    
    plt.bar(np.arange(36), count)
    plt.xlabel("Rotation angle/10")
    plt.ylabel("No of correct matches")
    plt.show()

