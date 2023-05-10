import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import os
import cv2

import SubtractDominantMotion

# write your script here, we recommend the above libraries for making your animation
sequence = np.load('../data/aerialseq.npy')


for i in range(1,31):
    It = sequence[:,:,i-1]
    It1 = sequence[:,:,i]
    mask = SubtractDominantMotion.SubtractDominantMotion(It, It1)

    tmp_img = np.zeros((It1.shape[0], It1.shape[1], 3))
    tmp_img[:, :, 0] = It1
    tmp_img[:, :, 1] = It1
    tmp_img[:, :, 2] = It1
    tmp_img[:, :, 0][mask==1] = 1

    if i==30 or i == 60 or i==90 or i ==120:
        cv2.imwrite('q3-3_{}.jpg'.format(i), tmp_img*255)

    It = It1
    print("round ",i)