import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import LucasKanade
import cv2

# write your script here, we recommend the above libraries for making your animation
sequence = np.load('../data/carseq.npy',allow_pickle=True)
rect = np.array([59, 116, 145, 151])[...,None].astype(np.float32)
carseqrects = np.empty((sequence.shape[2],4))
carseqrects[0] = rect.T.flatten()

pre_p = None
epsilon = 0.5
ori_rects = np.load('carseqrects.npy')

It = sequence[:,:,0]

for i in range(1, sequence.shape[2]):
    It1 = sequence[:,:,i]
    p =  LucasKanade(It,It1,rect)
    new_rect = [rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]]
    
    if i==1 or i == 100 or i==200 or i ==300 or i == 400:
        img = np.dstack((It1,It1,It1)).copy()
        cv2.rectangle(img, (np.round(new_rect[0][0]).astype(int), np.round(new_rect[1][0]).astype(int)), (np.round(new_rect[2][0]).astype(int), np.round(new_rect[3][0]).astype(int)), color=(0,255,255), thickness=2)
        cv2.rectangle(img, (np.round(ori_rects[i][0]).astype(int), np.round(ori_rects[i][1]).astype(int)), (np.round(ori_rects[i][2]).astype(int), np.round(ori_rects[i][3]).astype(int)), color=(0,255,0), thickness=2)
        cv2.imwrite('q1-4_{}.jpg'.format(i), img*255)
    
    if pre_p is None or np.sqrt(np.sum((p-pre_p)**2)) < epsilon :
        It = sequence[:,:,i]
        rect = new_rect
        pre_p = np.empty_like(p)
        pre_p[:] = p
    carseqrects[i] = np.array(rect).T.flatten()
    
        
rect_list = np.array(carseqrects)
np.save('carseqrects-wcrt.npy', carseqrects)