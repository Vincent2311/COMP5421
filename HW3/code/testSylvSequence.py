import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import LucasKanade
from LucasKanadeBasis import LucasKanadeBasis
import cv2

# write your script here, we recommend the above libraries for making your animation
sequence = np.load('../data/sylvseq.npy',allow_pickle=True)
ori_rect = np.array([101, 61, 155, 107])[...,None].astype(np.float32)
basis_rect = np.array([101, 61, 155, 107])[...,None].astype(np.float32)
sylvseqrects = np.empty((sequence.shape[2],4))
sylvseqrects[0] = basis_rect.T.flatten()
bases = np.load("../data/sylvbases.npy",allow_pickle=True)

It = sequence[:,:,0]

for i in range(1, sequence.shape[2]):
    It1 = sequence[:,:,i]
    p1 =  LucasKanade(It,It1,ori_rect)
    ori_rect = [ori_rect[0]+p1[0], ori_rect[1]+p1[1], ori_rect[2]+p1[0], ori_rect[3]+p1[1]]
    p2 = LucasKanadeBasis(It,It1,basis_rect,bases)
    basis_rect = [basis_rect[0]+p2[0], basis_rect[1]+p2[1], basis_rect[2]+p2[0], basis_rect[3]+p2[1]]

    
    if i==1 or i == 200 or i==300 or i ==350 or i == 400:
        img = np.dstack((It1,It1,It1)).copy()
        cv2.rectangle(img, (np.round(ori_rect[0][0]).astype(int), np.round(ori_rect[1][0]).astype(int)), (np.round(ori_rect[2][0]).astype(int), np.round(ori_rect[3][0]).astype(int)), color=(0,255,255), thickness=2)
        cv2.rectangle(img, (np.round(basis_rect[0][0]).astype(int), np.round(basis_rect[1][0]).astype(int)), (np.round(basis_rect[2][0]).astype(int), np.round(basis_rect[3][0]).astype(int)), color=(0,255,0), thickness=2)
        cv2.imwrite('q2-3_{}.jpg'.format(i), img*255)
    
    sylvseqrects[i] = np.array(basis_rect).T.flatten()
    It = sequence[:,:,i]

np.save('sylvseqrects.npy', sylvseqrects)