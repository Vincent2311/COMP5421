'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper
intrinsics = np.load('../data/intrinsics.npz')
data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')

pts1 = data['pts1']
pts2 = data['pts2']
M = max(im1.shape[0],im1.shape[1])
F = sub.eightpoint(pts1, pts2, M)
K1 = intrinsics['K1']
K2 = intrinsics['K2']
E = sub.essentialMatrix(F,K1,intrinsics['K2'])

M2s = helper.camera2(E)


C1 = K1 @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])

best_P = None
best_M2 = None
best_C2 = None
for i in range(M2s.shape[-1]):
    M2 = M2s[:, :, i]
    C2 = K2 @ M2
    P,err = sub.triangulate(C1,pts1,C2,pts2)
    if np.min(P[:, -1]) > 0:
        best_M2 = M2
        best_C2 =C2
        best_P = P
np.savez('../data/q3_3', M2=best_M2, C2=best_C2, P=best_P)
