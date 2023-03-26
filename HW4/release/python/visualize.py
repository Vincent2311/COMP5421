'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper
from mpl_toolkits import mplot3d
intrinsics = np.load('../data/intrinsics.npz')
data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

pts1 = data['pts1']
pts2 = data['pts2']
M = max(im1.shape[0],im1.shape[1])
F = sub.eightpoint(pts1, pts2, M)
K1 = intrinsics['K1']
K2 = intrinsics['K2']
E = sub.essentialMatrix(F,K1,intrinsics['K2'])

points = np.load('../data/templeCoords.npz')
x1 = points['x1'][:,0]
y1 = points['y1'][:,0]

pts1_selected = []
pts2_selected = []

for i in range(len(x1)):
    x2, y2 = sub.epipolarCorrespondence(im1, im2, F, x1[i], y1[i])
    pts1_selected.append([x1[i],y1[i]])
    pts2_selected.append([x2,y2])

pts1_selected = np.array(pts1_selected)
pts2_selected = np.array(pts2_selected)

M2s = helper.camera2(E)
M1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
C1 = K1 @ M1

best_P = None
best_M2 = None
best_C2 = None
for i in range(M2s.shape[-1]):
    M2 = M2s[:, :, i]
    C2 = K2 @ M2
    P,err = sub.triangulate(C1,pts1_selected,C2,pts2_selected)
    if np.min(P[:, -1]) > 0:
        best_M2 = M2
        best_C2 =C2
        best_P = P

np.savez('../data/q4_2.npz',F=F,M1=M1,M2=best_M2,C1=C1,C2=best_C2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(best_P[:, 0], best_P[:, 1], best_P[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()