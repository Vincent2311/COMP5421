import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper
data = np.load('../data/some_corresp.npz')
intrinsics = np.load('../data/intrinsics.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

N = data['pts1'].shape[0]
M = 640

# 2.1
F8 = sub.eightpoint(data['pts1'], data['pts2'], M)
assert F8.shape == (3, 3), 'eightpoint returns 3x3 matrix'
# helper.displayEpipolarF(im1, im2, F)

# 2.2
pts1 = np.array([[256,270],[162,152],[199,127],[147,131],[381,236],[193,290],[157,231]])
pts2 = np.array([[257,266],[161,151],[197,135],[146,133],[380,215],[194,284],[157,211]])
F7 = sub.sevenpoint(pts1, pts2, M)
# for F in F7:
#     print(F)
#     helper.displayEpipolarF(im1, im2, F)

# 3.1
E = sub.essentialMatrix(F8,intrinsics['K1'],intrinsics['K2'])
print(E)

# 3.2
