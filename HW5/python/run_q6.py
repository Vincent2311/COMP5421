import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
import skimage

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

dim = 32
# do PCA on training set
train_mean = np.sum(train_x,axis=0) / train_x.shape[0]
train_x -= train_mean
u, s, vh = np.linalg.svd(train_x)
M = vh[:dim,:]

# rebuild a low-rank version
train_lrank = train_x @ M.T

# rebuild it
train_recon = train_lrank @ M
train_recon += train_mean
train_x += train_mean


# do PCA on validation set
valid_mean = np.sum(valid_x,axis=0) / valid_x.shape[0]
valid_x -= valid_mean

# rebuild a low-rank version
valid_lrank = valid_x @ M.T

# rebuild it
valid_recon = valid_lrank @ M
valid_recon += valid_mean
valid_x += valid_mean


selected_idx = [7, 36, 166, 188, 265, 276, 355, 384, 432, 496]
for i in range(10):
    plt.subplot(2,1,1)
    plt.imshow(valid_x[selected_idx[i]].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(valid_recon[selected_idx[i]].reshape(32,32).T)
    plt.show()

total = []
for pred,gt in zip(valid_recon,valid_x):
    total.append(skimage.metrics.peak_signal_noise_ratio(gt,pred))
print(np.array(total).mean())