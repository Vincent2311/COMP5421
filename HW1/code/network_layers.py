import numpy as np
import scipy.ndimage
import os,time
import skimage

def extract_deep_feature(x,vgg16_weights):
	'''
	Extracts deep features from the given VGG-16 weights.

	[input]
	* x: numpy.ndarray of shape (H,W,3)
	* vgg16_weights: numpy.ndarray of shape (L,3)

	[output]
	* feat: numpy.ndarray of shape (K)
	'''
	if np.amax(x) > 1:
		x = x/255
	mean=[0.485,0.456,0.406]
	std=[0.229,0.224,0.225]
	x = skimage.transform.resize(x,(224,224,3))
	x = (x - mean)/std
	feat = None
	linear_count = 0
	for layer in vgg16_weights:
		if layer[0] == 'conv2d':
			feat = multichannel_conv2d(x,layer[1],layer[2])
		elif layer[0] == 'relu':
			feat = relu(x)
			if linear_count == 2:
				break
		elif layer[0] == 'maxpool2d':
			feat = max_pool2d(x,layer[1])
		elif layer[0] == 'linear':
			if len(x.shape) > 1:
				x = x.transpose(2, 0, 1)
				x = x.ravel()
			feat = linear(x,layer[1],layer[2])
			linear_count = linear_count + 1
		x = feat
	
	return feat


def multichannel_conv2d(x,weight,bias):
	'''
	Performs multi-channel 2D convolution.

	[input]
	* x: numpy.ndarray of shape (H,W,input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim,kernel_size,kernel_size)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* feat: numpy.ndarray of shape (H,W,output_dim)
	'''
	H,W,input_dim=x.shape
	output_dim,_,_,_ = weight.shape
	feat = np.empty((H,W,output_dim))
	for i in range(output_dim):
		kernel = weight[i,:,:,:]
		channel_feat = np.zeros((H,W))
		for j in range(input_dim):
			channel_feat = channel_feat + scipy.ndimage.convolve(x[:,:,j],kernel[j,::-1,::-1],mode='constant',cval=0)
		feat[:,:,i] = channel_feat + bias[i]
	return feat

def relu(x):
	'''
	Rectified linear unit.

	[input]
	* x: numpy.ndarray

	[output]
	* y: numpy.ndarray
	'''
	y = np.maximum(0, x)
	return y

def max_pool2d(x,size):
	'''
	2D max pooling operation.

	[input]
	* x: numpy.ndarray of shape (H,W,input_dim)
	* size: pooling receptive field

	[output]
	* y: numpy.ndarray of shape (H/size,W/size,input_dim)
	'''
	H,W,input_dim = x.shape
	patch_H = H//size
	patch_W = W//size
	patch = x[:patch_H*size,:patch_W*size,:]
	patch = patch.reshape(patch_W,size,patch_H,size,input_dim)
	y = np.amax(patch,axis=(1,3))
	return y

def linear(x,W,b):
	'''
	Fully-connected layer.

	[input]
	* x: numpy.ndarray of shape (input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* y: numpy.ndarray of shape (output_dim)
	'''
	y = W @ x + b
	return y

