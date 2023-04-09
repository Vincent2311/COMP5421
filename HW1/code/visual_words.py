import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import matplotlib.pyplot as plt
import util
import random

def extract_filter_responses(image):
	'''
	Extracts the filter responses for the given image.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	[output]
	* filter_responses: numpy.ndarray of shape (H,W,3F)
	'''
	
	# ----- TODO -----
	image_shape = image.shape
	if len(image_shape) == 2:
		image = np.matlib.repmat(image, 3, 1)
	else:
		if image_shape[-1] == 4:
			image = image[:, :, :-1]
	image = skimage.color.rgb2lab(image)
	scales = [1, 2, 4, 8, 8*np.sqrt(2)]
	filter_responses = np.empty([image.shape[0], image.shape[1], 0])
	LoG = np.empty([image.shape[0], image.shape[1], 3])

	for scale in scales:
		gaussian = scipy.ndimage.gaussian_filter(image,sigma=(scale,scale,0))
		LoG[:,:,0] = scipy.ndimage.gaussian_laplace(image[:, :, 0], sigma=scale)
		LoG[:,:,1] = scipy.ndimage.gaussian_laplace(image[:, :, 1], sigma=scale)
		LoG[:,:,2] = scipy.ndimage.gaussian_laplace(image[:, :, 2], sigma=scale)
		gauss_x = scipy.ndimage.gaussian_filter(image,sigma=(scale,scale,0),order=(0, 1, 0))
		gauss_y = scipy.ndimage.gaussian_filter(image,sigma=(scale,scale,0),order=(1, 0, 0))
		filter_responses = np.append(filter_responses,np.concatenate([gaussian,LoG,gauss_x,gauss_y],axis=2),axis=2)
	
	return filter_responses






def get_visual_words(image,dictionary):
	'''
	Compute visual words mapping for the given image using the dictionary of visual words.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	
	[output]
	* wordmap: numpy.ndarray of shape (H,W)
	'''
	
	# ----- TODO -----
	filter_resp = extract_filter_responses(image)
	filter_resp = filter_resp.reshape(filter_resp.shape[0]*filter_resp.shape[1], filter_resp.shape[2])

	distances = scipy.spatial.distance.cdist(filter_resp, dictionary)
	distances = distances.reshape(image.shape[0], image.shape[1], distances.shape[1])
	wordmap = np.argmin(distances, axis=2)
	return wordmap



def compute_dictionary_one_image(args):
	'''
	Extracts random samples of the dictionary entries from an image.
	This is a function run by a subprocess.

	[input]
	* i: index of training image
	* alpha: number of random samples
	* image_path: path of image file
	* time_start: time stamp of start time

	[saved]
	* sampled_response: numpy.ndarray of shape (alpha,3F)
	'''

	i,alpha,image_path = args
	# ----- TODO -----
	image = skimage.io.imread(image_path)
	image = image.astype('float')/255
	filter_responses = extract_filter_responses(image)

	index_x = np.random.permutation(image.shape[0])
	index_x = index_x[:alpha]
	index_y = np.random.permutation(image.shape[1])
	index_y = index_y[:alpha]
	filter_responses = filter_responses[index_x,index_y,:]
	tmp_dir = '../tmp'
	if not os.path.exists(tmp_dir):
		os.makedirs(tmp_dir)
	np.save(os.path.join(tmp_dir, 'responce_'+str(i)+'.npy'), filter_responses)


def compute_dictionary(num_workers=2):
	'''
	Creates the dictionary of visual words by clustering using k-means.

	[input]
	* num_workers: number of workers to process in parallel
	
	[saved]
	* dictionary: numpy.ndarray of shape (K,3F)
	'''

	train_data = np.load("../data/train_data.npz",allow_pickle=True)
	# ----- TODO -----
	image_names = train_data['image_names']
	k = 150
	alpha = 50
	arglist = []
	for i, image_path in enumerate(image_names):
		arglist.append((i,alpha,'../data/'+image_path[0]))
	pool = multiprocessing.Pool(num_workers)
	pool.map(compute_dictionary_one_image,arglist)

	responces = np.array([])
	tmp_dir = '../tmp'
	for file in os.listdir(tmp_dir):
		responce = np.load(os.path.join(tmp_dir,file))
		responces = np.append(responces,responce,axis=0) if responces.shape[0]!=0 else responce
	
	kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(responces)
	dictionary = kmeans.cluster_centers_
	np.save('dictionary.npy', dictionary)

