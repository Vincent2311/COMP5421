import numpy as np
import threading
import queue
import imageio
import os,time
import math
import visual_words
import skimage.io
import multiprocessing

def build_recognition_system(num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,M)
	* labels: numpy.ndarray of shape (N)
	* dictionary: numpy.ndarray of shape (K,3F)
	* SPM_layer_num: number of spatial pyramid layers
	'''
	train_data = np.load("../data/train_data.npz")
	dictionary = np.load("dictionary.npy")
	# ----- TODO -----
	## check layer_num
	SPM_layer_num = 2
	image_names = train_data['image_names']
	labels = train_data['labels']
	arglist = []
	for i, image_path in enumerate(image_names):
		arglist.append(('../data/'+image_path[0],dictionary,SPM_layer_num + 1,dictionary.shape[0]))
	pool = multiprocessing.Pool(num_workers)
	features = pool.map(get_image_feature,arglist)
	np.savez('trained_system.npz', features = features, labels = train_data['labels'], dictionary=dictionary, SPM_layer_num = SPM_layer_num)


def evaluate_recognition_system(num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''


	test_data = np.load("../data/test_data.npz")
	trained_system = np.load("trained_system.npz")
	# ----- TODO -----
	pass




def get_image_feature(file_path,dictionary,layer_num,K):
	'''
	Extracts the spatial pyramid matching feature.

	[input]
	* file_path: path of image file to read
	* dictionary: numpy.ndarray of shape (K,3F)
	* layer_num: number of spatial pyramid layers
	* K: number of clusters for the word maps

	[output]
	* feature: numpy.ndarray of shape (K)
	'''
	# ----- TODO -----
	image = skimage.io.imread(file_path)
	wordmap = visual_words.get_visual_words(image, dictionary)
	features = get_feature_from_wordmap_SPM(wordmap, layer_num, K)
	return features


def distance_to_set(word_hist,histograms):
	'''
	Compute similarity between a histogram of visual words with all training image histograms.

	[input]
	* word_hist: numpy.ndarray of shape (K)
	* histograms: numpy.ndarray of shape (N,K)

	[output]
	* sim: numpy.ndarray of shape (N)
	'''
	# ----- TODO -----
	sim = np.sum(np.minimum(word_hist,histograms),axis=1)
	return sim


def get_feature_from_wordmap(wordmap,dict_size):
	'''
	Compute histogram of visual words.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* dict_size: dictionary size K

	[output]
	* hist: numpy.ndarray of shape (K)
	'''
	
	# ----- TODO -----
	hist, _= np.histogram(wordmap, dict_size)
	hist = hist/np.sum(hist)
	return hist 



def get_feature_from_wordmap_SPM(wordmap,layer_num,dict_size):
	'''
	Compute histogram of visual words using spatial pyramid matching.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* layer_num: number of spatial pyramid layers
	* dict_size: dictionary size K

	[output]
	* hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
	'''
	
	# ----- TODO -----
	# need efficiency improvement
	hist_all = []
	for layer in range(layer_num):
		patch_H = (wordmap.shape[0] /2**layer).astype(int)
		patch_W = (wordmap.shape[1] /2**layer).astype(int)
		x = 0
		hist_patch = []
		for i in range(2**layer):
			y = 0
			for j in range(2**layer):
				patch = wordmap[x:x+patch_H,y:y+patch_W]
				hist = get_feature_from_wordmap(patch,dict_size)
				hist_patch = np.append(hist_patch,hist)
				y = y+patch_W
			x = x + patch_H 
		if layer <= 1:
			weight = float(2)**(-layer_num)
		else:
			weight = float(2)**(layer - layer_num - 1)
		hist_patch = hist_patch / np.sum(hist_patch)
		hist_patch = hist_patch * weight
		hist_all = np.append(hist_all,hist_patch)
	hist_all = hist_all/np.sum(hist_all)
	return hist_all
		







	

