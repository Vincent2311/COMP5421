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
	train_data = np.load("../data/train_data.npz",allow_pickle=True)
	dictionary = np.load("dictionary.npy",allow_pickle=True)
	# ----- TODO -----
	SPM_layer_num = 2
	image_names = train_data['image_names']
	arglist = []
	for i, image_path in enumerate(image_names):
		arglist.append(('../data/'+image_path[0],dictionary,SPM_layer_num + 1,dictionary.shape[0]))
	pool = multiprocessing.Pool(num_workers)
	features = pool.starmap(get_image_feature,arglist)
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


	test_data = np.load("../data/test_data.npz",allow_pickle=True)
	trained_system = np.load("trained_system.npz",allow_pickle=True)
	# ----- TODO -----
	dictionary = trained_system['dictionary']
	SPM_layer_num = trained_system['SPM_layer_num']

	image_names = test_data['image_names']
	SPM_layer_num = 2
	arglist = []
	for i, image_path in enumerate(image_names):
		arglist.append(('../data/'+image_path[0],dictionary,SPM_layer_num + 1,dictionary.shape[0]))
	pool = multiprocessing.Pool(num_workers)
	test_features = pool.starmap(get_image_feature,arglist)
	test_labels = test_data['labels']
	
	predicted_labels = []
	for feature in test_features:
		predicted_feature = np.argmax(distance_to_set(feature, trained_system['features']))
		predicted_label = (trained_system['labels'])[predicted_feature]
		predicted_labels = np.append(predicted_labels,predicted_label)
	predicted_labels.flatten()
	conf = np.zeros((8,8))
	for idx, label in enumerate(test_labels):
		conf[label,int(predicted_labels[idx])] += 1
		if label != predicted_labels[idx]:
			print("wrongly predicted: ",test_data['image_names'][idx], label, predicted_labels[idx])
	
	accuracy = np.diag(conf).sum()/conf.sum()
	return conf, accuracy


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
	H,W = wordmap.shape
	patch_num = 2**(layer_num-1)
	patch_H = H//patch_num
	patch_W = W//patch_num
	weight = 1/2
	hist_all = []

	patch_hist_all = np.empty((patch_num,patch_num,dict_size))
	for idx in range(patch_num*patch_num):
		row = idx//patch_num
		col = idx % patch_num
		patch = wordmap[row*patch_H:(row+1)*patch_H,col*patch_W:(col+1)*patch_W]
		patch_hist = get_feature_from_wordmap(patch, dict_size)
		patch_hist_all[row,col,:] = patch_hist
	
	hist_all = np.append(hist_all,(patch_hist_all*weight).flatten())

	pre_hist = patch_hist_all
	
	for layer in range(layer_num-2, -1, -1):
		patch_num = patch_num//2
		weight = weight/(2 if layer!=0 else 1)
		
		layer_hist = np.empty((patch_num,patch_num,dict_size))
		for idx in range(patch_num*patch_num):
			row = idx//patch_num
			col = idx % patch_num
			layer_hist[row,col,:] = np.sum(pre_hist[row*2:(row+1)*2,col*2:(col+1)*2,:],axis=(0,1))
		
		hist_all = np.append(hist_all,(layer_hist*weight).flatten())
		pre_hist = layer_hist

	hist_all = hist_all/np.sum(hist_all) 
	return hist_all







	

