import numpy as np
import multiprocessing
import threading
import queue
import imageio
import os,time
import torch
import skimage.transform
import torchvision.transforms
import util
import network_layers
import scipy

def build_recognition_system(vgg16,num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,K)
	* labels: numpy.ndarray of shape (N)
	'''


	train_data = np.load("../data/train_data.npz",allow_pickle=True)
	# ----- TODO -----
	image_names = train_data['image_names']
	arglist = []

	for i, image_path in enumerate(image_names):
		arglist.append((i,'../data/'+image_path[0],vgg16))
	
	for param in vgg16.parameters():
		param.requires_grad = False

	for arg in arglist:
		get_image_feature(arg)
	tmp_dir = '../vgg_tmp'

	features = [None]*len(train_data['image_names'])
	for file in os.listdir(tmp_dir):
		feature = np.load(os.path.join(tmp_dir,file))
		index = int(file.split('.')[0])
		features[index] = feature.flatten()
	features = np.array(features)

	np.savez('trained_system_deep.npz', features=features, labels=train_data['labels'])


def evaluate_recognition_system(vgg16,num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''

	
	test_data = np.load("../data/test_data.npz",allow_pickle=True)
	# ----- TODO -----
	trained_system = np.load("trained_system_deep.npz",allow_pickle=True)

	image_names = test_data['image_names']

	test_features = []
	for i, image_path in enumerate(image_names):
		image_path = '../data/' + image_path[0]
		image = skimage.io.imread(image_path)
		image = preprocess_image(image)
		image = torch.unsqueeze(image, 0)
		feat = vgg16(image.double()).detach()
		feat = torch.squeeze(feat,0)
		test_features.append(feat)
		print("tested feature: ",i)

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


def preprocess_image(image):
	'''
	Preprocesses the image to load into the prebuilt network.

	[input]
	* image: numpy.ndarray of shape (H,W,3)

	[output]
	* image_processed: torch.Tensor of shape (3,H,W)
	'''
	# ----- TODO -----
	image_shape = image.shape
	if len(image_shape) == 2:
		image = np.matlib.repmat(image, 3, 1)
	else:
		if image_shape[-1] == 4:
			image = image[:, :, :-1]
	
	if np.amax(image) > 1:
		image = image/255
	
	mean=[0.485,0.456,0.406]
	std=[0.229,0.224,0.225]
	image = skimage.transform.resize(image,(224,224,3))
	image = (image - mean)/std
	image = np.transpose(image, (2,0,1))
	image = torch.from_numpy(image)
	return image


def get_image_feature(args):
	'''
	Extracts deep features from the prebuilt VGG-16 network.
	This is a function run by a subprocess.
 	[input]
	* i: index of training image
	* image_path: path of image file
	* vgg16: prebuilt VGG-16 network.
	* time_start: time stamp of start time
 	[saved]
	* feat: evaluated deep feature
	'''
	i,image_path,vgg16 = args

	# ----- TODO -----
	image = skimage.io.imread(image_path)
	image = preprocess_image(image)
	image = torch.unsqueeze(image, 0)
	feat = vgg16(image.double()).detach()
	tmp_dir = '../vgg_tmp-1'
	if not os.path.exists(tmp_dir):
		os.makedirs(tmp_dir)
	np.save(os.path.join(tmp_dir, str(i) + '.npy'), feat) 
	print("saved one tmp file: ",i)




def distance_to_set(feature,train_features):
	'''
	Compute distance between a deep feature with all training image deep features.

	[input]
	* feature: numpy.ndarray of shape (K)
	* train_features: numpy.ndarray of shape (N,K)

	[output]
	* dist: numpy.ndarray of shape (N)
	'''

	# ----- TODO -----
	
	N,K = train_features.shape
	dist = -np.linalg.norm(feature - train_features,axis=1).reshape(N,)
	return dist