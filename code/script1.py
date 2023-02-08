# Problem 1: Image Alignment
import numpy as np
from alignChannels import alignChannels
from PIL import Image
import os
# 1. Load images (all 3 channels)
red = np.load('../data/red.npy')
green = np.load('../data/green.npy')
blue = np.load('../data/blue.npy')

height, width= red.shape

# 2. Find best alignment
output= alignChannels(red, green, blue)

rgbResult = Image.fromarray(output, 'RGB')
if not os.path.exists("../results"):
    os.makedirs("../results")
rgbResult.save("../results/rgb_output.jpg")