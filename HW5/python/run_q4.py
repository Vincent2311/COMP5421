import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw, cmap='gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()

    # find the rows using..RANSAC, counting, clustering, etc.
    mean_height = sum([b[2]-b[0] for b in bboxes])/len(bboxes)
    #get the center_x, center_y, width, height 
    centers = [((b[3]+b[1])//2,(b[2]+b[0])//2,b[3]-b[1],b[2]-b[0]) for b in bboxes]
    centers = sorted(centers,key = lambda center: center[1])

    rows = []
    row = []
    current_y = centers[0][1]
    for center in centers:
        if center[1] - current_y > mean_height:
            row = sorted(row,key = lambda center: center[0])
            rows.append(row)
            row = [center]
            current_y = center[1]
        else:
            row.append(center)
    row = sorted(row,key = lambda center: center[0])
    rows.append(row)

    
    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    data = []
    for row in rows:
        line = []
        for x,y,width,height in row:
            cropped = bw[y-height//2:y+height//2, x-width//2:x+width//2]
            if height > width:
                padding=((10,10),((height-width)//2 + 10,(height-width)//2 + 10))
            else:
                padding=(((width-height)//2 + width//10,(width-height)//2 + 10),(10,10))
            cropped = np.pad(cropped,padding,mode='constant',constant_values=(1, 1))
            cropped = skimage.transform.resize(cropped, (32, 32))
            cropped = skimage.morphology.erosion(cropped)
            line.append(cropped.T.flatten())
        data.append(line)

    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    
    for line in data:
        h1 = forward(line,params,'layer1')
        probs = forward(h1,params,'output',softmax)
        pred_idx = np.argmax(probs, axis = -1)
        strings = ''
        for idx in pred_idx:
            strings += letters[idx]
        print(strings)
