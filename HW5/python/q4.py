import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions
    image = skimage.restoration.denoise_bilateral(image,channel_axis=-1)
    gray = skimage.color.rgb2gray(image)
    
    # threshold & morphology
    thresh = skimage.filters.threshold_otsu(gray)
    bw = gray < thresh
    bw = skimage.morphology.closing(bw)

    # label, 0-valued pixels are considered as background pixels
    label_image = skimage.morphology.label(bw, connectivity=2)
    props = skimage.measure.regionprops(label_image)
    mean_area = sum([x.area for x in props])/len(props)
    bboxes = [x.bbox for x in props if x.area > mean_area/2]

    bw = (~bw).astype(np.float)
    return bboxes, bw