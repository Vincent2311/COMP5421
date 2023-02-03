import numpy as np
def alignChannels(red, green, blue):
    """Given 3 images corresponding to different channels of a color image,
    compute the best aligned result with minimum abberations

    Args:
      red, green, blue - each is a HxW matrix corresponding to an HxW image

    Returns:
      rgb_output - HxWx3 color image output, aligned as desired"""
    diff_RG = np.sum((red[:,:]-green[:,:])**2)
    diff_RB = np.sum((red[:,:]-blue[:,:])**2)
    
    return np.dstack((red,green,blue))








