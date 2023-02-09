import numpy as np
def alignChannels(red, green, blue):
    """Given 3 images corresponding to different channels of a color image,
    compute the best aligned result with minimum abberations

    Args:
      red, green, blue - each is a HxW matrix corresponding to an HxW image

    Returns:
      rgb_output - HxWx3 color image output, aligned as desired"""
    height, width= red.shape
    r_g_h, r_g_w = compare(red, green)
    r_b_h, r_b_w = compare(red, blue)
    blank_image = np.ones((height+60,width+60,3), np.uint8)
    blank_image[30:height+30, 30:width+30, 0] = red
    blank_image[30+r_g_h:height+30+r_g_h, 30+r_g_w:width+30+r_g_w, 1] = green
    blank_image[30+r_b_h:height+30+r_b_h, 30+r_b_w:width+30+r_b_w, 2] = blue
    return blank_image

def compare(x_old, y_old):
    height, width= x_old.shape
    dist = 1000000
    h_trans = 0
    w_trans = 0
    for h in range(-30, 31):
        for w in range(-30, 31):
            if(h<=0 & w<=0):
                x_new = x_old[0:(height - abs(h)), 0:(width - abs(w))]
                y_new = y_old[abs(h):height, abs(w):width]
            elif (h <= 0 & w > 0):
                x_new = x_old[0:(height - abs(h)), w:width]
                y_new = y_old[abs(h):height, 0:(width - w)]
            elif (h > 0 & w <= 0):
                x_new = x_old[h:height, 0:(width - abs(w))]
                y_new = y_old[0:(height - h), abs(w):width]
            else:
                x_new = x_old[h:height, w:width]
                y_new = y_old[0:(height - h), 0:(width - w)]
            dist1 = np.sqrt(np.sum((x_new-y_new)**2))
            if(dist1<dist):
                h_trans = h
                w_trans = w
                dist = dist1
    return h_trans, w_trans





