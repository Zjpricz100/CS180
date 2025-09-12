import numpy as np
from scipy.ndimage import convolve
from skimage.transform import resize

def gaussian_downsample(img):
    return resize(img, (img.shape[0] // 2, img.shape[1] // 2))

def build_gaussian_pyramid(img, course_size = 128):

    pyramid = [img]
    current_shape = img.shape
    current_img = img

    #print("Initial Shape: ", current_shape)
    while (current_shape[0] >= course_size and current_shape[1] >= course_size):
        current_img = gaussian_downsample(current_img)
        current_shape = current_img.shape
        pyramid.append(current_img)
    return pyramid







    