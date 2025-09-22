import stack
import numpy as np  
import utils as ut # File utility functions \
from scipy.signal import convolve2d
import cv2 as cv
import matplotlib.pyplot as plt

from align_image_code import align_images
from frequencies import convolve2d_with_color
from stack import *

# 2.3
img1_name = "data/spline/apple.jpeg"
img2_name = "data/spline/orange.jpeg"

img1 = ut.read_in_image(img1_name)
img2 = ut.read_in_image(img2_name)

def create_vertical_mask(img):
    H, W, C = img.shape

    # use linspace to arrange mask 
    mask = np.zeros((H, W, C))
    mask[:, :W // 2, :] = 1
    return mask

def multires_blend(img1, img2, mask, levels=4):

    # Create Laplacian Stacks for Each Image
    stack_1 = build_laplacian_stack(img1, levels=levels)
    stack_2 = build_laplacian_stack(img2, levels=levels)
    mask_stack = build_gaussian_stack(mask, levels=levels, sigma_0=5)
    ones = np.ones(mask.shape)


    blend_stack = []
    for i in range(levels):
        mask_i = mask_stack[i]
        l_i = stack_1[i] * mask_i + stack_2[i] * (ones - mask_i)
        blend_stack.append(l_i)

    # Reconstruct image
    final_img = np.zeros(mask.shape)
    for l_i in blend_stack:

        final_img += l_i
    
    # Normalize Final image
    display = ut.min_max_normalize_image(final_img)

    return display

mask = create_vertical_mask(img1)
blended_img = multires_blend(img1, img2, mask)
ut.write_output(blended_img, "output/oraple.jpeg")


