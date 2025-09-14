import numpy as np  
import utils as ut # File utility functions \
from scipy.signal import convolve2d
import cv2 as cv

def convolve2d_with_color(img, kernel):
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        out[..., c] = convolve2d(img[..., c], kernel, mode="same", boundary="symm")
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)

def apply_unsharp_mask_filter(img, alpha, kernel_size, sigma):
    G = cv.getGaussianKernel(kernel_size, sigma)
    G_kernel = np.outer(G, G.T)
    
    # Unsharp Matrix
    M = np.zeros((kernel_size, kernel_size))
    H, W = M.shape
    M[H // 2][W // 2] = (1 + alpha)

    G_kernel_scaled = alpha * G_kernel


    return convolve2d_with_color(img, (M - G_kernel_scaled))

img = ut.read_in_image("data/taj.jpg")
sharpened_img = apply_unsharp_mask_filter(img, 2, 5, 1)
ut.write_output(sharpened_img, "taj_sharpened.jpg")