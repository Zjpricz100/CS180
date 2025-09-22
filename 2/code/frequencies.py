import numpy as np  
import utils as ut # File utility functions \
from scipy.signal import convolve2d
import cv2 as cv
import matplotlib.pyplot as plt

from align_image_code import align_images


def convolve2d_with_color(img, kernel):
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        out[..., c] = convolve2d(img[..., c], kernel, mode="same", boundary="symm")
    return out

def apply_unsharp_mask_filter(img, alpha, kernel_size, sigma):
    G = cv.getGaussianKernel(kernel_size, sigma)
    G_kernel = np.outer(G, G.T)
    
    # Unsharp Matrix
    M = np.zeros((kernel_size, kernel_size))
    H, W = M.shape
    M[H // 2][W // 2] = (1 + alpha)

    G_kernel_scaled = alpha * G_kernel


    return convolve2d_with_color(img, (M - G_kernel_scaled))

def sharpen_img(imname, alpha, kernel_size, sigma):
    img = ut.read_in_image(imname)
    sharpened_img = apply_unsharp_mask_filter(img, alpha, kernel_size, sigma)
    ut.write_output(sharpened_img, "taj_sharpened.jpg")


def create_hybrid_img(img1_name, img2_name, kernel_size=25, sigma_1=1, sigma_2=1):
    """
    Creates a hybrid image from low frequency img1 with high frequency img2
    """
    img1 = ut.read_in_image(img1_name)
    img2 = ut.read_in_image(img2_name)


    # Align images converts to [0,1]
    img1_aligned, img2_aligned = align_images(img1, img2)


    hybrid = hybrid_img(img1_aligned, img2_aligned, kernel_size, sigma_1, sigma_2)
    ut.write_output(hybrid, "derekxcat.jpg")


# Creates a hybrid image where img1 will be low passed, img2 will be high passed
def hybrid_img(img1, img2, kernel_size=25, sigma_1=1, sigma_2=1):
    G_lp = cv.getGaussianKernel(kernel_size, sigma_1)
    G_lp = np.outer(G_lp, G_lp.T)
    G_hp = cv.getGaussianKernel(kernel_size, sigma_2)
    G_hp = np.outer(G_hp, G_hp.T)

    img1_low_pass = convolve2d_with_color(img1, G_lp)
    img2_low_pass = convolve2d_with_color(img2, G_hp)
    img2_high_pass = img2 - img2_low_pass

    hybrid = img1_low_pass + img2_high_pass
    hybrid = np.clip(hybrid, 0, 1)
    return hybrid


# 2.1
# sharpen_img("data/taj.jpg", 2, 5, 1)

#img1_name = "data/DerekPicture.jpg"
#img2_name = "data/nutmeg.jpg"
#create_hybrid_img(img1_name, img2_name, 45, 5, 8)
