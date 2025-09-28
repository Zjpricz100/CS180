import numpy as np  
import utils as ut # File utility functions \
from scipy.signal import convolve2d
import cv2 as cv
import matplotlib.pyplot as plt

from align_image_code import align_images

# Reading in Images
taj_img = ut.read_in_image("data/taj.jpg")
derek_img = ut.read_in_image("data/DerekPicture.jpg")
nutmeg_img = ut.read_in_image("data/nutmeg_rotated.png")
mononoke_img = ut.read_in_image("data/mononoke.jpg")
eboshi_img = ut.read_in_image("data/eboshi.jpg")
walter_img = ut.read_in_image("data/walter.jpg")
heisenburg_img = ut.read_in_image("data/heisenburg_2.jpg")
mononoke_eboshi_hybrid_img = ut.read_in_image("output/part_2/mononokexeboshi2.jpg")

lizard_img = ut.read_in_image("data/lizard_2.jpg")



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
    sharpened_img = convolve2d_with_color(img, (M - G_kernel_scaled))
    sharpened_img_clipped = np.clip(sharpened_img, 0, 1)
    return sharpened_img_clipped
    
# Hybrid Images

def low_pass_img(img, kernel_size=25, sigma=1):
    G_lp = cv.getGaussianKernel(kernel_size, sigma)
    G_lp = np.outer(G_lp, G_lp.T)
    img_low_pass = convolve2d_with_color(img, G_lp)
    return img_low_pass

def high_pass_img(img, kernel_size=25, sigma=1):
    img_low_pass = low_pass_img(img, kernel_size, sigma)
    return img - img_low_pass
    


# Creates a hybrid image where img1 will be low passed, img2 will be high passed
def hybrid_img(img1, img2, kernel_size=25, sigma_1=1, sigma_2=1, visualize=False):
    
    # Align images
    img1, img2 = align_images(img1, img2)

    G_lp = cv.getGaussianKernel(kernel_size, sigma_1)
    G_lp = np.outer(G_lp, G_lp.T)
    G_hp = cv.getGaussianKernel(kernel_size, sigma_2)
    G_hp = np.outer(G_hp, G_hp.T)

    img1_low_pass = convolve2d_with_color(img1, G_lp) 
    img2_low_pass = convolve2d_with_color(img2, G_hp) 

    img2_high_pass = img2 - img2_low_pass

    if visualize:
        img2_high_pass_display = ut.min_max_normalize_image(img2_high_pass)
        img1_low_pass_display = np.clip(img1_low_pass, 0, 1) 
        img2_high_pass_display = np.clip(img2_high_pass_display, 0, 1) 
        ut.write_output(img1_low_pass_display, "post_low_pass.jpg")
        ut.write_output(img2_high_pass_display, "post_high_pass.jpg")

    hybrid = img1_low_pass + img2_high_pass
    hybrid = np.clip(hybrid, 0, 1)
    return hybrid

# fft magnitude spectrum for color images
def compute_fft_magnitude_spectrum(img):
    channels = []
    for c in range(img.shape[2]):
        freq = np.log(np.abs(np.fft.fftshift(np.fft.fft2(img[..., c]))))
        freq = ut.min_max_normalize_image(freq)
        channels.append(freq)
    return np.dstack(channels) 

def test_sharpen_img():
    alphas = [0.5, 0.75, 1, 2, 2.5]
    for idx, alpha_i in enumerate(alphas):
        sharpen_i = apply_unsharp_mask_filter(taj_img, alpha_i, kernel_size=15, sigma=1)
        ut.write_output(sharpen_i, f"taj_sharpen_alpha_{idx}")

def test_hybrid_img(img1, img2, outpath, sigma_1=1, sigma_2=1, crop_params=None, visualize=False):
    hybrid = hybrid_img(img1, img2, kernel_size=25, sigma_1=sigma_1, sigma_2=sigma_2, visualize=True)

    if crop_params is not None:
        hybrid = ut.crop(hybrid, crop_params[0], crop_params[1], crop_params[2], crop_params[3])
    ut.write_output(hybrid, outpath)

def test_hybrid_frequencies(img1, img2, hybrid_img, outpaths):

    img1_low_pass = low_pass_img(img1, kernel_size=25, sigma=5)
    img2_high_pass = img2 - low_pass_img(img2, kernel_size=25, sigma=3.5)

    img1_freq = compute_fft_magnitude_spectrum(img1)
    img2_freq = compute_fft_magnitude_spectrum(img2)
    hybrid_freq = compute_fft_magnitude_spectrum(hybrid_img)

    img1_low_pass_freq = compute_fft_magnitude_spectrum(img1_low_pass)
    img2_high_pass_freq = compute_fft_magnitude_spectrum(img2_high_pass)

    print(img1_freq.shape)


    ut.write_output(img1_freq, outpaths[0])
    ut.write_output(img2_freq, outpaths[1])
    ut.write_output(hybrid_freq, outpaths[2])
    ut.write_output(img1_low_pass_freq, outpaths[3])
    ut.write_output(img2_high_pass_freq, outpaths[4])


# Sharpening the Taj
#test_sharpen_img()

# Derek and Nutmeg
#test_hybrid_img(derek_img, nutmeg_img, "derekxnutmeg.jpg", 4, 3)

# Mononoke and Eboshi
#test_hybrid_img(mononoke_img, eboshi_img, "mononokexeboshi2.jpg", 4, 3.5, crop_params=[0.1, 0, 0.2, 0], visualize=True)

# Walter and Heisenburg
#test_hybrid_img(heisenburg_img, walter_img, "walterxheisenburg.png", 5, 3.5)

# Frequency Components
#outpaths = ["mononoke_freq.jpg", "eboshi_freq.jpg", "mononokexeboshi_freq.jpg", "mononoke_low_pass_freq.jpg", "eboshi_high_pass_freq.jpg"]
#test_hybrid_frequencies(mononoke_img, eboshi_img, mononoke_eboshi_hybrid_img, outpaths)



