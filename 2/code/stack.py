# Implementation of Gaussian and Laplacian Stacks #

import numpy as np  
import utils as ut # File utility functions \
from scipy.signal import convolve2d
import cv2 as cv
import matplotlib.pyplot as plt

from align_image_code import align_images
from frequencies import convolve2d_with_color

def build_gaussian_stack(img, levels = 5, sigma_0 = 1.0, kernel_0 = 7):
    stack = []
    current_sigma = sigma_0
    current_kernel_size = kernel_0
    for i in range(levels):
        G = cv.getGaussianKernel(current_kernel_size, current_sigma)
        G_kernel = np.outer(G, G.T)

        G_i = convolve2d_with_color(img, G_kernel)
        stack.append(G_i)

        current_sigma *= 2

        # Standard Kernel size growth
        current_kernel_size = int(2 * np.ceil(3 * current_sigma) + 1)
        print(f"Level {i} Done For Gaussian Stack.")
    return stack

def build_laplacian_stack(img, levels=5, sigma_0 = 1.0, kernel_0 = 7):
    G_stack = build_gaussian_stack(img, levels, sigma_0, kernel_0)
    print("Gaussian Stack Complete, Building Laplacian Stack")
    L_stack = []
    for i in range(levels - 1):
        # Extract the band of frequencies from this level and the next (next level has lower frequencies)
        L_i = G_stack[i] - G_stack[i + 1]
        L_stack.append(L_i)
    
    # Append the very lowest frequencies
    L_stack.append(G_stack[-1])
    print("Laplacian Stack Complete.")
    return L_stack


def show_stack(stack):
    n = len(stack)
    plt.figure(figsize=(3*n, 3))
    for i, img in enumerate(stack):
        plt.subplot(1, n, i+1)
        
        if i < n - 1:
            disp = ut.min_max_normalize_image(img)
        else:
            disp = img / np.max(img)
        
        plt.imshow(disp)
        plt.axis("off")
        plt.title(f"Level {i}")
    plt.show()
    
def test_stacks():
    img1_name = "data/spline/apple.jpeg"
    img2_name = "data/spline/orange.jpeg"
    img1 = ut.read_in_image(img1_name)
    img2 = ut.read_in_image(img2_name)
    apple_stack = build_gaussian_stack(img1, levels=4, sigma_0=6, kernel_0=7)
    orange_stack = build_gaussian_stack(img2, levels=4, sigma_0=6, kernel_0=7)
    show_stack(apple_stack)
    show_stack(orange_stack)
    apple_stack_laplacian = build_laplacian_stack(img1, levels=6, sigma_0=6, kernel_0=7)
    orange_stack_laplacian = build_laplacian_stack(img2, levels=6, sigma_0=6, kernel_0=7)
    show_stack(apple_stack_laplacian)
    show_stack(orange_stack_laplacian)

# Run to test all functions for stacks (2.3, 2.4)
def __main__():
    test_stacks()

if __name__ == "__main__":
    __main__()










