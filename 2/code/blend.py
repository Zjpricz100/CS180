import stack
import numpy as np  
import utils as ut # File utility functions \
from scipy.signal import convolve2d
import cv2 as cv
import matplotlib.pyplot as plt

from align_image_code import align_images
from frequencies import convolve2d_with_color
from stack import *
import skimage.io as skio

# --- Mask Creation Functions ---
def create_vertical_mask(img):
    H, W, C = img.shape

    # use linspace to arrange mask 
    mask = np.zeros((H, W, C))
    mask[:, :W // 2, :] = 1
    return mask
# 2.3
img1_name = "data/spline/apple.jpeg"
img2_name = "data/spline/orange.jpeg"

img1 = ut.read_in_image(img1_name)
img2 = ut.read_in_image(img2_name)

sun_img = ut.read_in_image("data/sun.jpg")
black_hole_img = ut.read_in_image("data/black_hole.jpg")
road_img = ut.read_in_image("data/road.jpg")
night_sky_img = ut.read_in_image("data/night_sky.jpg")

black_hole_mask = ut.read_in_image("data/masks/black_hole_mask.jpg")[:, :, 0]
apple_mask = create_vertical_mask(img1)[:, :, 0]
road_mask = ut.read_in_image("data/masks/road_mask.jpg")[:, :, 0]




def multires_blend(img1, img2, mask, levels=4, sigma_blend=4, visualize=False):
    display_levels = range(levels)
    if visualize:
        levels = 5
        display_levels = [0, 2, 4]

        # Create a plot with 3 rows (for levels) and 3 columns
        fig, axs = plt.subplots(4, 3, figsize=(9, 9))
        plt.tight_layout()

    # Create Laplacian Stacks for Each Image
    stack_1 = build_laplacian_stack(img1, levels=levels, sigma_0=1)
    stack_2 = build_laplacian_stack(img2, levels=levels, sigma_0=1)
    mask_stack = build_gaussian_stack(mask, levels=levels, sigma_0=sigma_blend)
    ones = np.ones(mask.shape)


    blend_stack = []
    for i in range(levels):
        mask_i = mask_stack[i]

        laplacian_1 = stack_1[i]
        laplacian_2 = stack_2[i]

        weighted_1 = laplacian_1 * mask_i
        weighted_2 = laplacian_2 * (ones - mask_i)

        l_i = weighted_1 + weighted_2

        blend_stack.append(l_i)

        if visualize and i in display_levels:
            idx = i // 2
            axs[idx, 0].imshow(ut.min_max_normalize_image(weighted_1))
            axs[idx, 0].set_title(f"Apple, Level {i}")
            axs[idx, 0].axis('off')

            axs[idx, 1].imshow(ut.min_max_normalize_image(weighted_2))
            axs[idx, 1].set_title(f"Orange, Level {i}")
            axs[idx, 1].axis('off')

            axs[idx, 2].imshow(ut.min_max_normalize_image(l_i))
            axs[idx, 2].set_title(f"Blended, Level {i}")
            axs[idx, 2].axis('off')


    # Reconstruct image
    final_img = np.zeros(mask.shape)
    for l_i in blend_stack:
        final_img += l_i

    final_img = np.clip(final_img, 0, 1)


        
    
    # Normalize Final image
    display = ut.min_max_normalize_image(final_img)
    if visualize:
        axs[3, 0].imshow(img1 * mask_stack[-1])
        axs[3, 0].axis('off')
        
        
        axs[3, 1].imshow(img2 * (1 - mask_stack[-1]))
        axs[3, 1].axis('off')

        axs[3, 2].imshow(display)
        axs[3, 2].axis('off')
        plt.show()

    return display

def test_multires_blend(img1, img2, mask, outpath, levels=4, sigma_blend=4, visualize=False):

    print(img1.shape, img2.shape, mask.shape)

    # Normalize mask
    mask = mask / 255.0 if mask.max() > 1.0 else mask
    mask_3ch = np.stack([mask, mask, mask], axis=-1)

    # Ensure the image is also in the correct float format
    img1 = img1 / 255.0 if img1.max() > 1.0 else img1
    img2 = img2 / 255.0 if img2.max() > 1.0 else img2

    blended_img = multires_blend(img1, img2, mask_3ch, levels, sigma_blend, visualize)
    ut.write_output(blended_img, outpath)

def test_blend_images():
    test_multires_blend(black_hole_img, sun_img, black_hole_mask, "black_hole_sun_2.jpg", 5, 8)
    test_multires_blend(img1, img2, apple_mask, "oraple_final.jpg", levels=3, sigma_blend=6, visualize=True)
    test_multires_blend(night_sky_img, road_img, road_mask, outpath="night_road.jpg", levels=4, sigma_blend=6, visualize=False)

# Run to test all functions for blending (2.3, 2.4)
def __main__():
    test_blend_images()

if __name__ == "__main__":
    __main__()









