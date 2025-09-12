# CS180 (CS280A): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
from pyramid import gaussian_downsample, build_gaussian_pyramid
import os
import math

def read_in_image(imname, plot=False):
    im = skio.imread(imname)
    if plot:
        plt.imshow(im)
        plt.show()
    
    # Convert to float for saving memory
    im = sk.img_as_float(im)

    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(int)

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    # Return in BGR format
    return np.array([b, g, r])


# Returns tuple of (aligned img, optim_dx, optim_dy)
def align(dx_VAL, crop_factor = 0.1, level=0, pyramid=None, ref_pyramid=None, current_dx=0, current_dy=0):


    dx_VALS = range(-dx_VAL, dx_VAL + 1, 1)

    img_level = pyramid[level]
    ref_level = ref_pyramid[level]
    ref_cropped = crop_center(ref_level, crop_factor)
    v2 = ref_cropped.flatten() - np.mean(ref_cropped)



    # Multiply our displacements by 2 since we upsampled by 2 compared to previous call
    current_dx *= 2
    current_dy *= 2


    max_L = 0
    optim_dx = current_dx
    optim_dy = current_dy
    optim_img = img_level

    for dx in dx_VALS:
        for dy in dx_VALS:
            dx_total = dx + current_dx
            dy_total = dy + current_dy

            img_shifted  = np.roll(img_level, shift=dy_total, axis=0)
            img_shifted = np.roll(img_shifted, shift=dx_total, axis=1)

            img_cropped = crop_center(img_shifted, crop_factor)

            # Compute metric
            v1 = img_cropped.flatten() - np.mean(img_cropped)
            L = np.dot((v1 / np.linalg.norm(v1)), (v2 / np.linalg.norm(v2)))

            if L > max_L:
                max_L = L
                optim_dx = dx_total
                optim_dy = dy_total
                optim_img = img_shifted

    if level + 1 < len(pyramid):
        return align(dx_VAL//2, crop_factor, level + 1, pyramid, ref_pyramid, optim_dx, optim_dy)
    else:
        return (optim_img, optim_dx, optim_dy)
    
    
def crop_center(img, crop_factor=0.1):
    border_x = int(crop_factor * img.shape[0])
    border_y = int(crop_factor * img.shape[1])
    return img[border_x : -border_x, border_y : -border_y]



def align_images(img_seperated, dx_VAL, crop_factor=0.1):
    # Green is our reference image
    ref_img = img_seperated[1]

    ref_img = crop_center(ref_img, crop_factor)
    blue = crop_center(img_seperated[0], crop_factor)
    red = crop_center(img_seperated[2], crop_factor)


    blue_pyramid = list(reversed(build_gaussian_pyramid(blue)))
    red_pyramid = list(reversed(build_gaussian_pyramid(red)))
    ref_pyramid = list(reversed(build_gaussian_pyramid(ref_img)))

    # Align blue to reference
    aligned_blue, blue_dx, blue_dy = align(dx_VAL, crop_factor=crop_factor, level=0, pyramid=blue_pyramid, ref_pyramid=ref_pyramid)


    # Align red to reference
    aligned_red, red_dx, red_dy = align(dx_VAL, crop_factor=crop_factor, level=0, pyramid=red_pyramid, ref_pyramid=ref_pyramid)

    # Stack images
    im_out = np.dstack([aligned_red, ref_img, aligned_blue])

    # Logging info
    print("(x, y) Displacement to align Blue to Green: ", (blue_dx, blue_dy))
    print("(x, y) Displacement to align Red to Green: ", (red_dx, red_dy))

    return im_out

max_displacement = 30
crop_factor = 0.1

def colorize_all_images():
    data_dir = "data"
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.tif', '.jpg'))]



    n = len(files)
    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes = axes.flatten()

    for ax, fname in zip(axes, files):
        img = read_in_image(imname=os.path.join(data_dir, fname), plot=False)
        optimal = align_images(img, dx_VAL=max_displacement, crop_factor=crop_factor)
        im_out = crop_center(optimal, crop_factor=crop_factor)
        ax.imshow(im_out)
        ax.set_title(fname, fontsize=12)
        ax.axis("off")

    for ax in axes[len(files):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def colorize_single_image(imname):
    img = read_in_image(imname)
    optimal = align_images(img, dx_VAL=max_displacement, crop_factor=crop_factor)
    im_out = crop_center(optimal, crop_factor=crop_factor)

    os.makedirs("output", exist_ok=True)

    base_name = os.path.splitext(os.path.basename(imname))[0]
    out_path = os.path.join("output", f"{base_name}.jpg")

    plt.imsave(out_path, im_out)
    plt.imshow(im_out)
    plt.title(imname)
    plt.show()

def colorize_all_images_in_data():
    data_dir = "data"
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.tif', '.jpg'))]

    for fname in files:
        full_path = os.path.join(data_dir, fname)
        print(f"Processing: {fname}")
        colorize_single_image(full_path)

# Example usage
#colorize_all_images_in_data()

colorize_single_image('data/personal/bird.tif')
colorize_single_image('data/personal/ship.tif')
colorize_single_image('data/personal/camel.tif')