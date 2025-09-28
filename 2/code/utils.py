import os
import skimage.io as skio
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage import img_as_float

def write_output(image, imname, outpath="final_output"):
    os.makedirs(outpath, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(imname))[0]
    out_path = os.path.join(outpath, f"{base_name}.jpg")
    plt.imsave(out_path, image, cmap='gray')

    print("Saved to ", out_path)


def min_max_normalize_image(image):
    display = np.zeros(image.shape)

    if image.ndim == 3:
        for c in range(image.shape[2]):
            ch = image[..., c]
            display[..., c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
    else:
        display = (image - image.min()) / (image.max() - image.min() + 1e-8)

    return display

def crop(img, top_percent=0.0, bottom_percent=0.0, left_percent=0.0, right_percent=0.0):
    H, W = img.shape[:2]

    # Convert percents into pixel counts (rounded to int)
    top = int(H * top_percent)
    bottom = int(H * bottom_percent)
    left = int(W * left_percent)
    right = int(W * right_percent)

    return img[top:H - bottom if bottom > 0 else H,
               left:W - right if right > 0 else W]


def read_in_image(imname, gray=False, plot=False):
    im = skio.imread(imname)
    im = img_as_float(im)
    im = im[:, :, :3]
    if gray and im.ndim == 3:
        im = rgb2gray(im)
    if plot:
        plt.imshow(im)
        plt.show()
    return np.array(im)