import os
import skimage.io as skio
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray

def write_output(image, imname):
    os.makedirs("./output", exist_ok=True)

    base_name = os.path.splitext(os.path.basename(imname))[0]
    out_path = os.path.join("output", f"{base_name}.jpg")
    plt.imsave(out_path, image)

    print("Saved to ", out_path)


def min_max_normalize_image(image):
    display = np.zeros(image.shape)

    if image.ndim == 3:
        for c in range(image.shape[2]):
            ch = image[..., c]
            display[..., c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
    else:
        print("Not RGB")
    return display



def read_in_image(imname, gray=False, plot=False):
    im = skio.imread(imname)
    im = im[:, :, :3]
    if gray and im.ndim == 3:
        im = rgb2gray(im)
        print(im.shape)
    if plot:
        plt.imshow(im)
        plt.show()
    return np.array(im)