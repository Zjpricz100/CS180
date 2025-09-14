import os
import skimage.io as skio
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray

def write_output(image, imname):
    os.makedirs("./output", exist_ok=True)

    base_name = os.path.splitext(os.path.basename(imname))[0]
    out_path = os.path.join("output", f"{base_name}.jpg")
    plt.imsave(out_path, image, cmap="gray")

    print("Saved to ", out_path)

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