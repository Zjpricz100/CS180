import numpy as np  
import utils as ut # File utility functions \
from scipy.signal import convolve2d

# Kernels
D_x = np.array([[1, 0, -1]])
D_y = np.array([[1],
                [0],
                [-1]])
box_filter = np.empty((9, 9))
box_filter.fill(1 / 81)

def convolve2d_two_loops(image, kernel):
    kernel = np.flip(np.flip(kernel, axis=1), axis=0)
    kH, kW = kernel.shape
    H, W = image.shape

    out_H, out_W = H - kH - 1, W - kW - 1
    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            patch = image[i : i + kH, j : j + kW]
            output[i, j] = np.sum(patch * kernel)
    return output

def convolve2d_four_loops(image, kernel):
    kernel = np.flip(np.flip(kernel, axis=1), axis=0)
    kH, kW = kernel.shape
    H, W = image.shape

    out_H, out_W = H - kH - 1, W - kW - 1
    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            out_val = 0
            for m in range(kH):
                for n in range(kW):
                    out_val += (kernel[m][n] * image[i + m][j + n])
            output[i][j] = out_val
    return output

def create_edge_image(img, threshold):
    img_Dx = convolve2d(img, D_x, mode='same')
    img_Dy = convolve2d(img, D_y, mode='same')

    edge_img = np.sqrt((img_Dx ** 2) + (img_Dy ** 2))
    edge_img[edge_img >= threshold] = 1
    edge_img[edge_img < threshold] = 0
    return edge_img






img = ut.read_in_image("data/willem.jpg", gray=True)

    
# Part 1.1: Convolution From Scratch!
def run_one_point_one():

    img_box_filtered = convolve2d(img, box_filter)
    ut.write_output(img_box_filtered, "zach_box_scipy.jpg")

    img_Dx = convolve2d_two_loops(img, D_x)
    img_Dy = convolve2d_two_loops(img, D_y)
    ut.write_output(img_Dx, "zach_Dx.jpg")
    ut.write_output(img_Dy, "zach_Dy.jpg")

# Part 1.2: Finite Difference Operator
def run_one_point_two():
    edge_img = create_edge_image(img, threshold=1e-1)
    ut.write_output(edge_img, "willem_edge_img.jpg")




run_one_point_two()








#ut.write_output(img, "zach.jpg")

