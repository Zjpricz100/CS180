import numpy as np  
import utils as ut # File utility functions \
from scipy.signal import convolve2d
import cv2 as cv

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

# Creates the smoothed image using derivative of gaussian kernel
def create_edge_image_derivative(img, kernel_size=25, sigma=1, threshold=1e-1):
    G = cv.getGaussianKernel(kernel_size, sigma)
    G_kernel = np.outer(G, G.T)
    DoG_x = convolve2d(G_kernel, D_x)
    DoG_y = convolve2d(G_kernel, D_y)

    # Now per image we just need one convolution per direction
    img_Dx = convolve2d(img, DoG_x, mode='same')
    img_Dy = convolve2d(img, DoG_y, mode='same')

    edge_img = np.sqrt((img_Dx ** 2) + (img_Dy ** 2))
    edge_img[edge_img >= threshold] = 1
    edge_img[edge_img < threshold] = 0

    return edge_img


def create_edge_image_smoothed(img, kernel_size=25, sigma=1, threshold=1e-1):
    G = cv.getGaussianKernel(kernel_size, sigma)
    G_kernel = np.outer(G, G.T)
    img_smoothed = convolve2d(img, G_kernel)
    return create_edge_image(img_smoothed, threshold)

    
# Part 1.1: Convolution From Scratch!
def run_one_point_one():

    img = ut.read_in_image("data/zach.jpg", gray=True)


    img_box_filtered = convolve2d(img, box_filter)
    ut.write_output(img_box_filtered, "zach_box_scipy.jpg")

    img_Dx = convolve2d_two_loops(img, D_x)
    img_Dy = convolve2d_two_loops(img, D_y)
    ut.write_output(img_Dx, "zach_Dx.jpg")
    ut.write_output(img_Dy, "zach_Dy.jpg")

# Part 1.2: Finite Difference Operator
def run_one_point_two():
    threshold=1.25e-1
    img = ut.read_in_image("data/cameraman.png", gray=True)
    edge_img = create_edge_image(img, threshold=threshold)
    ut.write_output(edge_img, "cameraman_edge_img.png")



def run_one_point_three():
    threshold=1.25e-1
    sigma = 2
    img = ut.read_in_image("data/cameraman.png", gray=True)
    edge_img_improved = create_edge_image_smoothed(img, sigma=sigma, threshold=threshold)
    ut.write_output(edge_img_improved, f"cameraman_edge_img_smoothed.png")

    # Comparing kernels. Verifying it is the same if we convolve D_x and D_y prior 
    edge_img_DoG = create_edge_image_derivative(img, sigma=sigma, threshold=threshold)
    ut.write_output(edge_img_DoG, f"cameraman_edge_img_smoothed_DoG.png")










#ut.write_output(img, "zach.jpg")

