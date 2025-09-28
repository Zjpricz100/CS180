import numpy as np  
import utils as ut # File utility functions \
from scipy.signal import convolve2d
import cv2 as cv

# Library to compare runtimes for convolution implementations
import timeit

# Kernels
D_x = np.array([[1, 0, -1]])
D_y = np.array([[1],
                [0],
                [-1]])
box_filter = np.empty((9, 9))
box_filter.fill(1 / 81)

# Reading in Images
zach_img = ut.read_in_image("data/zach.jpg", gray=True)
camera_img = ut.read_in_image("data/cameraman.png", gray=True)


# Convolution Implementations. Passing padding as 0 is "valid" convolution
def convolve2d_two_loops(image, kernel, padding=0):
    kernel = np.flip(np.flip(kernel, axis=1), axis=0)
    kH, kW = kernel.shape
    H, W = image.shape

    # Padding
    padded = np.pad(image, pad_width=padding, mode="constant", constant_values=0)
    pH, pW = padded.shape
    out_H, out_W = pH - kH - 1, W - kW - 1
    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            patch = padded[i : i + kH, j : j + kW]
            output[i, j] = np.sum(patch * kernel)
    return output

def convolve2d_four_loops(image, kernel, padding=0):
    kernel = np.flip(np.flip(kernel, axis=1), axis=0)
    kH, kW = kernel.shape
    H, W = image.shape

    # Padding
    padded = np.pad(image, pad_width=padding, mode="constant", constant_values=0)
    pH, pW = padded.shape
    out_H, out_W = pH - kH - 1, W - kW - 1
    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            out_val = 0
            for m in range(kH):
                for n in range(kW):
                    out_val += (kernel[m][n] * image[i + m][j + n])
            output[i][j] = out_val
    return output

# Timing Functions for Convolution
def time_two_loops():
    return convolve2d_two_loops(image=zach_img, kernel=box_filter, padding=0)

def time_four_loops():
    return convolve2d_four_loops(image=zach_img, kernel=box_filter, padding=0)

def time_scipy_convolve():
    return convolve2d(zach_img, box_filter, mode="valid", boundary="fill", fillvalue=0)

def compare_convolve_runtimes():
    print("Comparing Runtimes Across Convolution Implementations Across 5 Experiments:")

    # Averaging out runtimes over 5 experiments and comparing
    print("Two loops:", timeit.timeit(time_two_loops, number=5))
    print("Four loops:", timeit.timeit(time_four_loops, number=5))
    print("Scipy:", timeit.timeit(time_scipy_convolve, number=5))


# Edge images and Derivative Filters
def create_edge_image(img, threshold):
    img_Dx = convolve2d(img, D_x, mode='same')
    img_Dy = convolve2d(img, D_y, mode='same')

    edge_img = np.sqrt((img_Dx ** 2) + (img_Dy ** 2))
    edge_img[edge_img >= threshold] = 1
    edge_img[edge_img < threshold] = 0
    return edge_img

# Creates the smoothed image using derivative of gaussian kernel
def create_edge_image_derivative(img, kernel_size=25, sigma=1, threshold=1e-1, visualize=False):
    G = cv.getGaussianKernel(kernel_size, sigma)
    G_kernel = np.outer(G, G.T)
    DoG_x = convolve2d(G_kernel, D_x)
    DoG_y = convolve2d(G_kernel, D_y)

    if visualize:
        ut.write_output(DoG_x, "DoG_x.jpg")
        ut.write_output(DoG_y, "DoG_y.jpg")

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
    img_smoothed = convolve2d(img, G_kernel, mode='same')
    return create_edge_image(img_smoothed, threshold) / 255.0
    
def test_convolution():
    zach_img_low_pass = convolve2d(zach_img, box_filter)
    ut.write_output(zach_img_low_pass, "zach_low_pass.jpg")

def test_finite_difference_operator():
    cameraman_dx = convolve2d(camera_img, D_x)
    cameraman_dy = convolve2d(camera_img, D_y)
    ut.write_output(cameraman_dx, "cameraman_dx.jpg")
    ut.write_output(cameraman_dy, "cameraman_dy.jpg")

def test_edge_image():
    img_Dx = convolve2d(camera_img, D_x, mode='same')
    img_Dy = convolve2d(camera_img, D_y, mode='same')

    edge_img = np.sqrt((img_Dx ** 2) + (img_Dy ** 2))
    ut.write_output(edge_img, "cameraman_gradient_magnitude.jpg")

def test_binary_edge_image():
    threshold = 3e-1
    edge_img = create_edge_image(camera_img, threshold=threshold)
    ut.write_output(edge_img, "cameraman_edge_img.png")

def test_binary_edge_image_denoised():
    threshold = 1.25e-1
    sigma = 1.3
    edge_img_improved = create_edge_image_smoothed(camera_img, sigma=sigma, threshold=threshold)
    ut.write_output(edge_img_improved, f"cameraman_edge_img_smoothed.png")

    # Comparing kernels. Verifying it is the same if we convolve D_x and D_y prior.
    edge_img_DoG = create_edge_image_derivative(camera_img, sigma=sigma, threshold=threshold)
    ut.write_output(edge_img_DoG, f"cameraman_edge_img_smoothed_DoG.png")


    # Comparing different values of sigma
    sigmas = [0.5, 0.75, 1, 2, 2.5]
    for idx, sigma_i in enumerate(sigmas):
        edge_img_i = create_edge_image_smoothed(camera_img, sigma=sigma_i, threshold=threshold)
        ut.write_output(edge_img_i, f"cameraman_edge_img_smoothed_sigma={idx}")

# Test convolution implementations and runtimes
#compare_convolve_runtimes()

# Test Finite Difference Operators
#test_finite_difference_operator()

# Test Naive Edge Image
#test_binary_edge_image()

# Test Denoised/Derivative of Gaussian Edge Image
#test_binary_edge_image_denoised()


