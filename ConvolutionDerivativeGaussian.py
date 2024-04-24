# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:51:18 2024

@author: Ana
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

def convolve_with_derivative_of_gaussian(image, kernel_size=15, sigma=3):
    """
    Convolve the input image along each column with the derivative of Gaussian kernel.
    
    Args:
        image (numpy.ndarray): Input image.
        kernel_size (int): Size of the kernel for Gaussian blur.
        sigma (float): Standard deviation for Gaussian blur.

    Returns:
        numpy.ndarray: Image convolved with the derivative of Gaussian along each column.
    """
    # Convert the image to grayscale if it's RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create the 1D Gaussian kernel
    gaussian_kernel = cv2.getGaussianKernel(kernel_size, sigma)

    # Calculate the derivative of Gaussian kernel
    gaussian_derivative_kernel = np.gradient(gaussian_kernel[:, 0])
    
    # gaussian_derivative_kernel = gaussian_kernel[:, 0]


    # Apply the 1D convolution along each column with the derivative of Gaussian kernel
    derivative_image = np.zeros_like(image, dtype=np.float32)
    for col in range(image.shape[1]):
        column_signal = image[:, col].astype(np.float32)  # Extract column signal
        derivative_column = np.convolve(column_signal, gaussian_derivative_kernel, mode='same')  # Perform 1D convolution
        derivative_image[:, col] = derivative_column

    # Convert the image to uint8 type
    # derivative_image = derivative_image.astype(np.uint8)

    return derivative_image

# Function to apply Sobel operator along y-axis
def sobel_y(image):
    """
    Apply Sobel operator along the y-axis.
    
    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Image after applying Sobel operator along y-axis.
    """
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = np.abs(sobel_y)
    sobel_y = np.uint8(sobel_y)
    return sobel_y



def resample_image(image, new_shape):
    """
    Resamples an image by interpolation using spline fits.

    Parameters:
        image (ndarray): The input image as a 2D numpy array.
        new_shape (tuple): The new shape of the resampled image (height, width).

    Returns:
        ndarray: The resampled image.
    """
    # Original image dimensions
    height, width = image.shape

    # Create grid for original image
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)

    # Create grid for resampled image
    new_width, new_height = new_shape
    new_x = np.linspace(0, width - 1, new_width)
    new_y = np.linspace(0, height - 1, new_height)

    # Create spline interpolation object
    spline = RectBivariateSpline(y, x, image)

    # Perform interpolation
    resampled_image = spline(new_y, new_x)

    return resampled_image


# Example usage:
import imageio.v3 as iio
image_path = r'd:\Users\Ana\Desktop\Wonderful PhD\image_analisys\compute circularity\final functions to extract data from imapct videos\grayscale.png'
image = iio.imread(image_path)
image = image[:,:,0]


# Resample image to double size
new_shape = (image.shape[0], image.shape[1]//4)
resampled_image = resample_image(image, new_shape)
# resampled_image=image
# Apply convolution with derivative of Gaussian
derivative_image = convolve_with_derivative_of_gaussian(resampled_image, kernel_size=15, sigma=4)

# Apply Sobel operator along y-axis
sobel_image_y = sobel_y(derivative_image)

# Plot the images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(derivative_image, cmap='gray')
plt.title('Convolution with a Derivative of Gaussian')
plt.axis('off')

# Plot the images
plt.figure(figsize=(10, 5))

plt.imshow(sobel_image_y, cmap='gray')
plt.title('Sobel Y Image')
plt.axis('off')

plt.tight_layout()
plt.show()



gray = derivative_image
# Threshold for edge detection
threshold = 100

# Store edge points
edge_points = []

# Iterate through columns
for x in range(gray.shape[1]):
    # Iterate through rows
    for y in range(gray.shape[0]):
        # Check if pixel value exceeds threshold
        if gray[y, x] <-5:
            # Store coordinates of edge pixel
            edge_points.append((x, y))
            break
    # Break the loop once the edge is found (optional)


# Convert edge points to numpy array for plotting
edge_points = np.array(edge_points)

# Plot the grayscale image
plt.imshow(resampled_image, cmap='gray')

# Plot the edge points
plt.scatter(edge_points[:, 0], edge_points[:, 1], c='lime', s=0.1)



from scipy.interpolate import CubicSpline

x,y = edge_points[:, 0], edge_points[:, 1]

spl = CubicSpline(x, y)
X = np.linspace(0, len(y), 100)
Y = spl(X)



plt.plot(X,Y) 




# very smooth...maybe too smooth?
from scipy.interpolate import UnivariateSpline
spline = UnivariateSpline(x, y, s=0.1)
X1 = np.linspace(0, len(y), 100)
Y1 = spline(X)
plt.plot(X1, Y1)
plt.show()

# Plot the images
plt.figure(figsize=(10, 5))
plt.scatter(x,y)
plt.plot(X,Y, label = "piecewise polynomial")
plt.plot(X1, Y1, label = "UnivariateSpline")

spline = UnivariateSpline(x, y, s=1)
X1 = np.linspace(0, len(y), 100)
Y1 = spline(X)
plt.plot(X1, Y1)

spline = UnivariateSpline(x, y, s=10)
X1 = np.linspace(0, len(y), 100)
Y1 = spline(X)
plt.plot(X1, Y1)

spline = UnivariateSpline(x, y, s=100)
X1 = np.linspace(0, len(y), 100)
Y1 = spline(X)
plt.plot(X1, Y1)

spline = UnivariateSpline(x, y, s=1000)
X1 = np.linspace(0, len(y), 100)
Y1 = spline(X)
plt.plot(X1, Y1)

spline = UnivariateSpline(x, y, s=10000)
X1 = np.linspace(0, len(y), 100)
Y1 = spline(X)
plt.plot(X1, Y1)
plt.legend(["data points","piecewise polynomial","UnivariateSpline s=0.1","UnivariateSpline s=1","UnivariateSpline s=10","UnivariateSpline s=100","UnivariateSpline s=1000","UnivariateSpline s=10000"])
plt.show()

