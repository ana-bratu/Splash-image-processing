# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:08:37 2024

@author: Ana
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import copy
import pandas as pd
from skimage import io, img_as_float, exposure
from scipy.interpolate import UnivariateSpline, CubicSpline, RectBivariateSpline
from scipy.ndimage import uniform_filter
from mpl_toolkits.mplot3d import Axes3D
# General system functions
import glob
import os
import shutil
import sys




# Draw a line on one image of the stack starting on the edge of the target and alignet with the trajectpry of the jet Store the coordinates

global length 

def draw_line(event, x, y, flags, param):
    global line_coordinates 
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(line_coordinates) < 2:
            line_coordinates.append((x, y))

            if len(line_coordinates) == 2:
                cv2.line(param, line_coordinates[0], line_coordinates[1], (255, 0, 0), 2)
                cv2.imshow("Draw Line", param)

def extract_line_coord(image):
    # Create a window for image display
    cv2.namedWindow("Draw Line", cv2.WINDOW_NORMAL)
    # Set the callback function for mouse events
    cv2.setMouseCallback("Draw Line", draw_line, param=image)
    # Display the image
    cv2.imshow("Draw Line", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def read_tiff_images_cv2(TopStack, line_coordinates):

    # Initialize an empty list to store all images
    all_images = []
    line =copy.deepcopy(line_coordinates)
    # Read the line coordinates

    reslice_jet = []
    reslice_splash = []
    # Iterate through images
    for file in TopStack[0:15]:
        # Check if the file is a TIFF image
        # if file.lower().endswith(".tiff") or file.lower().endswith(".tif"):
        # file_path = os.path.join(folder_path, file)

        # Read the TIFF image using cv2
        try:
            # image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            
            Gaussian = cv2.GaussianBlur(file, (3, 3), 0) 

            # cv2.imshow('Gaussian Blurring', Gaussian) 
            image = Gaussian
            # plt.figure()
            # plt.imshow(image)
            
            (x1, y1), (x2, y2) = line
            angle_jet = math.atan2(line[1][1] - line[0][1], line[1][0] - line[0][0])
            x2 = x1 + int(length * math.cos(angle_jet))
            y2 = y1 + int(length * math.sin(angle_jet))
         
            # Convert the image data to a list
            image_data = image.tolist()

            # Append the list of pixel values to the main list
            all_images.append(image_data)

            # Extract pixels along the drawn line using Bresenham's algorithm
            line_pixels_jet = []
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx - dy

            while True:
                line_pixels_jet.append(image[y1, x1].tolist())

                if x1 == x2 and y1 == y2:
                    break

                e2 = 2 * err
                if e2 > -dy:
                    err = err - dy
                    x1 = x1 + sx
                if e2 < dx:
                    err = err + dx
                    y1 = y1 + sy

            reslice_jet.append(line_pixels_jet)
            
            (x1, y1) = line[0]
            angle_splash = angle_jet-np.pi/2
            x3 = x1 + int(length * math.cos(angle_splash))
            y3 = y1 + int(length * math.sin(angle_splash))
            
            # Extract pixels along the drawn line using Bresenham's algorithm
            line_pixels_splash = []
            dx = abs(x3 - x1)
            dy = abs(y3 - y1)
            sx = 1 if x1 < x3 else -1
            sy = 1 if y1 < y3 else -1
            err = dx - dy

            while True:
                line_pixels_splash.append(image[y1, x1].tolist())

                if x1 == x3 and y1 == y3:
                    break

                e2 = 2 * err
                if e2 > -dy:
                    err = err - dy
                    x1 = x1 + sx
                if e2 < dx:
                    err = err + dx
                    y1 = y1 + sy

            reslice_splash.append(line_pixels_splash)
        
        except Exception as e:
            print(f"Error processing {file}: {e}")
    # drew_line=[x2, y2, x3, y3]
    reslice = [reslice_jet, reslice_splash]
    
    # print(angle_jet, angle_splash)
    return all_images, reslice

def mouse_callback(event, x, y, flags, param):

    global selected_points, display_image
    to_use_image = copy.deepcopy(display_image)
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append((x, y))
        cv2.circle(to_use_image, (x, y), 1, (0, 0, 255), -1)
        cv2.imshow('Select Points', to_use_image)
        
def average_stack(TopStack, count):

    # Initialize an empty list to store all images
    
    all_images = TopStack[0]
    # Iterate through images
    
    for file in TopStack[1:count]:
        # Check if the file is a TIFF image
        # if file.lower().endswith(".tiff") or file.lower().endswith(".tif"):
        # file_path = os.path.join(folder_path, file)

        # Read the TIFF image using cv2
        all_images = all_images + file 
        all_images = all_images/(count+1)
        # print(np.max(file))
    return all_images

def extract_spline(image):
    global selected_points, display_image
    display_image = copy.deepcopy(image)
    # List to store the selected points
    selected_points = []

    # Create a window and bind the mouse callback function to it
    cv2.namedWindow('Select Points', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Select Points', mouse_callback)

    # Display the original image for point selection
    cv2.imshow('Select Points', display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Convert selected points to numpy array
    selected_points = np.array(selected_points)
    
    xdata, ydata = selected_points[:, 0], selected_points[:, 1]
    ydata = ydata[::-1]
    # Fit a spline to the selected points
    spline = UnivariateSpline(selected_points[:, 0], selected_points[:, 1], s=0.01)



    # Visualize the original image and the fitted spline
    plt.figure(figsize=(8, 6))

    plt.imshow(display_image, cmap='gray')
    plt.plot(selected_points[:, 0], selected_points[:, 1], 'ro', label='Selected Points')
    
  
    ydata = ydata[::-1]
    ydata_edge = []
    line_on_image = []
    for i in range(0, len(xdata)):
        line_on_image = display_image[ydata[i]-3 : ydata[i]+3, xdata[i]]
        print(display_image)
        print(line_on_image)
        minim = min(line_on_image)
        print(minim)
        ydata_new = np.where(line_on_image == minim) + ydata[i]-3
        print(ydata_new[0][0])
        ydata_edge.append(ydata_new[0][0])
    plt.plot(xdata, ydata_edge, 'go', label='real EDGE')



    spline = UnivariateSpline(xdata, ydata_edge, s=0.1)
    # Define the points where you want to evaluate the spline
    x_new = np.linspace(xdata.min(), xdata.max(), 1000)
    y_new = spline(x_new)
    plt.plot(x_new, y_new, 'g', label='Fitted Spline')
    # plt.legend()
    plt.title('Fitting a Spline to an Edge in an Image')
    plt.show()
    return spline, xdata, y_new
    
def get_edge_from_spline(reslice):
    image = np.array(reslice, dtype=np.uint16)
    spline_jet, xdata, y_new = extract_spline(cv2.normalize(image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX))
    
    # Visualize the original image and the fitted spline
    plt.figure(figsize=(8, 6))

    plt.imshow(image, cmap='gray')

  
    y_new = y_new[::-1]
    y_new = [int(y_new[i]) for i in range(0, len(y_new))]
    ydata_edge = []
    line_on_image = []
    m,n =np.shape(image)

    for i in range(0, xdata[-1]):
        line_on_image = image[y_new[i]-3 : y_new[i]+3, i]
        
        print(line_on_image)
        minim = min(line_on_image)
        print(minim)
        ydata_new = np.where(line_on_image == minim) 
        print(ydata_new[0][0])
        ydata_edge.append(ydata_new[0][0] + y_new[i]-3)
        
    xdata_edge = np.array([*range(0, xdata[-1])])
    plt.plot(xdata_edge, ydata_edge, 'go', label='real EDGE')
    
    spline = UnivariateSpline(xdata_edge, ydata_edge, s=0.1)
    # Define the points where you want to evaluate the spline
    x_new = np.linspace(xdata.min(), xdata.max(), 1000)
    y_new = spline(x_new)
    plt.plot(x_new, y_new, 'g', label='Fitted Spline')
    # plt.legend()
    plt.title('Edge semi-automated detection')
    plt.show()
    
def compute_acceleration(reslice):
    image = np.array(reslice, dtype=np.uint16)
    spline_jet, xdata, ydata = extract_spline(cv2.normalize(image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX))
    
    x_new = np.linspace(xdata.min(), xdata.max(), 1000)
    y_new = spline_jet(x_new)
    # Compute the first and second derivatives
    y_prime = spline_jet.derivative(n=1)(x_new)
    y_double_prime = spline_jet.derivative(n=2)(x_new)

    # Plot the original data
    plt.figure(figsize=(8, 6))  # Create a new figure with specified size
    plt.scatter(xdata, ydata, color='red', label='Original Data')

    # Plot the spline
    plt.plot(x_new, y_new, label='Spline')


    plt.figure(figsize=(8, 6))  # Create a new figure with specified size

    # Plot the first derivative
    plt.plot(x_new, y_prime, label='First Derivative')

    # Plot the second derivative
    plt.plot(x_new, y_double_prime, label='Second Derivative')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Spline and Its Derivatives')
    plt.legend()
    plt.grid(True)
    plt.show()

def subtract_blur(image, kernel_size):
    blurred_image = uniform_filter(image, size=kernel_size)
    return blurred_image

def plot_img_and_hist(image, axes, bins=65536):
    """Plot an image along with its histogram and cumulative histogram."""
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


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

    # print("max image=", np.max(image))
    # Apply the 1D convolution along each column with the derivative of Gaussian kernel
    derivative_image = np.zeros_like(image, dtype=np.float64)
    for col in range(image.shape[1]):
        column_signal = image[:, col].astype(np.float64)  # Extract column signal
        derivative_column = np.convolve(column_signal, gaussian_derivative_kernel, mode='same')  # Perform 1D convolution
        derivative_image[:, col] = derivative_column
    # print("max deriv image=", np.max(derivative_image))
    # print("min deriv image=", np.min(derivative_image))
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
    # sobel_y = np.abs(sobel_y)
    # sobel_y = np.uint8(sobel_y)
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

def plot_max_pixel_coords(work_image, plot_image):
    """
    Plot the coordinates of pixels with maximum value in each column of the image.
    
    Args:
    - image: NumPy array representing the grayscale image
    
    Returns:
    None (displays the plot)
    """
    # Find maximum pixel coordinates for each column
    max_pixel_coords = []
    for col in range(work_image.shape[1]):
        max_pixel_row = np.argmax(work_image[:, col])  # Find index of maximum pixel value in the column
        max_pixel_coords.append((col, max_pixel_row))  # Append coordinates to the list

    # Plot the coordinates on the image
    plt.figure()
    plt.imshow(plot_image, cmap='gray')
    plt.scatter(*zip(*max_pixel_coords), color='red', s=5)  # Unzip coordinates and plot as scatter points
    plt.title('Pixels with Maximum Value in Each Column')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.show()
    x_coords, y_coords = zip(*max_pixel_coords)
    return x_coords, y_coords
    
# Specify the folder path containing TIFF images
folder_path = r"d:\Users\Ana\Desktop\experiments\experiments internship\5000fps\experiments\60degrees\H103mm_d2.7mm\SaveData"



# Specify experiment details
scale_pixels = 42.67
scale_mm = 2.7
ideal_diam = 2.7
fps = 5000
length = 40 #maybe find something else
angle = 60
height = 103


SD = pd.read_csv(folder_path + '\\SplashData_Cent_ST.csv',index_col = 'Ind')

StackList = SD.index  
jet_acc = []
splash_acc = []
Circularity = []
for s in StackList[0:1]:

    _, TopStack = cv2.imreadmulti(folder_path + '\\Top_' + s + '.tif', [], -1 )
    # count = 6
    # for file in TopStack: file = average_stack(TopStack, count)
    line_coordinates = []
    extract_line_coord(cv2.normalize(TopStack[20], dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX))
    # print(line_coordinates)
    # print(s)
    

    list_of_lists_cv2, reslice = read_tiff_images_cv2(TopStack, line_coordinates)    
    reslice_jet, reslice_splash = reslice
    
    reslice_jet =  np.array(reslice_jet, dtype=np.uint16)
    
    
    # plt.figure(figsize=(8, 6))  # Create a new figure with specified size

    # plt.imshow(reslice_jet, cmap='gray')  # Display the result
    # plt.title("Reslice Jet")
    # plt.show()
    
    

    img = reslice_jet
    # Resample image to double size
    resample_ratio = 1
    new_shape = (img.shape[0]*resample_ratio, img.shape[1]*resample_ratio)
    resampled_img = resample_image(img, new_shape)
    resampled_img  = img
    ks = 15 #kernel_size
    s = 0.01 #sigma
    derivative_image =convolve_with_derivative_of_gaussian(resampled_img, ks, s)


    # Apply Sobel operator along y-axis
    sobel_image_y = sobel_y(derivative_image)
    
    # Plot the images
    # plt.figure(figsize=(10, 5))
    
    # plt.subplot(3,1, 1)
    # plt.imshow(img, cmap='gray')
    # plt.title('Original Image')
    # plt.axis('off')
    
    # plt.subplot(3,1, 2)
    # plt.imshow(derivative_image, cmap='gray')
    # plt.title('Convolution with a Derivative of Gaussian')
    # plt.axis('off')
    
    # plt.subplot(3,1, 3)
    # plt.imshow(sobel_image_y, cmap='gray')
    # plt.title('Sobel Y Image')
    # plt.axis('off')
    
    # plt.tight_layout()
    # plt.show()


    
    # gray =  sobel_image_y
     
    # # Create meshgrid
    # x = np.arange(0, gray.shape[1], 1)
    # y = np.arange(0, gray.shape[0], 1)
    # x, y = np.meshgrid(x, y)
    
    # # Plot the surface
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # surface = ax.plot_surface(x, y, gray, cmap='gray')
    
    # Show plot
    # plt.show()  
    
    plot_max_pixel_coords(sobel_image_y, reslice_jet)
    
    
    # SPLASH
    img =  np.array(reslice_splash, dtype=np.uint16)
 
    # Resample image to double size
    resample_ratio = 1
    new_shape = (img.shape[0]*resample_ratio, img.shape[1]*resample_ratio)
    resampled_img = resample_image(img, new_shape)
    resampled_img  = img
    ks = 15 #kernel_size
    s = 0.01 #sigma
    derivative_image =convolve_with_derivative_of_gaussian(resampled_img, ks, s)


    # Apply Sobel operator along y-axis
    sobel_image_y = sobel_y(derivative_image)
    x, y  = plot_max_pixel_coords(sobel_image_y, reslice_splash)
    # very smooth...maybe too smooth?
    
    spline = UnivariateSpline(x, y, s=0.1)
    X1 = np.linspace(0, len(y), 100)
    Y1 = spline(X1)
    plt.plot(X1, Y1)
    plt.show()
    
    # Plot the images
    plt.figure(figsize=(10, 5))
    plt.scatter(x,y)
    plt.plot(X1, Y1, label = "UnivariateSpline")
    
    spline = UnivariateSpline(x, y, s=1)
    X1 = np.linspace(0, len(y), 100)
    Y1 = spline(X1)
    plt.plot(X1, Y1)
    
    spline = UnivariateSpline(x, y, s=10)
    X1 = np.linspace(0, len(y), 100)
    Y1 = spline(X1)
    plt.plot(X1, Y1)
    
    spline = UnivariateSpline(x, y, s=100)
    X1 = np.linspace(0, len(y), 100)
    Y1 = spline(X1)
    plt.plot(X1, Y1)
    
    spline = UnivariateSpline(x, y, s=1000)
    X1 = np.linspace(0, len(y), 100)
    Y1 = spline(X1)
    plt.plot(X1, Y1)
    
    spline = UnivariateSpline(x, y, s=10000)
    X1 = np.linspace(0, len(y), 100)
    Y1 = spline(X1)
    plt.plot(X1, Y1)
    plt.legend(["data points","piecewise polynomial","UnivariateSpline s=0.1","UnivariateSpline s=1","UnivariateSpline s=10","UnivariateSpline s=100","UnivariateSpline s=1000","UnivariateSpline s=10000"])
    plt.show()
    
    
    
    
    x = np.array(x)
    y = np.array(y)
    
    # Fit the points with a second-degree polynomial
    coefficients = np.polyfit(x, y, 2)  # Fit a second-degree polynomial (quadratic)
    
    # Generate the polynomial function using the coefficients
    poly_function = np.poly1d(coefficients)
    
    # Generate x values for plotting the polynomial curve
    x_values = np.linspace(min(x), max(x), 100)
    
    # Evaluate the polynomial function at the x values
    y_values = poly_function(x_values)
    
    # Plot the points
    plt.scatter(x, y, label='Data Points')
    
    # Plot the polynomial curve
    plt.plot(x_values, y_values, color='red', label='Polynomial Fit')
    
    # Add labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Second Degree Polynomial Fit')
    plt.legend()
    
    # Show plot
    plt.show()

    # SD.loc[s, 'angle[degrees]'] = angle
    # SD.loc[s, 'IdealDiam[mm]'] = ideal_diam
    # SD.loc[s, 'Height[cm]'] = height
    # SD.loc[s,'Top Velocity jet'] = velocity_jet
    # SD.loc[s,'Top Velocity splash'] = velocity_splash
    # # SD.to_csv(folder_path + '\\SplashData_Cent_ST.csv',index_label = 'Ind')