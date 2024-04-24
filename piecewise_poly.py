# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:06:53 2024

@author: Ana
"""





# image= np.random.randint(0, 255, (20,20))
# print(image.shape)
# print(image.dtype)

# plt.imshow(image)

# m, n = image.shape

        

# y = image[:, 0]
# x = [*range(0, len(y))]

# spl = CubicSpline(x, y)

# dspl = spl.derivative()
# dspl(1.1), spl(1.1, nu=1)

# dspl.roots() / np.pi

# plt.figure(figsize=(8, 6))
# plt.scatter(x,y)

# X = np.linspace(0, len(y), 100)
# Y = spl(X)

# plt.plot(X, Y)



# # for local maxima
# argrelextrema(Y, np.greater)

# # for local minima
# local_minima = argrelextrema(Y, np.less)



# edge_x = X[local_minima[0][0]]
# edge_y = Y[local_minima[0][0]]
# plt.plot(edge_x, edge_y, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")



# pts = np.array([[330,620],[950,620],[692,450],[587,450]])

# plt.imshow(image)
# plt.plot(640, 570, "og", markersize=10)  # og:shorthand for green circle
# plt.scatter(pts[:, 0], pts[:, 1], marker="x", color="red", s=200)
# plt.show()

# plt.plot(X, Y)

# edge_x = X[local_minima[0][0]]
# edge_y = Y[local_minima[0][0]]
# plt.plot(edge_x, edge_y, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")


# image= np.random.randint(0, 255, (20,20))
# print(image.shape)
# print(image.dtype)

# plt.imshow(image)

# m, n = image.shape
# plt.figure(figsize=(8, 6))
# pos_edge = []
# for i in range(m):
#     y = image[:, i]
#     x = [*range(0, len(y))]
#     spl = CubicSpline(x, y)
#     X = np.linspace(0, len(y), 100)
#     Y = spl(X)
#     # for local minima
#     local_minima = argrelextrema(Y, np.less)
#     first_local_minima = local_minima[0][0]
#     pos_edge.append(X[first_local_minima])

# plt.figure(figsize=(8, 6))
# plt.imshow(image)
# x = [*range(0, m)]
# plt.plot(x, pos_edge, "or", markersize=5)  # og:shorthand for green circle
# plt.show()
    
    
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import argrelextrema
import imageio.v3 as iio
from scipy.ndimage import uniform_filter


def subtract_blur(image, kernel_size):
    blurred_image = uniform_filter(image, size=kernel_size)
    subtracted_image = image - blurred_image
    return blurred_image


def plot_image_with_local_minima(image):
    # plt.figure(figsize=(8, 6))
    # plt.imshow(image)
    
    m, n = image.shape
    pos_edge = []
    for i in range(n):
        y = image[:, i]
        x = np.arange(0, len(y))
        spl = CubicSpline(x, y)
        X = np.linspace(0, len(y), 100)
        Y = spl(X)
        # for local minima
        local_minima = argrelextrema(Y, np.less)
        first_local_minima = local_minima[0][0]
        pos_edge.append(X[first_local_minima])

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    x = np.arange(0, n)
    plt.plot(x, pos_edge, "or", markersize=5)  # red circles
    plt.show()

# Example usage:
image_path = r'd:\Users\Ana\Desktop\Wonderful PhD\image_analisys\compute circularity\final functions to extract data from imapct videos\grayscale.png'
image = iio.imread(image_path)
image = image[:,:,0]
# image = image[50:200,:]


# Specify kernel size for blurring (e.g., 3x3)
kernel_size = (30,1)

# Subtract blur
result_image = subtract_blur(image, kernel_size)
result_image = subtract_blur(result_image, kernel_size)
# result_image = subtract_blur(result_image, kernel_size)


plot_image_with_local_minima(result_image)

# plot_image_with_local_minima(image)