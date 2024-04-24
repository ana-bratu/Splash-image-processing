# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:20:53 2024

@author: Ana
"""


path = r'd:\Users\Ana\Desktop\Wonderful PhD\cardioid\TOP_20032024_003_biomimetc60deg_5000fps_h103cm_d2mm_centered_C002H001S0001.png'

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Function to handle mouse clicks
def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
        cv2.imshow('image', img)

# Load the image
img = cv2.imread(path)

# Resize the image for better viewing
scale_percent = 300  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
# Create a window and bind the mouse click event
cv2.imshow('image', img)
points = []
cv2.setMouseCallback('image', click_event)

# Wait until the 'q' key is pressed
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Plot the clicked points on the image
for point in points:
    cv2.circle(img, point, 1, (0, 0, 255), -1)

# Convert BGR image to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Plot the image with points using matplotlib
plt.imshow(img_rgb)
for point in points:
    plt.scatter(point[0], point[1], c='r', marker='o', s=10)
plt.show()

# Convert points to numpy array
points = np.array(points)

# Plot the points on a graph
plt.figure()
plt.scatter(points[:, 0], points[:, 1], c='r', marker='o', label='Points selected from image')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Points Selected from Image')
plt.grid(True)
plt.legend()
plt.show()

# Define the cardioid equation
def cardioid(theta, a):
    return a * (1 - np.cos(theta))

# Initial guess for the scaling factor
x_center, y_center = 22+np.mean(points[:, 0]), np.mean(points[:, 1])
initial_guess = np.mean(np.linalg.norm(points - [x_center, y_center], axis=1))

# Fit the points to the cardioid equation with the refined initial guess
angles = np.arctan2(points[:, 1] - y_center, points[:, 0] - x_center)
distances = np.linalg.norm(points - [x_center, y_center], axis=1)
params, _ = curve_fit(cardioid, angles, distances, p0=[initial_guess])

# Generate points on the cardioid curve
theta_values = np.linspace(0, 2*np.pi, 100)
r_values = cardioid(theta_values, *params)
x_values = r_values * np.cos(theta_values) + x_center
y_values = r_values * np.sin(theta_values) + y_center

# Plot the points on a graph along with the fitted cardioid curve
plt.figure()
plt.scatter(points[:, 0], points[:, 1], c='r', marker='o', label='Points selected from image')
plt.plot(x_values, y_values, c='b', label='Fitted Cardioid Curve')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Points Selected from Image and Fitted Cardioid Curve')
plt.grid(True)


# Plot lines between consecutive points
for i in range(len(points) - 1):
    plt.plot([points[i][0], points[i+1][0]], [points[i][1], points[i+1][1]], c='k')

# Plot line between the last and the first points to close the loop
plt.plot([points[-1][0], points[0][0]], [points[-1][1], points[0][1]], c='k')

plt.legend()
plt.axis('equal')
plt.show()

cv2.destroyAllWindows()

