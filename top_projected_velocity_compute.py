# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 13:33:16 2024

@author: Ana
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import copy
import pandas as pd
from skimage import io

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


def extract_velocity(line):

    (x1, y1), (x2, y2) = line_coordinates
    
    velocity = -1
    if (x2-x1)==0:
        velocity = 0
    elif (y1-y2) ==0:
        print("error: infinite velocity")
    else:
        slope = -(y1-y2)/(x2-x1)
        velocity = 1/slope*fps*scale_mm/scale_pixels

    return velocity
        
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
# Specify the folder path containing TIFF images
folder_path = r"d:\Users\Ana\Desktop\experiments\experiments impact\45deg_d2mm\SaveData"


# Specify experiment details
scale_pixels = 119
scale_mm = 13
ideal_diam = 2
fps = 5000
length = 2*scale_pixels
angle = 45
height = 103





SD = pd.read_csv(folder_path + '\\SplashData_Cent_ST.csv',index_col = 'Ind')

StackList = SD.index  
jet_velocity = []
splash_velocity = []
Circularity = []
for s in StackList:

    _, TopStack = cv2.imreadmulti(folder_path + '\\Top_' + s + '.tif', [], -1 )
    # count = 6
    # for file in TopStack: file = average_stack(TopStack, count)
    line_coordinates = []
    extract_line_coord(cv2.normalize(TopStack[20], dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX))
    print(line_coordinates)
    print(s)
    
    # files = os.listdir(folder_path)
    # file_path = os.path.join(folder_path, files[20])
    # image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
 

        

    # Call the function to read TIFF images, draw the line, and extract pixels along that line
    list_of_lists_cv2, reslice = read_tiff_images_cv2(TopStack, line_coordinates)    
    reslice_jet, reslice_splash = reslice

        
    image = np.array(reslice_jet, dtype=np.uint16)
    line_coordinates = []
    extract_line_coord(cv2.normalize(image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX))

    velocity_jet = extract_velocity(line_coordinates)
    print("Velocity jet=", velocity_jet, "mm/s")

    image = np.array(reslice_splash, dtype=np.uint16)
    line_coordinates = []
    extract_line_coord(cv2.normalize(image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX))

    velocity_splash = extract_velocity(line_coordinates)
    print("Velocity splash=", velocity_splash, "mm/s")



    jet_velocity.append(velocity_jet)
    splash_velocity.append(velocity_splash)

    SD.loc[s, 'angle[degrees]'] = angle
    SD.loc[s, 'IdealDiam[mm]'] = ideal_diam
    SD.loc[s, 'Height[cm]'] = height
    SD.loc[s,'Top Velocity jet'] = velocity_jet
    SD.loc[s,'Top Velocity splash'] = velocity_splash
    # SD.to_csv(folder_path + '\\SplashData_Cent_ST.csv',index_label = 'Ind')