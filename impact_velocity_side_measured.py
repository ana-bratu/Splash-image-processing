# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:33:11 2024

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

    reslice = []
    # Iterate through images
    for file in TopStack[0:50]:
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
            y2 = y1
            angle = math.atan2(line[1][1] - line[0][1], line[1][0] - line[0][0])
            x2 = x1 
            y2 = y1 + int(length) #*int(length * math.sin(angle))
         
            # Convert the image data to a list
            image_data = image.tolist()

            # Append the list of pixel values to the main list
            all_images.append(image_data)

            # Extract pixels along the drawn line using Bresenham's algorithm
            line_pixels = []
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx - dy

            while True:
                line_pixels.append(image[y1, x1].tolist())

                if x1 == x2 and y1 == y2:
                    break

                e2 = 2 * err
                if e2 > -dy:
                    err = err - dy
                    x1 = x1 + sx
                if e2 < dx:
                    err = err + dx
                    y1 = y1 + sy

            reslice.append(line_pixels)
            
        
        except Exception as e:
            print(f"Error processing {file}: {e}")
    # drew_line=[x2, y2, x3, y3]
    
    
    print(angle)

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
        print(slope)
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
folder_path = r"d:\Users\Ana\Desktop\experiments\experiments internship\5000fps\experiments\60degrees\H103mm_d5.4mm\SaveData"# Specify experiment details
scale_pixels = 75
scale_mm = 5.4
fps = 5000
length = scale_pixels
angle = 30
height = 103





SD = pd.read_csv(folder_path + '\\SplashData_Cent_ST.csv',index_col = 'Ind')

StackList = SD.index  
impact_v = []
Circularity = []
for s in StackList:

    _, TopStack = cv2.imreadmulti(folder_path + '\\SIDE_' + s + '.tif', [], -1 )
    # count = 6
    # for file in TopStack: file = average_stack(TopStack, count)
    line_coordinates = []
    extract_line_coord(cv2.normalize(TopStack[1], dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX))
    # print(line_coordinates)
    # print(s)
    
    # files = os.listdir(folder_path)
    # file_path = os.path.join(folder_path, files[20])
    # image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
 

        

    # Call the function to read TIFF images, draw the line, and extract pixels along that line
    list_of_lists_cv2, reslice = read_tiff_images_cv2(TopStack, line_coordinates)    
    

        
    image = np.array(reslice, dtype=np.uint16)
    line_coordinates = []
    extract_line_coord(cv2.normalize(image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX))
    print(line_coordinates)
    impact_velocity = extract_velocity(line_coordinates)
    print("Impact Velocity", impact_velocity, "mm/s")

    


    impact_v.append(impact_velocity)
    
    SD.loc[s,'Impact Velocity [mm/s]'] = impact_velocity
    # SD.to_csv(folder_path + '\\SplashData_Cent_ST.csv',index_label = 'Ind')