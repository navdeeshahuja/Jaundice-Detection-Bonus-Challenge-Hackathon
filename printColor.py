
# Required modules
import cv2
import numpy
from colorthief import ColorThief
import numpy as np
import math

filename = 'recentClick.jpg'

# Constants for finding range of skin color in YCrCb
min_YCrCb = numpy.array([0,133,77],numpy.uint8)
max_YCrCb = numpy.array([255,173,127],numpy.uint8)

# Get pointer to video frames from primary device
# videoFrame = cv2.VideoCapture(0)

# if(True): # any key pressed has a value >= 0
# Grab video frame, decode it and return next video frame
# readSucsess, sourceImage = videoFrame.read()
sourceImage = cv2.imread(filename)
# print sourceImage.shape

# Convert image to YCrCb
imageYCrCb = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2YCR_CB)

# Find region with skin tone in YCrCb image
skinRegion = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)

color_thief = ColorThief(filename)

dominant_color = color_thief.get_color(quality=1)
## Y Equation
Y = math.trunc((0.257 * dominant_color[0]) + (0.504 * dominant_color[1]) + (0.098 * dominant_color[2]) + 16)
##Cb Equation
Cb = math.trunc(((-0.148) * dominant_color[0]) - (0.291 * dominant_color[1]) + (0.439 * dominant_color[2]) + 128)
## Cr Equantion
Cr = math.trunc((0.439 * dominant_color[0]) - (0.368 * dominant_color[1]) - (0.071 * dominant_color[2]) + 128)
print("rgb -> ", dominant_color)
print("YCbCr -> ", (Y, Cb, Cr))


