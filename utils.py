
import os
import cv2
import numpy
from colorthief import ColorThief
import numpy as np
import math

JAUNDICE_FOLDER = "./Jaundice/"
NON_JAUNDICE_FOLDER = "./No Jaundice/"


def getJaundiceFilesArray():
    return os.listdir(JAUNDICE_FOLDER)

def getNONJaundiceFilesArray():
    return os.listdir(NON_JAUNDICE_FOLDER)

def getJaundiceYellowIntensity(filename):
    color_thief = ColorThief(JAUNDICE_FOLDER+filename)
    dominant_color = color_thief.get_color(quality=1)
    return (float(dominant_color[0]) + float(dominant_color[1]))

def getNONJaundiceYellowIntensity(filename):
    color_thief = ColorThief(NON_JAUNDICE_FOLDER+filename)
    dominant_color = color_thief.get_color(quality=1)
    return (float(dominant_color[0]) + float(dominant_color[1]))

def getYellowIntensityWithFileName(filename):
    color_thief = ColorThief(filename)
    dominant_color = color_thief.get_color(quality=1)
    return (float(dominant_color[0]) + float(dominant_color[1]))
