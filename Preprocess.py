from cv2 import cv2
import numpy as np


def preprocess(imgOriginal):

    imgGreyScale = extractValue(imgOriginal)

    imgGreyScaleMaxContrast = maximizeContrast(imgGreyScale)

    heigth, width = imgGreyScale.shape

    imgBlurred = np.zeros((heigth, width, 1), np.uint8)

    imgBlurred = cv2.GaussianBlur(imgGreyScaleMaxContrast, (5, 5), cv2.BORDER_DEFAULT)

    ret, imgThresh = cv2.threshold(imgBlurred, 90, 255, cv2.THRESH_BINARY_INV)

    # imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 9)
    return imgGreyScale, imgThresh

def preprocessPlate(imgOriginal):

    imgGreyScale = extractValue(imgOriginal)

    imgGreyScaleMaxContrast = maximizeContrast(imgGreyScale)

    heigth, width = imgGreyScale.shape

    imgBlurred = np.zeros((heigth, width, 1), np.uint8)

    imgBlurred = cv2.GaussianBlur(imgGreyScaleMaxContrast, (5, 5), 0)

    ret, imgThresh = cv2.threshold(imgGreyScale, 90, 255, cv2.THRESH_BINARY_INV)

    # imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 9)
    return imgGreyScale, imgThresh



def extractValue(imgOriginal):

    #*.shape return height(num of pix), width(num of pix), channels
    height, width, numChannels = imgOriginal.shape

    #BGR to HSV conversion and return value of HSV image
    imgHSV = np.zeros((height, width, 3), np.uint8)

    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)


    return imgValue


def maximizeContrast(imgGrayScale):

    height, width = imgGrayScale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(imgGrayScale, cv2.MORPH_TOPHAT, kernel)
    imgBlackHat = cv2.morphologyEx(imgGrayScale, cv2.MORPH_BLACKHAT, kernel)

    imgGrayScaleMaxContrast = cv2.add(imgGrayScale, imgTopHat)
    imgGrayScaleMaxContrast = cv2.subtract(imgGrayScaleMaxContrast, imgBlackHat)

    return imgGrayScaleMaxContrast
