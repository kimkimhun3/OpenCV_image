#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

def treatise(frame, size):
    frame1 = cv2.resize(frame, dsize=size, interpolation=cv2.INTER_CUBIC)
    gridsize = 8
    lab = cv2.cvtColor(frame1, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))  # Convert tuple to list
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(gridsize, gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])  # Modify the L channel
    lab = cv2.merge(lab_planes)  # Merge channels back
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

def histogram_equalize(img, size):
    img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_CUBIC)
    b, g, r = cv2.split(img)
    red = cv2.equalizeHist(r)
    green = cv2.equalizeHist(g)
    blue = cv2.equalizeHist(b)
    return cv2.merge((blue, green, red))

def histogram_equalize_treat(img, size):
    img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_CUBIC)
    b, g, r = cv2.split(img)
    red = cv2.equalizeHist(r)
    green = cv2.equalizeHist(g)
    blue = cv2.equalizeHist(b)
    bgr = cv2.merge((blue, green, red))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))  # Convert tuple to list
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

def histogram_equalize_hsv(img, size):
    img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img)
    h = cv2.equalizeHist(h)
    s = cv2.equalizeHist(s)
    v = cv2.equalizeHist(v)
    bgr = cv2.merge((h, s, v))
    return bgr

def histogram_equalize_yuv(img, size):
    img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    yuv = img.copy()
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return img_output

video_input = cv2.VideoCapture(0)
size = (600, 450)  # Resizing dimensions


frame2 = cv2.imread('s1.jpg', 1)

# Save images once before the loop starts
cv2.imwrite('input.jpg', frame2)  # Save 'frame2' as 'input.jpg'

bgr = treatise(frame2, size)
cv2.imwrite('equalizeHist.jpg', bgr) 

bgr = histogram_equalize(frame2, size)
cv2.imwrite('equalizeHist_rgb.jpg', bgr)

bgr = histogram_equalize_yuv(frame2, size)
cv2.imwrite('equalizeHist_yuv.jpg', bgr)

bgr = histogram_equalize_hsv(frame2, size)
cv2.imwrite('equalizeHist_hsv.jpg', bgr)

bgr = histogram_equalize_treat(frame2, size)
cv2.imwrite('equalize_treat.jpg', bgr)

# Now enter the loop for displaying the images
while True:
    cv2.imshow("input", frame2)

    # Apply the transformations again for display (without saving)
    bgr = treatise(frame2, size)
    cv2.imshow("equalizeHist", bgr)

    bgr = histogram_equalize(frame2, size)
    cv2.imshow("equalizeHist_rgb", bgr)

    bgr = histogram_equalize_yuv(frame2, size)
    cv2.imshow("equalizeHist_yuv", bgr)

    bgr = histogram_equalize_hsv(frame2, size)
    cv2.imshow("equalizeHist_hsv", bgr)

    bgr = histogram_equalize_treat(frame2, size)
    cv2.imshow("equalize_treat", bgr)

    # Wait for the user to press 'Esc' to break, or close the window by clicking the 'X' button
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # Press 'Esc' to exit
        break

    # Optionally check if the window was closed by clicking the 'X' button
    if cv2.getWindowProperty("input", cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()  # Close all windows when done
