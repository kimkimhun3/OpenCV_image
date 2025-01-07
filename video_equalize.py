#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

# def plot_hist(frame, frame1, k=0, l=0, frame_name='frame', frame1_name='frame1'):
#     # ヒストグラムの表示
#     ax[k, l].plot(frame, label=str(frame_name))
#     ax[k, l].plot(frame1, label=str(frame1_name))
#     ax[k, l].set_xlim([0, 256])
#     ax[k, l].set_ylim([0, max(max(frame), max(frame1))])
#     ax[k, l].legend()

def something(frame, frame1, YCrCb=0):
    # BGRをYCrCbに変換
    orgYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    lutYCrCb = cv2.cvtColor(frame1, cv2.COLOR_BGR2YCR_CB)
    # 輝度のヒストグラムを作成
    histOrgY = cv2.calcHist([orgYCrCb], [YCrCb], None, [256], [0, 256])  # 0:Y 1:Cr 2:Cb
    histLutY = cv2.calcHist([lutYCrCb], [YCrCb], None, [256], [0, 256])
    return histOrgY, histLutY

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

#video_input = cv2.VideoCapture(0)
size = (600, 450)  # Resizing dimensions

# while True:
#     #frame = cv2.imread('s7.jpg', 1)
#     frame2 = cv2.imread('s1.jpg', 1)
    
#     # cv2.imshow("frame", frame)
#     # cv2.imwrite('frame.jpg', frame)  # Save 'frame' as 'frame.jpg'

#     cv2.imshow("input", frame2)
#     cv2.imwrite('input.jpg', frame2)  # Save 'frame2' as 'frame2.jpg'
    
#     bgr = treatise(frame2, size)
#     cv2.imshow("equalizeHist", bgr)
#     cv2.imwrite('equalizeHist.jpg', bgr)  # Save the equalized image in 'equalizeHist.jpg'

#     bgr = histogram_equalize(frame2, size)
#     cv2.imshow("equalizeHist_rgb", bgr)
#     cv2.imwrite('equalizeHist_rgb.jpg', bgr)  # Save the equalized RGB image in 'equalizeHist_rgb.jpg'

#     bgr = histogram_equalize_yuv(frame2, size)
#     cv2.imshow("equalizeHist_yuv", bgr)
#     cv2.imwrite('equalizeHist_yuv.jpg', bgr)  # Save the equalized YUV image in 'equalizeHist_yuv.jpg'

#     bgr = histogram_equalize_hsv(frame2, size)
#     cv2.imshow("equalizeHist_hsv", bgr)
#     cv2.imwrite('equalizeHist_hsv.jpg', bgr)  # Save the equalized HSV image in 'equalizeHist_hsv.jpg'

#     bgr = histogram_equalize_treat(frame2, size)
#     cv2.imshow("equalize_treat", bgr)
#     cv2.imwrite('equalize_treat.jpg', bgr)  # Save the treated image in 'equalize_treat.jpg'

        
#     # fig, ax = plt.subplots(2, 3, figsize=(12, 4))
#     # histOrgY, histLutY = something(frame, frame2, 0)
#     # plot_hist(histOrgY, histLutY, 0, 0, "frame", "frame2")
#     # histOrgY, histLutY = something(frame, frame2, 1)
#     # plot_hist(histOrgY, histLutY, 0, 1, "frame", "frame2")
#     # histOrgY, histLutY = something(frame, frame2, 2)
#     # plot_hist(histOrgY, histLutY, 0, 2, "frame", "frame2")
    
#     # histOrgY, histLutY = something(frame, bgr, 0)
#     # plot_hist(histOrgY, histLutY, 1, 0, "frame", "bgr")
#     # histOrgY, histLutY = something(frame, bgr, 1)
#     # plot_hist(histOrgY, histLutY, 1, 1, "frame", "bgr")
#     # histOrgY, histLutY = something(frame, bgr, 2)
#     # plot_hist(histOrgY, histLutY, 1, 2, "frame", "bgr")
#     # # plt.show()
#     # plt.close()
    
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break


cap = cv2.VideoCapture(0)  # Start capturing from the webcam

while True:
    ret, frame2 = cap.read()  # Capture a frame from the webcam
    if not ret:
        print("Failed to grab frame")
        break
    
    # Display the input frame
    cv2.imshow("input", frame2)

    # Apply equalization and show results
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

    # Check for the 'Esc' key to break the loop and stop capturing
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
