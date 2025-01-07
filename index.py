#-*- coding:utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 入力画像を読み込み
img = cv2.imread("original.jpg")
b, g, r = cv2.split(img)

def calc_hist(b,g,r):
    hist_r = cv2.calcHist([r],[0],None,[256],[0,255])
    hist_g = cv2.calcHist([g],[0],None,[256],[0,255])
    hist_b = cv2.calcHist([b],[0],None,[256],[0,255])
    return hist_r, hist_g, hist_b

def plot_hist(hist_r,hist_g,hist_b,file='0'):
    # グラフの作成
    a=plt.figure()
    plt.xlim(0, 255)
    plt.plot(hist_r, "-r", label="Red"+str(255))
    plt.plot(hist_g, "-g", label="Green"+str(255))
    plt.plot(hist_b, "-b", label="Blue"+str(255))
    plt.xlabel("Pixel value", fontsize=20)
    plt.ylabel("Number of pixels", fontsize=20)
    plt.legend()
    plt.grid()
    plt.savefig('original_'+str(file)+'.jpg')
    plt.show()
    plt.close()

def processing(img,file):
    b, g, r = cv2.split(img)
    hist_r, hist_g, hist_b=calc_hist(b,g,r)
    plot_hist(hist_r, hist_g, hist_b,file)

# 入力画像を読み込み
file1="s1"
img = cv2.imread(file1+'.jpg')
processing(img,file1)
file2="s7"
img7 = cv2.imread(file2+'.jpg')
processing(img7,file2)

img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
processing(img,file1+'_YCrCb')
img7 = cv2.cvtColor(img7, cv2.COLOR_BGR2YCR_CB)
processing(img7,file2+'_YCrCb')

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
processing(img,file1+'_HSV')
img7 = cv2.cvtColor(img7, cv2.COLOR_BGR2HSV)
processing(img7,file2+'_HSV')

img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
processing(img,file1+'_YUV')
img7 = cv2.cvtColor(img7, cv2.COLOR_BGR2YUV)
processing(img7,file2+'_YUV')
