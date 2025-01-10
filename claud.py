#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

def enhance_visibility(img, size, mode='color_clahe', clip_limit=2.0, tile_size=(16,16)):
    # Resize image
    img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_CUBIC)
    
    if mode == 'color_clahe':
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
    elif mode == 'gray_clahe':
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        enhanced = clahe.apply(gray)
        
    elif mode == 'color_hist':
        # Split channels and apply histogram equalization
        b, g, r = cv2.split(img)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        enhanced = cv2.merge((b, g, r))
        
    elif mode == 'gray_hist':
        # Convert to grayscale and apply histogram equalization
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.equalizeHist(gray)
        
    else:
        raise ValueError("Invalid mode selected")
        
    return enhanced

def process_video(video_path, size=(960, 540)):
    cap = cv2.VideoCapture(video_path)
    mode = 'color_clahe'  # Default mode
    clip_limit = 2.0
    tile_size = (8, 8)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        enhanced = enhance_visibility(frame, size, mode, clip_limit, tile_size)
        cv2.imshow("Enhanced", enhanced)
        
        # Key controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('1'):  # Color CLAHE
            mode = 'color_clahe'
            print("Mode: Color CLAHE")
        elif key == ord('2'):  # Grayscale CLAHE
            mode = 'gray_clahe'
            print("Mode: Grayscale CLAHE")
        elif key == ord('3'):  # Color Histogram Equalization
            mode = 'color_hist'
            print("Mode: Color Histogram Equalization")
        elif key == ord('4'):  # Grayscale Histogram Equalization
            mode = 'gray_hist'
            print("Mode: Grayscale Histogram Equalization")
        elif key == ord('+'):  # Increase CLAHE clip limit
            clip_limit = min(clip_limit + 0.5, 40.0)
            print(f"CLAHE clip limit: {clip_limit}")
        elif key == ord('-'):  # Decrease CLAHE clip limit
            clip_limit = max(clip_limit - 0.5, 40.0)
            print(f"CLAHE clip limit: {clip_limit}")
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    video_path = "720.mp4"
    process_video(video_path)