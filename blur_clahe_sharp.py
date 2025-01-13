import cv2
import numpy as np

def compare_methods_with_unsharp_masking(image):
    # Function to resize images for display
    def resize_image(img, scale=0.5):
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    
    # Unsharp Masking Function
    def apply_unsharp_masking(img, strength=1.5):
        gaussian_blurred = cv2.GaussianBlur(img, (5, 5), 0)
        return cv2.addWeighted(img, 1 + strength, gaussian_blurred, -strength, 0)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Average Blur
    blur_avg = cv2.blur(gray, (5, 5))
    blur_avg_clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8)).apply(blur_avg)
    blur_avg_clahe_unsharp = apply_unsharp_masking(blur_avg_clahe)
    
    # Gaussian Blur
    blur_gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
    blur_gaussian_clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8)).apply(blur_gaussian)
    blur_gaussian_clahe_unsharp = apply_unsharp_masking(blur_gaussian_clahe)
    
    # Median Blur
    blur_median = cv2.medianBlur(gray, 5)
    blur_median_clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8)).apply(blur_median)
    blur_median_clahe_unsharp = apply_unsharp_masking(blur_median_clahe)
    
    # Bilateral Filter
    blur_bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    blur_bilateral_clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8)).apply(blur_bilateral)
    blur_bilateral_clahe_unsharp = apply_unsharp_masking(blur_bilateral_clahe)
    
    # Custom 2D Filter
    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float32) / 16
    blur_2d = cv2.filter2D(gray, -1, kernel)
    blur_2d_clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8)).apply(blur_2d)
    blur_2d_clahe_unsharp = apply_unsharp_masking(blur_2d_clahe)
    
    # Resize images
    scale_factor = 0.5
    gray_resized = resize_image(gray, scale=scale_factor)
    blur_avg_resized = resize_image(blur_avg_clahe_unsharp, scale=scale_factor)
    blur_gaussian_resized = resize_image(blur_gaussian_clahe_unsharp, scale=scale_factor)
    blur_median_resized = resize_image(blur_median_clahe_unsharp, scale=scale_factor)
    blur_bilateral_resized = resize_image(blur_bilateral_clahe_unsharp, scale=scale_factor)
    blur_2d_resized = resize_image(blur_2d_clahe_unsharp, scale=scale_factor)
    
    # Display results
    cv2.imshow('Original Gray', gray_resized)
    cv2.imshow('Avg Blur + CLAHE + Unsharp', blur_avg_resized)
    cv2.imshow('Gaussian Blur + CLAHE + Unsharp', blur_gaussian_resized)
    cv2.imshow('Median Blur + CLAHE + Unsharp', blur_median_resized)
    cv2.imshow('Bilateral Filter + CLAHE + Unsharp', blur_bilateral_resized)
    cv2.imshow('Custom 2D Filter + CLAHE + Unsharp', blur_2d_resized)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Read and process image
image = cv2.imread('input.jpg')
if image is None:
    print("Error: Couldn't open the image file.")
    exit()

# Compare methods
compare_methods_with_unsharp_masking(image)
