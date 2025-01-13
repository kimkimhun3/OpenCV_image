import cv2
import numpy as np

def compare_sharpen_methods_after_clahe(image):
    # Function to resize images for display
    def resize_image(img, scale=0.5):
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE first
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    
    # Sharpening Techniques
    
    # Basic Sharpening
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    sharpen_basic = cv2.filter2D(clahe_img, -1, sharpen_kernel)
    
    # Laplacian Sharpening
    laplacian = cv2.Laplacian(clahe_img, cv2.CV_64F)
    sharpen_laplacian = cv2.convertScaleAbs(clahe_img - laplacian)
    
    # Unsharp Masking
    blurred = cv2.GaussianBlur(clahe_img, (5, 5), 0)
    sharpen_unsharp = cv2.addWeighted(clahe_img, 1.5, blurred, -0.5, 0)
    
    # High-Pass Filtering
    high_pass_kernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    sharpen_high_pass = cv2.filter2D(clahe_img, -1, high_pass_kernel)
    
    # Custom Sharpening
    custom_sharpen_kernel = np.array([
        [-1, -1, -1, -1, -1],
        [-1, 2, 2, 2, -1],
        [-1, 2, 8, 2, -1],
        [-1, 2, 2, 2, -1],
        [-1, -1, -1, -1, -1]
    ], dtype=np.float32) / 8.0
    sharpen_custom = cv2.filter2D(clahe_img, -1, custom_sharpen_kernel)
    
    # Resize images
    scale_factor = 0.5
    gray_resized = resize_image(gray, scale=scale_factor)
    clahe_resized = resize_image(clahe_img, scale=scale_factor)
    sharpen_basic_resized = resize_image(sharpen_basic, scale=scale_factor)
    sharpen_laplacian_resized = resize_image(sharpen_laplacian, scale=scale_factor)
    sharpen_unsharp_resized = resize_image(sharpen_unsharp, scale=scale_factor)
    sharpen_high_pass_resized = resize_image(sharpen_high_pass, scale=scale_factor)
    sharpen_custom_resized = resize_image(sharpen_custom, scale=scale_factor)
    
    # Display results
    cv2.imshow('Original Gray', gray_resized)
    cv2.imshow('After CLAHE', clahe_resized)
    cv2.imshow('CLAHE + Basic Sharpening', sharpen_basic_resized)
    cv2.imshow('CLAHE + Laplacian Sharpening', sharpen_laplacian_resized)
    cv2.imshow('CLAHE + Unsharp Masking', sharpen_unsharp_resized)
    cv2.imshow('CLAHE + High-Pass Filter', sharpen_high_pass_resized)
    cv2.imshow('CLAHE + Custom Sharpening', sharpen_custom_resized)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Read and process image
image = cv2.imread('input.jpg')
if image is None:
    print("Error: Couldn't open the image file.")
    exit()

# Compare sharpening methods
compare_sharpen_methods_after_clahe(image)
