import cv2
import numpy as np

def compare_sharpen_methods_before_clahe(image):
    # Function to resize images for display
    def resize_image(img, scale=0.5):
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Sharpening Techniques
    
    # Basic Sharpening
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    sharpen_basic = cv2.filter2D(gray, -1, sharpen_kernel)
    
    # Laplacian Sharpening
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpen_laplacian = cv2.convertScaleAbs(gray - laplacian)
    
    # Unsharp Masking
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sharpen_unsharp = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    
    # High-Pass Filtering
    high_pass_kernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    sharpen_high_pass = cv2.filter2D(gray, -1, high_pass_kernel)
    
    # Custom Sharpening
    custom_sharpen_kernel = np.array([
        [-1, -1, -1, -1, -1],
        [-1, 2, 2, 2, -1],
        [-1, 2, 8, 2, -1],
        [-1, 2, 2, 2, -1],
        [-1, -1, -1, -1, -1]
    ], dtype=np.float32) / 8.0
    sharpen_custom = cv2.filter2D(gray, -1, custom_sharpen_kernel)
    
    # Apply CLAHE after sharpening
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    sharpen_basic_clahe = clahe.apply(sharpen_basic)
    sharpen_laplacian_clahe = clahe.apply(sharpen_laplacian)
    sharpen_unsharp_clahe = clahe.apply(sharpen_unsharp)
    sharpen_high_pass_clahe = clahe.apply(sharpen_high_pass)
    sharpen_custom_clahe = clahe.apply(sharpen_custom)
    
    # Resize images
    scale_factor = 0.5
    gray_resized = resize_image(gray, scale=scale_factor)
    sharpen_basic_resized = resize_image(sharpen_basic_clahe, scale=scale_factor)
    sharpen_laplacian_resized = resize_image(sharpen_laplacian_clahe, scale=scale_factor)
    sharpen_unsharp_resized = resize_image(sharpen_unsharp_clahe, scale=scale_factor)
    sharpen_high_pass_resized = resize_image(sharpen_high_pass_clahe, scale=scale_factor)
    sharpen_custom_resized = resize_image(sharpen_custom_clahe, scale=scale_factor)
    
    # Display results
    cv2.imshow('Original Gray', gray_resized)
    cv2.imshow('Basic Sharpening + CLAHE', sharpen_basic_resized)
    cv2.imshow('Laplacian Sharpening + CLAHE', sharpen_laplacian_resized)
    cv2.imshow('Unsharp Masking + CLAHE', sharpen_unsharp_resized)
    cv2.imshow('High-Pass Filter + CLAHE', sharpen_high_pass_resized)
    cv2.imshow('Custom Sharpening + CLAHE', sharpen_custom_resized)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Read and process image
image = cv2.imread('input.jpg')
if image is None:
    print("Error: Couldn't open the image file.")
    exit()

# Compare sharpening methods
compare_sharpen_methods_before_clahe(image)
