import cv2
import numpy as np

def enhance_foggy_image(frame):
    # 1. Convert to grayscale if not already
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    # 3. Apply Gaussian blur using a custom kernel
    # gaussian_kernel = np.array([
    # [1, 2, 1],
    # [2, 4, 2],
    # [1, 2, 1]
    # ], dtype=np.float32)
    # gaussian_kernel /= np.sum(gaussian_kernel)  #Normalize the kernel
    # blurred = cv2.filter2D(clahe_img, -1, gaussian_kernel)


    kernel = np.ones((5,5),np.float32)/25
    blurred = cv2.filter2D(clahe_img,-1,kernel)

    # 4. Apply 2D sharpening
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)
    alpha = 1.2  # Contrast control
    beta = -40    # Brightness control
    final = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=beta)

    return final

# MP4 Video Processing
cap = cv2.VideoCapture("720.mp4")
if not cap.isOpened():
    print("Error: Couldn't open the video file.")
    exit()

while True:
    ret, frame = cap.read()  # Capture a frame from the video
    if not ret:
        print("Failed to grab frame or end of video reached")
        break
    
    # Apply the enhancement process
    enhanced_frame = enhance_foggy_image(frame)

    # Display the result
    cv2.imshow("Enhanced Frame", enhanced_frame)

    # Check for the 'Esc' key to break the loop and stop capturing
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
