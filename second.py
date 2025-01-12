import cv2
import numpy as np

def enhance_foggy_image(frame):
    # 1. Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Apply 2D sharpening
    sharpen_kernel = np.array([
        [0, -0.5, 0], 
        [-0.5, 3, -0.5], 
        [0, -0.5, 0] 
    ], dtype=np.float32)
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)

        # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(16, 16))
    clahe_img = clahe.apply(sharpened)

        # Just Extra
    alpha = 1.2  # Contrast control
    beta = -60    # Brightness control
    final = cv2.convertScaleAbs(clahe_img, alpha=alpha, beta=beta)

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
