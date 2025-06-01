# Import required libraries
import cv2
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf
import keras

# Load the pre-trained ASL (American Sign Language) classification model
model = keras.models.load_model("asl_classifier.h5")

# Dictionary to map model output to corresponding letters
labels_dict = {
    0: '0', 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F',
    7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M',
    14: 'N', 15: 'O', 16: 'P', 17: "Q", 18: 'R', 19: 'S',
    20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z'
}

# Rectangle and preprocessing settings
color_dict = (0, 255, 0)  # Color for the rectangle (green)
x, y, w, h = 0, 0, 64, 64  # Dimensions (unused rectangle definition)

# Image preprocessing parameters
img_size = 128
minValue = 70

# Start video capture from default webcam
source = cv2.VideoCapture(0)

# Variables to manage prediction frequency and output string
count = 0
string = " "
prev = " "
prev_val = 0

# Main loop for video processing
while True:
    ret, img = source.read()  # Read frame from webcam
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Draw a fixed rectangle on the frame (ROI for hand sign input)
    cv2.rectangle(img, (24, 24), (250, 250), color_dict, 2)

    # Crop the grayscale image to the rectangle
    crop_img = gray[24:250, 24:250]

    count += 1
    if count % 100 == 0:
        prev_val = count

    # Display current frame count on the screen
    cv2.putText(img, str(prev_val // 100), (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    # Apply Gaussian blur and adaptive thresholding to prepare for model input
    blur = cv2.GaussianBlur(crop_img, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Resize, normalize, and reshape image for the model
    resized = cv2.resize(res, (img_size, img_size))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, img_size, img_size, 1))

    # Predict the letter using the model
    result = model.predict(reshaped)
    label = np.argmax(result, axis=1)[0]

    # Append predicted letter to the final string every 300 frames
    if count == 300:
        count = 99  # Reset the count
        prev = labels_dict[label]  # Get the predicted label

        if label == 0:
            # If label is 0, append a space
            string += " "
        else:
            # Otherwise, append the predicted letter
            string += prev

    # Display the latest prediction and full typed string
    cv2.putText(img, prev, (24, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, string, (275, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # Show the processed thresholded image and live camera feed
    cv2.imshow("Gray", res)
    cv2.imshow('LIVE', img)

    # Exit when ESC key is pressed
    key = cv2.waitKey(1)
    if key == 27:
        break

# Print the final predicted string
print(string)

# Release resources and close all windows
cv2.destroyAllWindows()
source.release()
cv2.destroyAllWindows()
