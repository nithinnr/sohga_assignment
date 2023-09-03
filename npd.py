import os
import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt

# Path to the input folder containing images
input_folder = r'C:\Users\Venkata krishnareddy\OneDrive\Desktop\NPD\dataset/images'

# Path to the output folder for saving result images
output_folder = r'C:\Users\Venkata krishnareddy\OneDrive\Desktop\NPD\outds/images'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the input folder
for image_filename in os.listdir(input_folder):
    if image_filename.endswith(('.jpg', '.jpeg', '.png')):  # Check for image file extensions
        # Load the image
        image_path = os.path.join(input_folder, image_filename)
        img = cv2.imread(image_path)
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter for noise reduction
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply edge detection using Canny
        edged = cv2.Canny(bfilter, 30, 200)
        
        # Find contours
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        # Draw bounding boxes around potential number plates
        for contour in contours:
            # Approximate the contour with a polygon
            approx = cv2.approxPolyDP(contour, 10, True)
            
            # Check if the contour is a rectangle with 4 corners (a potential number plate)
            if len(approx) == 4:
                # Extract the coordinates of the 4 corners
                (x, y, w, h) = cv2.boundingRect(approx)
                
                # Draw bounding box on the original image
                res = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Save the result image in the output folder
        output_path = os.path.join(output_folder, image_filename)
        cv2.imwrite(output_path, res)

# Display a message when processing is complete
print("Number plate detection completed for all images.")




