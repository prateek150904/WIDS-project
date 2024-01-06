# counting_coin.py
import numpy as np
import cv2
import cvzone

# Load the images and resize it
img = cv2.imread("coin.png")
img = cv2.resize(img, (640, 800))
# image_copy = cv2.imread("one.jpg")
# image_copy = cv2.resize(image_copy, (640, 800))



# Creat a Preprocessing function to apply necessary operations on the input image
def preProcessing(img):
    
    # Convert image to grayscale
    imgPre = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply blur to the image
    imgPre = cv2.blur(img, (15, 15), 0)
    
    
    # Apply Canny edge detection
    # Select some nice thresh1 , thresh2 so that go get a almost clear edge of wanted object 
    
    imgPre = cv2.Canny(imgPre, 20 , 20)
    
    # Apply dilation and morphological closing
    kernel = np.ones((3, 3), np.uint8)
    imgPre = cv2.dilate(imgPre, kernel, iterations=1)
    imgPre = cv2.morphologyEx(imgPre, cv2.MORPH_CLOSE, kernel)
    return imgPre

# Call preprocessing function
imgPre = preProcessing(img)

# Find contours in the preprocessed image having minimum area of 2000 ( set according to your image )
imgContours, conFound = cvzone.findContours(img, imgPre, minArea=2000)

    
# Stack the images for display 
# imgStacked = cvzone.stackImages([image_copy, img, imgPre, imgContours], 2, 0.5)

# Create a dictionary to store contour areas
area = {}

# Calculate and store the area of each contour
for i in range(len(conFound)):
    cnt = conFound[i]
    ar = cv2.contourArea(cnt['cnt'], True)
    area[i] = np.abs(ar)
print(area)
# Sort the areas in descending order
srt = sorted(area.items(), key=lambda x: x[1], reverse=True)

# Convert the sorted results to a numpy array
results = np.array(srt).astype("int")

# Count the number of contours with area grater than 2000 ( set according to your image )
num = np.argwhere((results[:, 1]) > 2000).shape[0]
print("Number of coins is ", num-1)

# Display the stacked images
cv2.imshow("Image", imgContours)
cv2.waitKey(0)