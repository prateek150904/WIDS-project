import numpy as np
import cv2
import cvzone

img = cv2.imread("coin.png")
img = cv2.resize(img, (800, 700))
img_copy = img.copy()
imgPre = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
imgPre = cv2.GaussianBlur(imgPre,(9,9),3 )


# imgPre = cv2.Canny(imgPre , 20,20)
(T, thresh) = cv2.threshold(imgPre,160 , 255,cv2.THRESH_BINARY)

contours , _ = cv2.findContours(thresh , cv2.RETR_TREE , cv2.CHAIN_APPROX_NONE)
# img_copy = cv2.drawContours(img_copy , contours , -1 , (0,255,0) , 3)
# cv2.imshow("Threshold", img_copy)
area = {}
for i in range(len(contours)):
    area[i] = cv2.contourArea(contours[i])
# print(area)

srt = sorted(area.items() , key = lambda x : x[1] , reverse = True)
results = np.array(srt).astype("int")
num = np.argwhere(results[: , 1]>500).shape[0]

for i in range(1,num):
    img_copy = cv2.drawContours(img_copy , contours , results[i,0] , (0,255,0) , 3)

print("THe number of coins is : ", num-1)

cv2.imshow('coin' ,img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()


