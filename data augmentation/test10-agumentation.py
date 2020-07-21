import cv2

path = "D:/CV group/competition-data/X ray/objects/slingshot/400003_1.jpg"

img = cv2.imread(path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray_image', img_gray)
cv2.waitKey(0)

ret, th = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('thresh', th)
cv2.waitKey(0)
