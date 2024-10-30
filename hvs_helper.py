import cv2
import numpy as np

def nothing(x):
    pass

image_path = 'C:\\Users\\Lab\\Desktop\\megakaryocytes\\data\\11_54.png'
image = cv2.imread(image_path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

cv2.namedWindow("HSV Adjustments")

cv2.createTrackbar("Lower H", "HSV Adjustments", 0, 179, nothing)
cv2.createTrackbar("Lower S", "HSV Adjustments", 0, 255, nothing)
cv2.createTrackbar("Lower V", "HSV Adjustments", 0, 255, nothing)

cv2.createTrackbar("Upper H", "HSV Adjustments", 179, 179, nothing)
cv2.createTrackbar("Upper S", "HSV Adjustments", 255, 255, nothing)
cv2.createTrackbar("Upper V", "HSV Adjustments", 255, 255, nothing)

while True:
    lower_h = cv2.getTrackbarPos("Lower H", "HSV Adjustments")
    lower_s = cv2.getTrackbarPos("Lower S", "HSV Adjustments")
    lower_v = cv2.getTrackbarPos("Lower V", "HSV Adjustments")
    upper_h = cv2.getTrackbarPos("Upper H", "HSV Adjustments")
    upper_s = cv2.getTrackbarPos("Upper S", "HSV Adjustments")
    upper_v = cv2.getTrackbarPos("Upper V", "HSV Adjustments")

    lower_hsv = np.array([lower_h, lower_s, lower_v])
    upper_hsv = np.array([upper_h, upper_s, upper_v])
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    result = cv2.bitwise_and(image, image, mask=mask)

    cv2.imshow("Mask", mask)
    cv2.imshow("Detected Megakaryocytes", result)

    if cv2.waitKey(1) & 0xFF == 27:  # Нажмите Esc, чтобы выйти
        break

cv2.destroyAllWindows()
