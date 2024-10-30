import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

'''
Детектирование мегакариоцитов с помощью классических методов из opencv
'''


# path = 'E:/DataSetMegakaryocytes/15_1013_HE_1/'
path = 'C:\\Users\\Lab\\Desktop\\megakaryocytes\\data'

for file_path in glob.glob(path + '/*.png'):

    image = cv2.imread(file_path)
    print(file_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (1, 1), 15)

    # cv2.imshow('blurred', blurred)
    # cv2.waitKey(0)

    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

    # cv2.imshow('thresh', thresh)
    # cv2.waitKey(0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)

    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Порог для количества черного и белого цвета
    black_threshold = 100
    white_threshold = 180

    # Пороги для площади
    min_area = 600

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, 255, -1)

            mean_val = cv2.mean(gray, mask=mask)[0]

            if mean_val > black_threshold:
                if mean_val < white_threshold:
                    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

    cv2.imshow('magakaryocytes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
