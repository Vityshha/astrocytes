import cv2
import numpy as np
import glob

path = 'C:\\Users\\Lab\\Desktop\\megakaryocytes\\data'
# path = 'E:/DataSetMegakaryocytes/15_1013_HE_1/'


lower_hsv = np.array([20, 54, 0])
upper_hsv = np.array([179, 255, 255])

for file_path in glob.glob(path + '/*.png'):
    image = cv2.imread(file_path)

    image_copy = cv2.GaussianBlur(image, (1, 1), 2)

    hsv = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel, iterations=2)

    cv2.imshow('mask', mask)
    cv2.waitKey(0)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 400
    max_area = 4000
    black_threshold = 100
    white_threshold = 180
    convexity_threshold = 0.6

    for contour in contours:
        area = cv2.contourArea(contour)

        if min_area < area < max_area:
            contour_mask = np.zeros_like(mask)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            mean_val = cv2.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mask=contour_mask)[0]

            if black_threshold < mean_val < white_threshold:
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)

                if hull_area > 0:
                    convexity_ratio = area / hull_area
                    if convexity_ratio >= convexity_threshold:
                        # Проверка формы на эксцентриситет
                        moments = cv2.moments(contour)
                        if moments["m00"] != 0:
                            x_center = int(moments["m10"] / moments["m00"])
                            y_center = int(moments["m01"] / moments["m00"])
                            _, (minor_axis, major_axis), _ = cv2.fitEllipse(contour)
                            eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)

                            # if eccentricity < 0.8:  # Значение для круговых форм
                            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

    cv2.imshow('Detected Megakaryocytes in HSV and LAB', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
