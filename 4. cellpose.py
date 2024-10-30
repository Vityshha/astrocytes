import os
import numpy as np
import matplotlib.pyplot as plt
from cellpose import core, utils, io, models
from glob import glob
from cellpose import denoise, io
import cv2
import glob
# from sklearn.cluster import MeanShift, estimate_bandwidth
# from sklearn.cluster import DBSCAN
from skimage.measure import regionprops, label


use_GPU = core.use_gpu()
yn = ['NO', 'YES']
print(f'>>> GPU activated? {yn[use_GPU]}')

io.logger_setup()


model = denoise.CellposeDenoiseModel(gpu=True, model_type="cyto3", restore_type="deblur_cyto3")


image = cv2.imread('./data/13_54.png')

masks_pred, flows, styles, diams = model.eval(image, diameter=0, channels=[0, 0], niter=2000, flow_threshold=0.8, cellprob_threshold=0.8, min_size=600, augment=True)

clear_mask = np.zeros_like(masks_pred)
clear_mask[masks_pred != 0] = 255
clear_mask = clear_mask.astype(np.uint8)

# kernel = np.ones((3, 3), np.uint8) * 6
# clear_mask = cv2.morphologyEx(clear_mask, cv2.MORPH_OPEN, kernel, iterations=8)

# cv2.imshow('clear mask', clear_mask)
# cv2.waitKey(0)

contours, _ = cv2.findContours(clear_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Порог для количества черного и белого цвета
black_threshold = 100
white_threshold = 180

# Пороги для площади
min_area = 600

# Преобразование изображения в градации серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

for contour in contours:
    area = cv2.contourArea(contour)
    if area > min_area:
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Определение среднего значения внутри контура
        mean_val = cv2.mean(gray, mask=mask)[0]

        # Проверка: если средняя яркость ниже white_threshold, оставляем контур
        # if mean_val > white_threshold:
        #     continue
        #     # 1. Получаем область внутри контура
        #     region = cv2.bitwise_and(image, image, mask=mask)
        #     # 2. Выполняем повторную бинаризацию
        #     _, binary_region = cv2.threshold(
        #         cv2.cvtColor(region, cv2.COLOR_BGR2GRAY), 20, 255,
        #         cv2.THRESH_BINARY)
        #
        #
        #
        #     # 3. Находим контуры в бинарной области
        #     new_contours, _ = cv2.findContours(binary_region,
        #                                        cv2.RETR_EXTERNAL,
        #                                        cv2.CHAIN_APPROX_SIMPLE)
        #
        #     cv2.drawContours(region, new_contours, -1, 255, 2)
        #
        #     cv2.imshow('region', region)
        #     cv2.imshow('binary', binary_region)
        #     cv2.waitKey(0)
        #
        #     # 4. Рисуем новые контуры
        #     for new_contour in new_contours:
        #
        #         mask = np.zeros_like(gray)
        #         cv2.drawContours(mask, [new_contour], -1, 255, -1)
        #
        #         # Определение среднего значения внутри контура
        #         mean_val = cv2.mean(gray, mask=mask)[0]
        #         if mean_val < white_threshold:
        #             cv2.drawContours(image, [new_contour], -1, (255, 0, 0), 2)
        # else:
        #     pass
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

cv2.imshow('image', image)
cv2.waitKey(0)