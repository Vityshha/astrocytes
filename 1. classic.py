import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


path = 'E:/DataSetMegakaryocytes/15_1013_HE_1/'

for file_path in glob.glob(path + '/*.png'):

    # Загрузка изображения
    image = cv2.imread(file_path)
    print(file_path)

    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение размытия для уменьшения шума
    blurred = cv2.GaussianBlur(gray, (1, 1), 15)

    # cv2.imshow('blurred', blurred)
    # cv2.waitKey(0)

    # Применение порогового преобразования для выделения фиолетовых областей
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

    # cv2.imshow('thresh', thresh)
    # cv2.waitKey(0)

    # Применение морфологических операций для удаления шумов
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)

    # Нахождение контуров на бинаризованном изображении
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Порог для количества черного цвета
    black_threshold = 100  # Вы можете настроить этот порог в зависимости от вашего изображения
    white_threshold = 180

    # Выделение крупных объектов (предположительно мегакариоцитов)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 600:  # Устанавливаем порог на основе размера области
            # Создание маски для текущего контура
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, 255, -1)  # Заполнение контура белым цветом на маске

            mean_val = cv2.mean(gray, mask=mask)[0]  # Получаем среднее значение серого цвета внутри контура

            if mean_val > black_threshold:  # Если среднее значение черного меньше порога
                if mean_val < white_threshold:
                    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

    # Отображение результата
    cv2.imshow('Image Select', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
