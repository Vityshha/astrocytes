import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pylibCZIrw import czi as pyczi
from tqdm import tqdm
from utils import *
from scipy.stats import norm
from matplotlib.colors import to_hex

from vispy import scene, app

from sklearn.cluster import DBSCAN

from skimage.restoration import estimate_sigma
import bm3d

import open3d as o3d


def numpy_norm_image(img):
    """
    Нормализует изображение, приводя его значения пикселей к диапазону [0, 1].

    :param img: Входное изображение (numpy array).
    :return: Нормализованное изображение.
    """
    return img / np.max(img)


def numpy_image_to255(img):
    """
    Преобразует изображение из диапазона [0, 1] в диапазон [0, 255].

    :param img: Входное нормализованное изображение.
    :return: Изображение с значениями пикселей в диапазоне [0, 255].
    """
    return (img * 255).astype(np.uint8)


def denoise_bm3d(img):
    """
    Применяет метод BM3D для удаления шума на изображении.

    :param img: Входное изображение (0..255).
    :return: Шумопониженное изображение.
    """
    norm_img = numpy_norm_image(img)
    sigma_psd = estimate_sigma(norm_img, average_sigmas=True)
    img_bm3d = numpy_image_to255(bm3d.bm3d(norm_img, sigma_psd=sigma_psd, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING))
    return img_bm3d


# def filter_contours(img):
#     """
#     Находит контуры на изображении и фильтрует их по площади.
#
#     :param img: Входное бинарное изображение.
#     :return: Изображение с отфильтрованными контурами.
#     """
#     contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     contour_areas = [cv2.contourArea(cnt) for cnt in contours]
#
#     # Оценка среднего и стандартного отклонения площади контуров
#     mean, std = norm.fit(contour_areas)
#     th = mean + std  # Порог для фильтрации
#     filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > th]
#
#     blank_image = np.zeros_like(img)
#     cv2.drawContours(blank_image, filtered_contours, -1, (255, 155, 255), thickness=cv2.FILLED)
#     return blank_image

def filter_contours(img):
    """
    Находит контуры на изображении и выбирает только максимальный по площади контур.

    :param img: Входное бинарное изображение.
    :return: Изображение с отфильтрованным максимальным контуром.
    """
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Если контуры найдены
    if contours:
        # Находим контур с максимальной площадью
        max_contour = max(contours, key=cv2.contourArea)

        # Создаем пустое изображение для отображения только максимального контура
        blank_image = np.zeros_like(img)

        # Рисуем максимальный контур на пустом изображении
        cv2.drawContours(blank_image, [max_contour], -1, (255, 155, 255), thickness=cv2.FILLED)

        return blank_image
    else:
        # Если контуры не найдены, возвращаем пустое изображение
        return np.zeros_like(img)



def get_segm_one_layer(_img):
    """
    Применяет сегментацию на одном слое изображения, включая шумопонижение и фильтрацию контуров.

    :param _img: Входное изображение (0..255).
    :return: Изображение с выделенными сегментами.
    """
    _img = denoise_bm3d(_img)

    mean, std = norm.fit(_img.flatten())
    th = mean + std + 15
    _, img = cv2.threshold(_img, th, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    img = filter_contours(img)

    # Отображение исходного и сегментированного изображения
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1), plt.xticks([]), plt.yticks([]), plt.imshow(_img, cmap="gray")
    plt.subplot(1, 2, 2), plt.xticks([]), plt.yticks([]), plt.imshow(img, cmap="gray")
    plt.show()
    plt.close()

    return img


def czi_get_hw(czi_path):
    """
    Извлекает размеры изображения из файла CZI.

    :param czi_path: Путь к файлу CZI.
    :return: Высота и ширина изображения.
    """
    with pyczi.open_czi(czi_path) as czi_file:
        img = czi_get_layer_channel(czi_file, 0, 1)
        assert len(img.shape) == 2
        h, w = img.shape
    return h, w


def plot_3d_segm(czi_path):
    """
    Строит 3D сегментацию с использованием DBSCAN для кластеризации точек.

    :param czi_path: Путь к файлу CZI.
    """
    height, width = czi_get_hw(czi_path)

    with pyczi.open_czi(czi_path) as czi_file:
        z_layers = czi_file.total_bounding_box["Z"][1]
        data = np.zeros((height, width, z_layers))
        assert z_layers > 1
        for i in tqdm(range(z_layers)):
            img = czi_get_layer_channel(czi_file, i, 1)
            img = cv2.equalizeHist(img)
            img = get_segm_one_layer(img)
            img = cv2.resize(img, (height, width))
            data[:, :, i] = img

    # Извлечение координат точек из 3D массива
    x, y, z = data.nonzero()
    points = np.array([x, y, z]).T
    print(f'points shape: {points.shape}, data: {points}')

    # Кластеризация с помощью DBSCAN
    dbscan = DBSCAN(eps=4, min_samples=4, metric='euclidean', n_jobs=-1)
    labels = dbscan.fit_predict(points)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f'DBSCAN found {n_clusters} clusters')

    # Определение размеров кластеров
    if -1 in labels:
        cluster_sizes = np.bincount(labels + 1)[1:]
    else:
        cluster_sizes = np.bincount(labels)
    print(f'cluster_sizes: {cluster_sizes.shape}, data: {cluster_sizes}')

    sizesorted_cluster_idx = np.argsort(cluster_sizes)
    print(f'sizesorted_cluster_idx = {sizesorted_cluster_idx}')
    print(f'sorted sizesorted_cluster_idx = {sorted(sizesorted_cluster_idx)}')
    print(f'sizes of clusters = {cluster_sizes[sizesorted_cluster_idx]}')

    # Отображение астроцитов
    sizesorted_cluster_sz = cluster_sizes[sizesorted_cluster_idx]
    sizesorted_cluster_sz_z_scores = [(sz - np.mean(sizesorted_cluster_sz)) / np.std(sizesorted_cluster_sz) for sz in
                                      sizesorted_cluster_sz]
    astrocyte_labels = [i for z, i in zip(sizesorted_cluster_sz_z_scores, sizesorted_cluster_idx) if abs(z) > 1]
    print('Selected astrocytes indexes: ', astrocyte_labels)
    print('Selected astrocytes sizes: ', cluster_sizes[astrocyte_labels])

    # Визуализация астроцитов в 3D
    for astr_idx in astrocyte_labels:
        cluster_points_idx = np.where(labels == astr_idx)
        cluster_points = np.array([x[cluster_points_idx], y[cluster_points_idx], z[cluster_points_idx] * 3]).T

        # Создание облака точек с использованием Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cluster_points)

        o3d.visualization.draw_geometries([pcd])  # Визуализация


def main():
    """
    Основная функция для запуска обработки CZI файла и визуализации сегментации.
    """
    czi_path = "D:\\astrocytes\\астроциты_новые_данные\\Lipachev_astrocytes_2021.03.09\\2021.03.09_1.1.3_Image 6.czi"
    plot_3d_segm(czi_path)


if __name__ == "__main__":
    main()
