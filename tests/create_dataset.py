from tqdm import tqdm
from utils import *
from scipy.stats import norm
from skimage.restoration import estimate_sigma
import bm3d
import hashlib
import time


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


def generate_unique_filename(base_filename, layer, czi_path):
    """
    Генерирует уникальное имя файла на основе базового имени, слоя и пути к CZI файлу.
    """
    unique_string = f"{base_filename}_{layer}_{czi_path}_{time.time()}"
    hash_object = hashlib.sha256(unique_string.encode())
    hash_hex = hash_object.hexdigest()[:8]  # Первые 8 символов хэша
    return f"{base_filename}_{hash_hex}"


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


def get_segm_one_layer(_img, output_folder, base_filename, image_path='/train_images', mask_path='/train_masks'):
    """
    Применяет сегментацию на одном слое изображения, включая шумопонижение и фильтрацию контуров.
    Сохраняет оригинальное изображение и маску в указанную папку с синхронизированными именами.

    :param _img: Входное изображение (0..255).
    :param output_folder: Папка для сохранения изображений.
    :param base_filename: Базовое имя для файла.
    :return: Изображение с выделенными сегментами.
    """
    # Применяем шумопонижение
    _img = denoise_bm3d(_img)

    # Применяем пороговое значение
    mean, std = norm.fit(_img.flatten())
    th = mean + std + 15
    _, img = cv2.threshold(_img, th, 255, cv2.THRESH_BINARY)

    # Морфологическая операция для удаления шумов
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # Фильтрация контуров
    img = filter_contours(img)

    # Сохранение оригинального изображения и маски
    original_filename = os.path.join(output_folder + image_path, f"{base_filename}_original.png")
    mask_filename = os.path.join(output_folder + mask_path, f"{base_filename}_mask.png")

    cv2.imwrite(original_filename, _img)
    cv2.imwrite(mask_filename, img)

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


def process_czi_image(czi_path, output_folder):
    """
    Обрабатывает CZI файл и сохраняет сегментированные маски и оригинальные изображения.

    :param czi_path: Путь к файлу CZI.
    :param output_folder: Папка для сохранения результатов.
    """
    height, width = czi_get_hw(czi_path)

    with pyczi.open_czi(czi_path) as czi_file:
        z_layers = czi_file.total_bounding_box["Z"][1]

        # Обработка каждого слоя изображения
        for i in tqdm(range(z_layers)):
            img = czi_get_layer_channel(czi_file, i, 1)
            img = cv2.equalizeHist(img)  # Эквализация гистограммы для улучшения контраста
            base_filename = f"{os.path.basename(czi_path)}_layer_{i + 1}"

            # Сегментация и сохранение маски
            unique_filename = generate_unique_filename(base_filename, i, czi_path)
            get_segm_one_layer(img, output_folder, unique_filename)


def main():
    """
    Основная функция для запуска обработки CZI файла и сохранения сегментации.
    """
    output_folder = "output"
    image_path = '/train_images'
    mask_path = '/train_masks'
    os.makedirs(output_folder + image_path, exist_ok=True)
    os.makedirs(output_folder + mask_path, exist_ok=True)


    input_dir =  "D:\\astrocytes\\астроциты_новые_данные\\"
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith((".czi")):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                try:
                    process_czi_image(input_dir+relative_path, output_folder)
                except Exception as e:
                    print(e)


if __name__ == "__main__":
    main()
