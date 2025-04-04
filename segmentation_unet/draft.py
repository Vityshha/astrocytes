import os
import cv2
import torch
import numpy as np
from PIL import Image
from model import UNET
import albumentations as A
from skimage import morphology
from matplotlib import pyplot as plt
from utils import load_checkpoint
from albumentations.pytorch import ToTensorV2
from skimage.segmentation import flood_fill


class ImageProcessor:
    def __init__(self, model_path, save_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = UNET(in_channels=3, out_channels=1).to(self.device)
        load_checkpoint(torch.load(model_path, map_location=self.device), self.model)
        self.model.eval()
        self.save_path = save_path
        self.transforms = A.Compose([
            A.Resize(height=720, width=720),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ])

    def find_endpoints(self, skeleton):
        """Находит конечные точки на скелете."""
        endpoints = []
        height, width = skeleton.shape
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if skeleton[y, x] == 255:
                    neighbors = sum(skeleton[y + dy, x + dx] == 255 for dy in [-1, 0, 1] for dx in [-1, 0, 1] if
                                    not (dy == 0 and dx == 0))
                    if neighbors == 1:
                        endpoints.append((x, y))
        print(f"Найдено конечных точек: {len(endpoints)}")
        return endpoints

    def find_center_point(self, skeleton):
        """Находит центральную точку на скелете."""
        height, width = skeleton.shape
        max_neighbors = -1
        candidate_points = []
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if skeleton[y, x] == 255:
                    neighbors = sum(skeleton[y + dy, x + dx] == 255 for dy in [-1, 0, 1] for dx in [-1, 0, 1] if
                                    not (dy == 0 and dx == 0))
                    if neighbors > max_neighbors:
                        max_neighbors = neighbors
                        candidate_points = [(x, y)]
                    elif neighbors == max_neighbors:
                        candidate_points.append((x, y))
        if candidate_points:
            center_x = int(np.mean([point[0] for point in candidate_points]))
            center_y = int(np.mean([point[1] for point in candidate_points]))
            return (center_x, center_y)
        return None

    def overlay_mask(self, image, mask, color=(0, 0, 255), alpha=0.5):
        """Наложение маски на изображение с заданным цветом и прозрачностью."""
        mask = mask.astype(np.uint8)
        colored_mask = np.zeros_like(image)
        colored_mask[mask == 255] = color
        result = image.copy()
        result[mask == 255] = cv2.addWeighted(image[mask == 255], 1 - alpha, colored_mask[mask == 255], alpha, 0)
        return result

    def find_brightest_point(self, image, mask):
        """Находит самую яркую точку внутри маски."""
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        gray_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        _, max_val, _, max_loc = cv2.minMaxLoc(gray_image)
        return max_loc

    def region_growing(self, image, mask, tolerance=10):
        """Выполняет Region Growing, начиная с самой яркой точки внутри маски."""
        brightest_point = self.find_brightest_point(image, mask)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filled_image = flood_fill(gray_image, (brightest_point[1], brightest_point[0]), new_value=255,
                                  tolerance=tolerance)
        region_mask = (filled_image == 255).astype(np.uint8) * 255
        return region_mask

    def check_neighborhood(self, skeleton_pixels, main_body_astrocytes, body_pixels, branch_pixels, max_distance=0):
        """
        Проверяет пиксели с учетом диапазона в диапазоне max_distance пикселей,
        и добавляет их в body_pixels или branch_pixels в зависимости от того, есть ли тело в окрестности.
        """
        for pixel in skeleton_pixels:
            y, x = pixel
            is_body_pixel = False

            for dy in range(-max_distance, max_distance + 1):
                for dx in range(-max_distance, max_distance + 1):
                    ny, nx = y + dy, x + dx

                    if 0 <= ny < main_body_astrocytes.shape[0] and 0 <= nx < main_body_astrocytes.shape[1]:
                        if main_body_astrocytes[ny, nx] == 255:
                            is_body_pixel = True
                            break

            if is_body_pixel:
                body_pixels.append((y, x))
            else:
                branch_pixels.append((y, x))

        return body_pixels, branch_pixels

    def preprocess_image(self, image_np):
        """Предобработка изображения: контрастирование, эквализация гистограммы, сглаживание, гамма-коррекция."""
        image_np = cv2.normalize(image_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)
        channels = list(cv2.split(image_np))
        channels[0] = cv2.equalizeHist(channels[0])
        image_np = cv2.merge(channels)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_YCrCb2RGB)
        image_np = cv2.GaussianBlur(image_np, (5, 5), sigmaX=1.5)
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        image_np = cv2.LUT(image_np, table)
        return image_np

    def postprocess_mask(self, mask):
        """Постобработка маски: морфологические операции и фильтрация контуров."""
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(mask)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(filtered_mask, [max_contour], -1, 255, thickness=cv2.FILLED)
        return filtered_mask

    def refine_mask(self, image_np, filtered_mask, tolerance):
        """Уточнение маски через Region Growing."""
        refined = self.region_growing(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), filtered_mask, tolerance=tolerance)
        contours, hierarchy = cv2.findContours(refined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_contour_mask = np.zeros_like(refined)
        if contours:
            max_contour_index = -1
            max_area = 0
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    max_contour_index = i
            max_contour = contours[max_contour_index]
            cv2.drawContours(max_contour_mask, [max_contour], -1, 255, thickness=cv2.FILLED)
            for i, cnt in enumerate(contours):
                if hierarchy[0][i][3] == max_contour_index:
                    cv2.drawContours(max_contour_mask, [cnt], -1, 0, thickness=cv2.FILLED)
        refined_mask = max_contour_mask
        kernel = np.ones((3, 3), np.uint8)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        return refined_mask

    def extract_main_body(self, refined_mask):
        """Извлечение основного тела астроцита."""
        kernel_size = 15
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        opened = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel, iterations=3)
        main_body = cv2.morphologyEx(opened, cv2.MORPH_DILATE, kernel, iterations=1)
        contours, _ = cv2.findContours(main_body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_body_astrocytes = np.zeros_like(main_body)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(main_body_astrocytes, [max_contour], -1, 255, thickness=cv2.FILLED)
        return main_body_astrocytes

    def find_branch_endpoints(self, branch_pixels, body_pixels, skeleton):
        """Находит концы ветвей на скелете."""
        branch_endpoints = []
        for pixel in branch_pixels:
            y, x = pixel
            # Проверяем, есть ли среди соседей пиксель из тела
            has_body_neighbor = any(
                (y + dy, x + dx) in body_pixels
                for dy in [-1, 0, 1]
                for dx in [-1, 0, 1]
                if not (dy == 0 and dx == 0)
            )
            if has_body_neighbor:
                branch_endpoints.append((x, y))

        print(f"Найдено концов ветвей: {len(branch_endpoints)}")
        return branch_endpoints

    def separate_branches(self, branch_pixels, skeleton):
        """
        Разделяет ветки на отдельные группы (связные компоненты).
        Возвращает список веток, где каждая ветка — это список пикселей (y, x).
        """
        # Создаем пустое изображение для веток
        branch_image = np.zeros_like(skeleton, dtype=np.uint8)

        # Заполняем пиксели веток на изображении
        for pixel in branch_pixels:
            y, x = pixel
            branch_image[y, x] = 255  # Помечаем пиксели веток белым цветом

        # Находим связные компоненты
        num_labels, labels = cv2.connectedComponents(branch_image, connectivity=8)

        # Собираем пиксели каждой ветки в отдельный список
        branches = []
        for label in range(1, num_labels):  # Пропускаем фон (label = 0)
            branch_pixels = np.argwhere(labels == label)  # Получаем пиксели текущей ветки
            branches.append(branch_pixels.tolist())  # Добавляем в список веток

        print(f"Найдено веток: {len(branches)}")
        return branches

    def find_endpoints_for_branch(self, branch, skeleton):
        """
        Находит количество кончиков (endpoints), которые касаются ветки.
        Вход:
            - branch: список пикселей ветки [(y1, x1), (y2, x2), ...].
            - skeleton: изображение скелета (бинарное, где скелет = 255).
        Возвращает:
            - Количество кончиков, касающихся ветки.
        """
        endpoints_count = 0
        height, width = skeleton.shape

        for pixel in branch:
            y, x = pixel
            # Проверяем, является ли текущий пиксель кончиком
            neighbors = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue  # Пропускаем текущий пиксель
                    ny, nx = y + dy, x + dx
                    # Проверяем, что координаты не выходят за границы
                    if 0 <= ny < height and 0 <= nx < width:
                        if skeleton[ny, nx] == 255:
                            neighbors += 1
            if neighbors == 1:  # Если у пикселя только один сосед, это кончик
                endpoints_count += 1

        return endpoints_count

    def process_image(self, image_path, tolerance, index):
        """Основной метод обработки изображения."""
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        image_np = np.array(image)
        image_np_origin = np.copy(image_np)

        image_np = self.preprocess_image(image_np)

        transformed = self.transforms(image=image_np)
        image_tensor = transformed["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            output = torch.sigmoid(output)
            output = (output > 0.2).float()

        output_image = output.squeeze().cpu().numpy()
        output_image = (output_image * 255).astype(np.uint8)
        output_image = cv2.resize(output_image, original_size, interpolation=cv2.INTER_NEAREST)

        filtered_mask = self.postprocess_mask(output_image)

        refined_mask = self.refine_mask(image_np, filtered_mask, tolerance)

        main_body_astrocytes = self.extract_main_body(refined_mask)

        skeleton = morphology.skeletonize(refined_mask // 255, method='lee')
        skeleton = (skeleton * 255).astype(np.uint8)

        endpoints = self.find_endpoints(skeleton)

        skeleton_pixels = np.argwhere(skeleton == 255)
        body_pixels = []
        branch_pixels = []
        body_pixels, branch_pixels = self.check_neighborhood(skeleton_pixels, main_body_astrocytes, body_pixels,
                                                             branch_pixels, max_distance=0)

        result_image = image_np_origin.copy()
        for pixel in body_pixels:
            result_image[pixel[0], pixel[1]] = [255, 0, 0]  # Красный (тело)
        for pixel in branch_pixels:
            result_image[pixel[0], pixel[1]] = [0, 255, 0]  # Зеленый (ветви)
        if endpoints:
            for endpoint in endpoints:
                cv2.circle(result_image, endpoint, radius=3, color=(0, 255, 255),
                           thickness=-1)  # Голубой (конечные точки)

        branches = self.separate_branches(branch_pixels, skeleton)

        # Итерируемся по каждой ветке
        for i, branch in enumerate(branches):
            endpoints_count = self.find_endpoints_for_branch(branch, skeleton)
            print(f"Количество кончиков для ветки {i + 1}: {endpoints_count}")
            # Если у ветки нет кончиков, считаем ее телом
            if endpoints_count == 0:
                for pixel in branch:
                    y, x = pixel
                    result_image[y, x] = [255, 0, 0]  # Перекрашиваем в красный (тело)
                    body_pixels.append((y, x))  # Добавляем в список пикселей тела

        # Находим концы ветвей
        # branch_endpoints = self.find_branch_endpoints(branch_pixels, body_pixels, skeleton)

        # Отображаем концы ветвей на изображении
        # for endpoint in branch_endpoints:
        # cv2.circle(result_image, endpoint, radius=3, color=(255, 0, 255), thickness=3)  # Фиолетовый (концы ветвей)

        self.save_images(
            original=image_np_origin,
            model_mask=self.overlay_mask(image_np_origin, mask=output_image),
            filtered_mask=self.overlay_mask(image_np_origin, mask=filtered_mask),
            refined_mask=self.overlay_mask(image_np_origin, mask=refined_mask),
            result_image=result_image,
            index=index
        )

    def display_images(self, original, model_mask, filtered_mask, refined_mask, result_image):
        """Отображение всех изображений."""
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(model_mask, cmap='gray')
        axes[1].set_title("Model Mask")
        axes[1].axis("off")

        axes[2].imshow(filtered_mask, cmap='gray')
        axes[2].set_title("Filtered Mask")
        axes[2].axis("off")

        axes[3].imshow(refined_mask, cmap='gray')
        axes[3].set_title("Refined Mask (Region Growing)")
        axes[3].axis("off")

        axes[4].imshow(cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        axes[4].set_title("Result with Skeleton and Points")
        axes[4].axis("off")

        plt.show()

    def save_images(self, original, model_mask, filtered_mask, refined_mask, result_image, index=0):
        """Сохранение всех изображений."""
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        cv2.imwrite(os.path.join(self.save_path, f"original_image_{index}.png"),
                    cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(self.save_path, f"model_mask_{index}.png"), model_mask)
        cv2.imwrite(os.path.join(self.save_path, f"filtered_mask_{index}.png"), filtered_mask)
        cv2.imwrite(os.path.join(self.save_path, f"refined_mask_{index}.png"), refined_mask)
        cv2.imwrite(os.path.join(self.save_path, f"result_image_{index}.png"),
                    cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    input_dir = "data/val_images"

    processor = ImageProcessor(model_path="saved_models/my_checkpoint_46.pth.tar", save_path="algo_results2/")
    index = 0
    tolerance = 50

    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Файл {file} не является изображением. Пропускаем.")
                continue
            input_path = os.path.join(root, file)
            print(f"Обработка изображения: {input_path}")
            processor.process_image(input_path, tolerance, index)
            index += 1
    print("Обработка завершена.")