import os
import cv2
import torch
import numpy as np
from PIL import Image
from model import UNET
import albumentations as A
from skimage import morphology
from utils import load_checkpoint
from albumentations.pytorch import ToTensorV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

IMAGE_HEIGHT = 720
IMAGE_WIDTH = 720
DOORSTEP = 0.2

MODEL_NUM = '46'
MODEL_PATH = f"saved_models/my_checkpoint_{MODEL_NUM}.pth.tar"
SAVE_PATH = 'algo_results/'

model = UNET(in_channels=3, out_channels=1).to(DEVICE)
load_checkpoint(torch.load(MODEL_PATH, map_location=DEVICE), model)
model.eval()

transforms = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])

def find_endpoints(skeleton):
    """Находит конечные точки скелета."""
    endpoints = []
    height, width = skeleton.shape

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if skeleton[y, x] == 255:
                # Считаем количество соседей
                neighbors = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue  # Пропускаем текущий пиксель
                        if skeleton[y + dy, x + dx] == 255:
                            neighbors += 1
                # Если только один сосед, это конечная точка
                if neighbors == 1:
                    endpoints.append((x, y))

    print(f"Найдено конечных точек: {len(endpoints)}")

    return endpoints

def find_center_point(skeleton):
    """Находит точку скелета с наибольшим количеством соседей. Если таких точек несколько, вычисляет их центр масс."""
    height, width = skeleton.shape
    max_neighbors = -1
    candidate_points = []  # Точки с максимальным количеством соседей

    # Проходим по каждому пикселю скелета
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if skeleton[y, x] == 255:  # Если это пиксель скелета
                # Считаем количество соседей
                neighbors = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue  # Пропускаем текущий пиксель
                        if skeleton[y + dy, x + dx] == 255:
                            neighbors += 1

                # Если количество соседей больше текущего максимума
                if neighbors > max_neighbors:
                    max_neighbors = neighbors
                    candidate_points = [(x, y)]  # Начинаем новый список
                elif neighbors == max_neighbors:
                    candidate_points.append((x, y))  # Добавляем точку в список

    # Если найдены точки с максимальным количеством соседей
    if candidate_points:
        # Вычисляем центр масс всех кандидатов
        x_coords = [point[0] for point in candidate_points]
        y_coords = [point[1] for point in candidate_points]
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        return (center_x, center_y)
    else:
        return None  # Если точек не найдено

def find_skeleton_center(endpoints):
    """Находит центр масс скелета относительно всех конечных точек."""
    if not endpoints:
        return None

    x_coords = [point[0] for point in endpoints]
    y_coords = [point[1] for point in endpoints]
    center_x = int(np.mean(x_coords))
    center_y = int(np.mean(y_coords))

    return (center_x, center_y)

def process_image(image_path, output_path='draft_out.png'):

    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image = np.array(image)
    image_np = image.copy()

    transformed = transforms(image=image)
    image_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output)
        output = (output > DOORSTEP).float()

    output_image = output.squeeze().cpu().numpy()
    output_image = (output_image * 255).astype(np.uint8)

    output_image = cv2.resize(output_image, original_size, interpolation=cv2.INTER_NEAREST)

    _, mask = cv2.threshold(output_image, 127, 255, cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(filtered_mask, [max_contour], -1, 255, thickness=cv2.FILLED)

    result = filtered_mask

    # Разделение тела и ветвей
    image_gray = result
    _, binary_image = cv2.threshold(image_gray, 127, 255, cv2.THRESH_OTSU)
    binary_image = cv2.dilate(binary_image, np.ones((5, 5), np.uint8), iterations=1)
    binary_image = cv2.erode(binary_image, np.ones((3, 3), np.uint8), iterations=1)

    # Скелетизация
    skeleton = morphology.skeletonize(binary_image // 255)
    skeleton = (skeleton * 255).astype(np.uint8)  # Преобразуем в формат для OpenCV

    # Нахождение конечных точек скелета
    endpoints = find_endpoints(skeleton)

    # Нахождение центра скелета (точки, где ветви сходятся)
    center_point = find_center_point(skeleton)

    # Разделение по радиусу
    skeleton_pixels = np.argwhere(skeleton == 255)
    if center_point:
        distances = np.linalg.norm(skeleton_pixels - np.array(center_point[::-1]), axis=1)
        threshold_radius = np.percentile(distances, 35)
        body_pixels = skeleton_pixels[distances <= threshold_radius]
        branch_pixels = skeleton_pixels[distances > threshold_radius]
    else:
        body_pixels = []
        branch_pixels = []

    # Окрашивание на исходном изображении
    for pixel in body_pixels:
        image_np[pixel[0], pixel[1]] = [255, 0, 0]  # Красный (тело)
    for pixel in branch_pixels:
        image_np[pixel[0], pixel[1]] = [0, 255, 0]  # Зеленый (ветви)
    if center_point:
        cv2.circle(image_np, center_point, radius=3, color=(255, 255, 0), thickness=-1)
    if endpoints:
        for endpoint in endpoints:
            cv2.circle(image_np, endpoint, radius=3, color=(0, 255, 255), thickness=-1)

    cv2.imwrite(output_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))



if __name__ == '__main__':
    input_dir = "data/val_images/"
    for root, _, files in os.walk(input_dir):
        for file in files:
            input_path = os.path.join(root, file)
            try:
                process_image(input_path, output_path=SAVE_PATH + file)
            except Exception as e:
                print(e)