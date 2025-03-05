import os
import torch
import numpy as np
import cv2
from PIL import Image
from skimage import morphology
from safetensors.torch import load_file
from ben2 import BEN_Base
import traceback

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"DEVICE: {device}")

input_dir = "../datasets/astrocytes_new_data"
output_dir = "results"

model_path = "model.safetensors"

weights = load_file(model_path)
model = BEN_Base()
model.load_state_dict(weights)
model.to(device).eval()


def process_image(image_path, output_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        foreground = model.inference(image, refine_foreground=False)
        foreground = np.array(foreground)

        # Очистка
        b, g, r, a = cv2.split(foreground)
        _, mask = cv2.threshold(a, 127, 255, cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # Фильтрация по площади
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 500
        filtered_mask = np.zeros_like(mask)
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

        b = cv2.bitwise_and(b, filtered_mask)
        g = cv2.bitwise_and(g, filtered_mask)
        r = cv2.bitwise_and(r, filtered_mask)
        a = cv2.bitwise_and(a, filtered_mask)
        result = cv2.merge((b, g, r, a))

        # Разделение тела и ветвей
        image_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(image_gray, 127, 255, cv2.THRESH_OTSU)
        binary_image = cv2.dilate(binary_image, np.ones((5, 5), np.uint8), iterations=1)
        binary_image = cv2.erode(binary_image, np.ones((3, 3), np.uint8), iterations=1)

        # Центр масс
        white_pixels = np.argwhere(binary_image == 255)
        center_of_mass = white_pixels.mean(axis=0).astype(int)[::-1]

        # Скелетизация
        skeleton = morphology.skeletonize(binary_image // 255)
        skeleton_pixels = np.argwhere(skeleton == 1)

        # Разделение по радиусу
        distances = np.linalg.norm(skeleton_pixels - np.array(center_of_mass[::-1]), axis=1)
        threshold_radius = np.percentile(distances, 35)
        body_pixels = skeleton_pixels[distances <= threshold_radius]
        branch_pixels = skeleton_pixels[distances > threshold_radius]

        # Окрашивание на исходном изображении
        for pixel in body_pixels:
            image_np[pixel[0], pixel[1]] = [255, 0, 0]  # Красный (тело)
        for pixel in branch_pixels:
            image_np[pixel[0], pixel[1]] = [0, 255, 0]  # Зеленый (ветви)
        cv2.circle(image_np, tuple(center_of_mass), radius=5, color=(255, 255, 0), thickness=-1)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    except Exception as e:
        print(f"Ошибка при обработке файла {image_path}: {e}")
        print(traceback.format_exc())


for root, _, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, relative_path)
            print(f"Processing: {input_path} → {output_path}")
            try:
                process_image(input_path, output_path)
            except Exception as e:
                print(f"Ошибка при обработке файла {input_path}: {e}")
                print(traceback.format_exc())