import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET
from utils import load_checkpoint, get_val_loader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 720
doorstep = 0.2
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"

# Загрузка модели
model = UNET(in_channels=3, out_channels=1).to(DEVICE)
load_checkpoint(torch.load("saved_models/my_checkpoint.pth.tar"), model)
model.eval()

# Трансформации для валидационных изображений
val_transforms = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])

# Загрузка данных
val_loader = get_val_loader(
    VAL_IMG_DIR, VAL_MASK_DIR,
    batch_size=1, val_transform=val_transforms, num_workers=0, pin_memory=True
)


# Функция для наложения маски на изображение
def overlay_mask(image, mask, alpha=0.5):
    mask = mask.squeeze().cpu().numpy()
    image = image.squeeze().permute(1, 2, 0).cpu().numpy()

    # Денормализация изображения
    image = (image * 255).astype(np.uint8)

    # Бинаризация маски
    mask = (mask > doorstep).astype(np.uint8)

    # Создание красной маски
    red_mask = np.zeros_like(image)
    red_mask[mask == 1] = [0, 0, 255]

    result = image.copy()
    result[mask == 1] = cv2.addWeighted(image[mask == 1], 1 - alpha, red_mask[mask == 1], alpha, 0)

    return result


# Проход по валидационным данным
for idx, (data, _) in enumerate(val_loader):
    data = data.to(DEVICE)
    with torch.no_grad():
        preds = torch.sigmoid(model(data))
        preds = (preds > doorstep).float()

    try:
        # Наложение маски на изображение
        result = overlay_mask(data, preds)

        # Оригинальное изображение (без маски)
        original_image = data.squeeze().permute(1, 2, 0).cpu().numpy()
        original_image = (original_image * 255).astype(np.uint8)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Преобразование результата в RGB
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        # Создание фигуры для отображения двух изображений рядом
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(result)
        axes[1].set_title("Image with Mask")
        axes[1].axis("off")

        # Сохранение результата
        plt.savefig(f"results/result_{idx}.png", bbox_inches="tight", pad_inches=0)
        plt.show()
    except:
        pass