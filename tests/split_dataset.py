import os
import shutil
import random


def split_data(input_folder, output_folder, val_ratio=0.2):
    """Разделяет данные на обучающую и валидационную выборки."""
    train_images = os.path.join(input_folder, "train_images")
    train_masks = os.path.join(input_folder, "train_masks")
    val_images = os.path.join(output_folder, "val_images")
    val_masks = os.path.join(output_folder, "val_masks")

    os.makedirs(val_images, exist_ok=True)
    os.makedirs(val_masks, exist_ok=True)

    images = sorted(os.listdir(train_images))
    masks = sorted(os.listdir(train_masks))

    assert len(images) == len(masks)

    val_size = int(len(images) * val_ratio)
    val_indices = random.sample(range(len(images)), val_size)

    for idx in val_indices:
        img_name = images[idx]
        mask_name = masks[idx]

        shutil.move(os.path.join(train_images, img_name), os.path.join(val_images, img_name))
        shutil.move(os.path.join(train_masks, mask_name), os.path.join(val_masks, mask_name))

    print(f"Перемещено {val_size} изображений и масок в валидационную выборку.")


output_dir = "output"
split_data(output_dir, output_dir, val_ratio=0.2)
