import os
import czifile
import numpy as np
from PIL import Image
from tqdm import tqdm

# Папки
input_dir = "D:/astrocytes/actro_new_data"
output_dir = "../datasets"

os.makedirs(output_dir, exist_ok=True)

for root, _, files in os.walk(input_dir):
    for file in tqdm(files, desc="Processing files"):
        if file.endswith(".czi"):
            file_path = os.path.join(root, file)
            with czifile.CziFile(file_path) as czi:
                data = czi.asarray()

                if data.shape[1] < 2:
                    print(f"Файл {file} имеет только {data.shape[1]} слоя, пропускаем.")
                    continue

                second_layer = data[0, 1, 0, :, :, :, 0]  # Берем второй слой (индекс 1)

                base_name = os.path.splitext(file)[0]
                for i in range(second_layer.shape[0]):
                    img_array = second_layer[i]
                    img_array = (img_array / img_array.max() * 255).astype(np.uint8)

                    save_path = os.path.join(output_dir, f"{base_name}_img_{i}.png")
                    Image.fromarray(img_array).save(save_path)

print("Обработка завершена!")
