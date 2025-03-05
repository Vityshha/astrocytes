import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Get sorted file lists
        self.image_files = sorted(os.listdir(image_dir), key=lambda x: os.path.splitext(x)[0])
        self.mask_files = sorted(os.listdir(mask_dir), key=lambda x: os.path.splitext(x)[0])

        # Mapping image names without suffix
        self.image_dict = {os.path.splitext(f)[0].replace("_original", ""): os.path.join(image_dir, f) for f in self.image_files}
        self.mask_dict = {os.path.splitext(f)[0].replace("_mask", ""): os.path.join(mask_dir, f) for f in self.mask_files}

        # Filter matching pairs
        self.image_files = [img for img in self.image_dict.keys() if img in self.mask_dict]

        if len(self.image_files) == 0:
            print("⚠️ Error! No matching image-mask pairs found.")
            raise ValueError(f"No matching image-mask pairs in {image_dir} and {mask_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_name = self.image_files[index]
        img_path = self.image_dict[img_name]
        mask_path = self.mask_dict[img_name]

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0  # Convert mask to binary

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask
