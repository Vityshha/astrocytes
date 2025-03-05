import os
import shutil
import numpy as np
import cv2
from pylibCZIrw import czi as pyczi

def force_create_dir(dir_name):
    # Check if the directory already exists
    if os.path.exists(dir_name):
        # If it exists, remove the directory and its contents
        shutil.rmtree(dir_name)
    # Create the directory
    os.makedirs(dir_name)


def czi_get_layer_channel(czidoc, z, ch):
    # z_layers = czidoc.total_bounding_box["Z"][1]
    # channels = czidoc.total_bounding_box["C"][1]
    img = czidoc.read(plane={"T": 0, "Z": z, "C": ch})
    img = (img.astype('float32')/np.max(img) * 255).astype('uint8')
    img = cv2.equalizeHist(img)
    return img