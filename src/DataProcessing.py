import os
import numpy as np
from PIL import Image
from tensorflow.keras import layers, models
import cv2



def compute_optical_flow(im1, im3):

    gray1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    gray3 = cv2.cvtColor(im3, cv2.COLOR_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray3, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow  # shape: (H, W, 2)

def process_images(parent_folder):
    """Extract im1, im3, compute optical flow, use all as X; im2 as Y."""
    x_data, y_data = [], []

    for folder in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder)
        if os.path.isdir(folder_path):
            img1_path, img2_path, img3_path = [os.path.join(folder_path, f"im{i}.png") for i in [1, 2, 3]]
            if all(os.path.exists(p) for p in [img1_path, img2_path, img3_path]):
                im1 = np.array(Image.open(img1_path).convert("RGB").resize((256, 256)))
                im2 = np.array(Image.open(img2_path).convert("RGB").resize((256, 256)))
                im3 = np.array(Image.open(img3_path).convert("RGB").resize((256, 256)))

                flow = compute_optical_flow(im1, im3)
                combined_input = np.concatenate([im1, im3, flow], axis=-1)  # (256, 256, 8)

                x_data.append(combined_input)
                y_data.append(im2)

    return np.array(x_data), np.array(y_data)
