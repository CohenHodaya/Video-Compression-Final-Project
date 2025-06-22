import numpy as np
import cv2
import os
from PIL import Image
import sys
sys.path.append(rf"C:\Users\user1\PycharmProjects\Image_segmentation")
from Image_segmentationTRY import segment_image

def Creating_a_mask(image_path, output_dir,size):
    mask_result = segment_image(image_path, output_dir)

    # אם התוצאה היא Tensor, המר אותה לתמונה
    if hasattr(mask_result, 'numpy'):  # זה Tensor
        import torch
        if isinstance(mask_result, torch.Tensor):
            # המרה מ-Tensor לnumpy array
            if mask_result.is_cuda:
                mask_array = mask_result.cpu().numpy()
            else:
                mask_array = mask_result.numpy()

            # אם יש ממד נוסף (batch), הסר אותו
            if len(mask_array.shape) == 4:
                mask_array = mask_array[0]

            # אם יש 3 ממדים והראשון הוא channels, העבר אותו לאחרון
            if len(mask_array.shape) == 3 and mask_array.shape[0] <= 4:
                mask_array = np.transpose(mask_array, (1, 2, 0))

            # המרה למסכה בינארית (0 או 255)
            if mask_array.max() <= 1.0:
                # אם הערכים בין 0-1, המר לבינארי
                mask_array = (mask_array > 0.5).astype(np.uint8) * 255
            else:
                # אם הערכים כבר בטווח 0-255, המר לבינארי
                mask_array = (mask_array > 128).astype(np.uint8) * 255

            # וידוא שהמסכה היא grayscale
            if len(mask_array.shape) == 3:
                mask_array = np.mean(mask_array, axis=2).astype(np.uint8)
                mask_array = (mask_array > 128).astype(np.uint8) * 255

            mask_pil = Image.fromarray(mask_array, mode='L')
            mask_pil = mask_pil.resize(size, Image.NEAREST)  # שמירה על מסכה חדה
#mask_array = np.array(mask_pil)

            # שמירת המסכה כקובץ
            mask_path = os.path.join(output_dir, "generated_mask.jpg")
            mask_pil.save(mask_path)
            print(f"מסכה נשמרה ב: {mask_path}")
            return mask_path
            # שמירת המסכה כקובץ
            """
            mask_path = os.path.join(output_dir, "generated_mask.jpg")
            Image.fromarray(mask_array, mode='L').save(mask_path)
            print(f"מסכה נשמרה ב: {mask_path}")
            return mask_path
"""
    # אם זה נתיב קובץ, טען אותו ווודא שהוא בינארי
    if isinstance(mask_result, str) and os.path.exists(mask_result):
        mask_img = Image.open(mask_result).convert('L')
        mask_array = np.array(mask_img)
        # המרה לבינארי
        mask_array = (mask_array > 128).astype(np.uint8) * 255
        # שמירה חזרה
        Image.fromarray(mask_array, mode='L').save(mask_result)
        print(f"מסכה נשמרה ב: {mask_result}")
        return mask_result

    return mask_result

def load_or_create_mask(image_path, output_dir, existing_mask_path=None):
    """טעינת מסכה קיימת או יצירת מסכה חדשה"""
    if existing_mask_path and os.path.exists(existing_mask_path):
        print(f"טוען מסכה קיימת מ: {existing_mask_path}")
        # טעינת המסכה הקיימת
        mask_img = Image.open(existing_mask_path).convert('L')
        mask_array = np.array(mask_img)
        # וידוא שהמסכה בינארית
        mask_array = (mask_array > 128).astype(np.uint8) * 255
        return mask_array, existing_mask_path
    else:
        print("יוצר מסכה חדשה...")
        # יצירת מסכה חדשה
        mask_path = Creating_a_mask(image_path, output_dir)
        mask_img = Image.open(mask_path).convert('L')
        mask_array = np.array(mask_img)
        return mask_array, mask_path

def prepare_mask_for_compression(mask_array, target_size):
    """הכנת המסכה לשימוש בדחיסה"""
    # וידוא שהתמונה והמסכה באותו גודל
    if (mask_array.shape[1], mask_array.shape[0]) != target_size:
        print(f"משנה גודל מסכה מ-{mask_array.shape} ל-{target_size}")
        mask_pil = Image.fromarray(mask_array, mode='L')
        mask_pil = mask_pil.resize(target_size)
        mask_array = np.array(mask_pil)

    # וידוא שהמסכה בינארית (0 או 255)
    mask_array = (mask_array > 128).astype(np.uint8) * 255

    return mask_array

def is_block_black(mask_block, threshold=128):

    return np.mean(mask_block) < threshold