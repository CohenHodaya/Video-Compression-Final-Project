"""
חלק 1: טעינת תמונה והמרת צבע
"""

import numpy as np
import cv2
from PIL import Image
"""
def load_image(image_path):
    """ #טעינת תמונה מקובץ
"""
    print(f"Loading image: {image_path}")

    # טעינת התמונה
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None

    # המרה מ-BGR ל-RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Image shape: {image.shape}")

    return image
"""
image_path = rf"C:\Users\user1\Pictures\28022.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(f"Image shape: {image.shape}")
def rgb_to_ycbcr(image):
    """המרת תמונה RGB למרחב צבע YCbCr"""
    # נרמול לטווח 0-1
    image = image.astype(np.float32) / 255.0

    # מטריצת המרה
    transform_matrix = np.array([
        [0.299, 0.587, 0.114],        # Y (luminance)
        [-0.168736, -0.331264, 0.5],  # Cb (blue-difference)
        [0.5, -0.418688, -0.081312]   # Cr (red-difference)
    ])

    # ביצוע ההמרה
    ycbcr = np.dot(image, transform_matrix.T)

    # הסטת Cb ו-Cr למרכז סביב 0
    ycbcr[:, :, 1:] += 0.5

    return (ycbcr * 255).astype(np.uint8)

def ycbcr_to_rgb(image):
    """המרת תמונה YCbCr חזרה ל-RGB"""
    # נרמול והסטה
    image = image.astype(np.float32) / 255.0
    image[:, :, 1:] -= 0.5

    # מטריצת המרה הפוכה
    transform_matrix = np.array([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ])

    # ביצוע ההמרה
    rgb = np.dot(image, transform_matrix.T)

    # חיתוך לטווח תקין
    rgb = np.clip(rgb, 0, 1)

    return (rgb * 255).astype(np.uint8)

# דוגמה לשימוש
if __name__ == "__main__":
    # יצירת תמונת דוגמה
    sample_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    print("Original RGB image shape:", sample_image.shape)

    # המרה ל-YCbCr
    ycbcr_image = rgb_to_ycbcr(sample_image)
    print("YCbCr image shape:", ycbcr_image.shape)

    # המרה חזרה ל-RGB
    rgb_reconstructed = ycbcr_to_rgb(ycbcr_image)
    print("Reconstructed RGB shape:", rgb_reconstructed.shape)

    # בדיקת הפרש
    diff = np.mean(np.abs(sample_image.astype(float) - rgb_reconstructed.astype(float)))
    print(f"Mean difference after conversion: {diff:.2f}")