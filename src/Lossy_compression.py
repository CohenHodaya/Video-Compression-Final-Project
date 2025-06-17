import numpy as np
import cv2
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt
sys.path.append(rf"C:\Users\user1\PycharmProjects\Image_segmentation")
from Image_segmentationTRY import segment_image

# === טבלאות קוונטיזציה ===

# טבלת קוונטיזציה בסיסית (JPEG standard)
q_table_base_Y = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103,  77],
    [24, 35, 55, 64, 81, 104, 113,  92],
    [49, 64, 78, 87,103, 121, 120, 101],
    [72, 92, 95, 98,112, 100, 103,  99]
])

q_table_base_C = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])

def create_quantization_tables(normal_quality=10, strong_quality=50):
    """
    יצירת טבלאות קוונטיזציה לפי רמת איכות

    Args:
        normal_quality (int): רמת איכות לדחיסה רגילה (1-100, 1=איכות גבוהה, 100=איכות נמוכה)
        strong_quality (int): רמת איכות לדחיסה חזקה (1-100, 1=איכות גבוהה, 100=איכות נמוכה)

    Returns:
        tuple: (q_table_Y, q_table_C, q_table_Y_strong, q_table_C_strong)
    """

    def scale_quantization_table(base_table, quality):
        """יצירת טבלת קוונטיזציה לפי רמת איכות"""
        if quality < 1:
            quality = 1
        elif quality > 100:
            quality = 100

        if quality < 50:
            scale = 5000 / quality
        else:
            scale = 200 - quality * 2

        scaled_table = (base_table * scale + 50) / 100
        scaled_table = np.clip(scaled_table, 1, 255).astype(np.int32)
        return scaled_table

    # יצירת טבלאות לדחיסה רגילה
    q_table_Y = scale_quantization_table(q_table_base_Y, normal_quality)
    q_table_C = scale_quantization_table(q_table_base_C, normal_quality)

    # יצירת טבלאות לדחיסה חזקה - כל הערוצים באותה רמה
    q_table_Y_strong = scale_quantization_table(q_table_base_Y, strong_quality)
    q_table_C_strong = scale_quantization_table(q_table_base_C, strong_quality)

    return q_table_Y, q_table_C, q_table_Y_strong, q_table_C_strong

# === פונקציות עזר ===

def Creating_a_mask(image_path, output_dir):
    """יצירת מסכה באמצעות סגמנטציה"""
    print("יוצר מסכה חדשה...")
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

            # שמירת המסכה כקובץ
            mask_path = os.path.join(output_dir, "generated_mask.jpg")
            Image.fromarray(mask_array, mode='L').save(mask_path)
            print(f"מסכה נשמרה ב: {mask_path}")
            return mask_path

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

def rgb_to_ycbcr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

def ycbcr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)

def split_into_blocks(channel, block_size=8):
    """חלוקת ערוץ לבלוקים של 8x8"""
    h, w = channel.shape
    padded_h = (h + block_size - 1) // block_size * block_size
    padded_w = (w + block_size - 1) // block_size * block_size
    padded = np.zeros((padded_h, padded_w), dtype=channel.dtype)
    padded[:h, :w] = channel
    blocks = []
    for i in range(0, padded_h, block_size):
        for j in range(0, padded_w, block_size):
            block = padded[i:i+block_size, j:j+block_size]
            blocks.append(block)
    return blocks, (h, w), (padded_h, padded_w)

def blocks_to_image(blocks, original_shape, padded_shape, block_size=8):
    """חיבור בלוקים חזרה לתמונה"""
    padded_image = np.zeros(padded_shape, dtype=blocks[0].dtype)
    idx = 0
    for i in range(0, padded_shape[0], block_size):
        for j in range(0, padded_shape[1], block_size):
            padded_image[i:i+block_size, j:j+block_size] = blocks[idx]
            idx += 1
    return padded_image[:original_shape[0], :original_shape[1]]

def is_block_black(mask_block, threshold=128):
    """בדיקה אם בלוק במסכה הוא כולו שחור (או רוב הפיקסלים שחורים)
    במסכה בינארית: שחור = 0, לבן = 255
    אם הערך הממוצע קטן מהסף - הבלוק נחשב שחור"""
    return np.mean(mask_block) < threshold

# === טרנספורמציית DCT ו-IDCT ===

def dct2(block):
    return cv2.dct(block.astype(np.float32))

def idct2(block):
    return cv2.idct(block.astype(np.float32))

# === שלבי קידוד עם מסכה ===

def process_channel_with_mask(channel, mask_channel, q_table_normal, q_table_strong):
    """עיבוד ערוץ עם בחירת טבלת קוונטיזציה לפי המסכה"""
    print(f"מעבד ערוץ בגודל: {channel.shape}")

    # חלוקה לבלוקים
    blocks, original_shape, padded_shape = split_into_blocks(channel)
    mask_blocks, _, _ = split_into_blocks(mask_channel)

    print(f"נוצרו {len(blocks)} בלוקים")

    dct_blocks = []
    quantized_blocks = []
    quantization_info = []  # מידע על איזו טבלה נבחרה לכל בלוק

    strong_compression_count = 0
    normal_compression_count = 0

    for i, (block, mask_block) in enumerate(zip(blocks, mask_blocks)):
        # חישוב DCT
        dct_block = dct2(block)
        dct_blocks.append(dct_block)

        # בחירת טבלת קוונטיזציה לפי המסכה
        if is_block_black(mask_block):
            # בלוק שחור במסכה - דחיסה חזקה
            q_table = q_table_strong
            quantization_info.append("strong")
            strong_compression_count += 1
        else:
            # בלוק לא שחור במסכה - דחיסה רגילה
            q_table = q_table_normal
            quantization_info.append("normal")
            normal_compression_count += 1

        # קוונטיזציה
        quantized_block = np.round(dct_block / q_table).astype(np.int32)
        quantized_blocks.append(quantized_block)

    print(f"דחיסה חזקה: {strong_compression_count} בלוקים, דחיסה רגילה: {normal_compression_count} בלוקים")
    return quantized_blocks, original_shape, padded_shape, quantization_info

# === שלבי שחזור עם מסכה ===

def reconstruct_channel_with_mask(quantized_blocks, mask_channel, q_table_normal, q_table_strong, original_shape, padded_shape):
    """שחזור ערוץ עם בחירת טבלת קוונטיזציה לפי המסכה"""
    mask_blocks, _, _ = split_into_blocks(mask_channel)

    idct_blocks = []
    for quantized_block, mask_block in zip(quantized_blocks, mask_blocks):
        # בחירת טבלת קוונטיזציה (אותה בחירה כמו בקידוד)
        if is_block_black(mask_block):
            q_table = q_table_strong
        else:
            q_table = q_table_normal

        # דה-קוונטיזציה
        dequantized_block = quantized_block * q_table

        # IDCT
        idct_block = idct2(dequantized_block)
        idct_blocks.append(idct_block)

    # חיבור הבלוקים לתמונה
    channel = blocks_to_image(idct_blocks, original_shape, padded_shape)
    return np.clip(channel, 0, 255).astype(np.uint8)

# === עיבוד תמונה שלמה עם מסכה ===

def jpeg_compress_and_reconstruct_with_mask(image_rgb, mask_gray, normal_quality=10, strong_quality=50):
    """
    דחיסה ושחזור של תמונה עם מסכה

    Args:
        image_rgb: תמונה RGB
        mask_gray: מסכה בגווני אפור
        normal_quality: רמת איכות לדחיסה רגילה (1-100, ככל שנמוך יותר - איכות טובה יותר)
        strong_quality: רמת איכות לדחיסה חזקה (1-100, ככל שגבוה יותר - דחיסה חזקה יותר)
    """

    # יצירת טבלאות קוונטיזציה
    print(f"יוצר טבלאות קוונטיזציה: רגילה={normal_quality}, חזקה={strong_quality}")
    q_table_Y, q_table_C, q_table_Y_strong, q_table_C_strong = create_quantization_tables(normal_quality, strong_quality)

    # המרה ל-YCbCr
    image_ycbcr = rgb_to_ycbcr(image_rgb)
    y, cb, cr = cv2.split(image_ycbcr)

    # עיבוד כל ערוץ עם המסכה
    print("מעבד ערוץ Y...")
    q_y, orig_y, pad_y, info_y = process_channel_with_mask(y, mask_gray, q_table_Y, q_table_Y_strong)

    print("מעבד ערוץ Cb...")
    q_cb, orig_cb, pad_cb, info_cb = process_channel_with_mask(cb, mask_gray, q_table_C, q_table_C_strong)

    print("מעבד ערוץ Cr...")
    q_cr, orig_cr, pad_cr, info_cr = process_channel_with_mask(cr, mask_gray, q_table_C, q_table_C_strong)

    # שחזור
    print("משחזר ערוצים...")
    y_rec = reconstruct_channel_with_mask(q_y, mask_gray, q_table_Y, q_table_Y_strong, orig_y, pad_y)
    cb_rec = reconstruct_channel_with_mask(q_cb, mask_gray, q_table_C, q_table_C_strong, orig_cb, pad_cb)
    cr_rec = reconstruct_channel_with_mask(q_cr, mask_gray, q_table_C, q_table_C_strong, orig_cr, pad_cr)

    # חיבור ערוצים והמרה חזרה ל-RGB
    rec_ycbcr = cv2.merge([y_rec, cb_rec, cr_rec])
    rec_rgb = ycbcr_to_rgb(rec_ycbcr)

    return rec_rgb, (info_y, info_cb, info_cr)

# === הצגת תוצאות ===

def visualize_quantization_map(quantization_info, original_shape, padded_shape):
    """יצירת מפת הקוונטיזציה - הצגה של איפה הופעלה דחיסה חזקה"""
    # יצירת מפה בינארית: 255 = דחיסה חזקה, 0 = דחיסה רגילה
    quantization_blocks = []
    for info in quantization_info:
        if info == "strong":
            quantization_blocks.append(np.ones((8, 8), dtype=np.uint8) * 255)
        else:
            quantization_blocks.append(np.zeros((8, 8), dtype=np.uint8))

    quantization_map = blocks_to_image(quantization_blocks, original_shape, padded_shape)
    return quantization_map

# === דוגמת שימוש ===

if __name__ == "__main__":
    # === הגדרות נתיבים - שנה כאן את הנתיבים שלך ===
    image_path = rf"C:\Users\user1\Pictures\WIN_20250604_00_09_30_Pro.jpg"  # נתיב התמונה
    output_dir = rf"C:\Users\user1\Pictures\Masks"  # תיקיית פלט למסכות

    # *** שנה את הנתיב הזה למסכה שלך או השאר None ליצירת מסכה חדשה ***
    existing_mask_path = None  # rf"C:\path\to\your\mask.jpg"  # נתיב למסכה קיימת או None ליצירת מסכה חדשה

    # === יצירת או טעינת מסכה ===
    print("=== טיפול במסכה ===")

    # טעינת התמונה המקורית
    print("טוען תמונה מקורית...")
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    existing_mask_path = rf"C:\Users\user1\Pictures\Masks\WIN_20250604_00_09_30_Pro.jpg"
    # טעינת או יצירת מסכה
    mask_np, mask_path = load_or_create_mask(image_path, output_dir, existing_mask_path)

    # וידוא שהתמונה והמסכה באותו גודל
    if image.size != (mask_np.shape[1], mask_np.shape[0]):
        print(f"משנה גודל מסכה מ-{mask_np.shape} ל-{image.size}")
        mask_pil = Image.fromarray(mask_np, mode='L')
        mask_pil = mask_pil.resize(image.size)
        mask_np = np.array(mask_pil)

    # וידוא שהמסכה בינארית (0 או 255)
    mask_np = (mask_np > 128).astype(np.uint8) * 255

    print(f"גודל תמונה: {image_np.shape}")
    print(f"גודל מסכה: {mask_np.shape}")
    print(f"ערכים במסכה: min={mask_np.min()}, max={mask_np.max()}")

    # === הגדרות דחיסה - שנה כאן את רמות הדחיסה ===
    NORMAL_QUALITY = 50   # דחיסה רגילה (1-100, ככל שנמוך יותר - איכות טובה יותר)
    STRONG_QUALITY = 30   # דחיסה חזקה (1-100, ככל שגבוה יותר - דחיסה חזקה יותר)

    print(f"הגדרות דחיסה: רגילה={NORMAL_QUALITY}, חזקה={STRONG_QUALITY}")

    # === דחיסה עם מסכה ===
    print("\n=== דחיסה עם מסכה ===")
    compressed, quantization_info = jpeg_compress_and_reconstruct_with_mask(
        image_np, mask_np, NORMAL_QUALITY, STRONG_QUALITY
    )

    # יצירת מפת הקוונטיזציה
    print("יוצר מפת קוונטיזציה...")
    quantization_map = visualize_quantization_map(quantization_info[0], image_np.shape[:2], image_np.shape[:2])

    # === שמירת תוצאות ===
    print("\n=== שמירת תוצאות ===")
    compressed_path = "compressed_with_mask.jpg"
    quantization_map_path = "quantization_map.jpg"

    Image.fromarray(compressed).save(compressed_path)
    Image.fromarray(quantization_map).save(quantization_map_path)

    print(f"תמונה דחוסה נשמרה ב: {compressed_path}")
    print(f"מפת קוונטיזציה נשמרה ב: {quantization_map_path}")

    # === הצגת תוצאות ===
    print("\n=== הצגת תוצאות ===")
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.title("תמונה מקורית")
    plt.imshow(image_np)
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title("מסכה בינארית\n(שחור = דחיסה חזקה)")
    plt.imshow(mask_np, cmap='gray')
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.title("מפת קוונטיזציה\n(לבן = דחיסה חזקה)")
    plt.imshow(quantization_map, cmap='gray')
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.title("תמונה דחוסה")
    plt.imshow(compressed)
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.title("הפרש (מקורית - דחוסה)")
    diff = np.abs(image_np.astype(np.float32) - compressed.astype(np.float32))
    plt.imshow(diff.astype(np.uint8))
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.title("הפרש בגווני אפור")
    diff_gray = np.mean(diff, axis=2)
    plt.imshow(diff_gray, cmap='hot')
    plt.colorbar()
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # === סיכום סטטיסטיקות ===
    print("\n=== סיכום ===")

    # סטטיסטיקות ערוץ Y
    strong_blocks_y = sum(1 for info in quantization_info[0] if info == "strong")
    normal_blocks_y = sum(1 for info in quantization_info[0] if info == "normal")
    total_blocks_y = len(quantization_info[0])

    # סטטיסטיקות ערוץ Cb
    strong_blocks_cb = sum(1 for info in quantization_info[1] if info == "strong")
    normal_blocks_cb = sum(1 for info in quantization_info[1] if info == "normal")
    total_blocks_cb = len(quantization_info[1])

    # סטטיסטיקות ערוץ Cr
    strong_blocks_cr = sum(1 for info in quantization_info[2] if info == "strong")
    normal_blocks_cr = sum(1 for info in quantization_info[2] if info == "normal")
    total_blocks_cr = len(quantization_info[2])

    print(f"ערוץ Y - סה\"כ בלוקים: {total_blocks_y}")
    print(f"  דחיסה חזקה (איכות {STRONG_QUALITY}): {strong_blocks_y} ({strong_blocks_y/total_blocks_y*100:.1f}%)")
    print(f"  דחיסה רגילה (איכות {NORMAL_QUALITY}): {normal_blocks_y} ({normal_blocks_y/total_blocks_y*100:.1f}%)")

    print(f"\nערוץ Cb - סה\"כ בלוקים: {total_blocks_cb}")
    print(f"  דחיסה חזקה (איכות {STRONG_QUALITY}): {strong_blocks_cb} ({strong_blocks_cb/total_blocks_cb*100:.1f}%)")
    print(f"  דחיסה רגילה (איכות {NORMAL_QUALITY}): {normal_blocks_cb} ({normal_blocks_cb/total_blocks_cb*100:.1f}%)")

    print(f"\nערוץ Cr - סה\"כ בלוקים: {total_blocks_cr}")
    print(f"  דחיסה חזקה (איכות {STRONG_QUALITY}): {strong_blocks_cr} ({strong_blocks_cr/total_blocks_cr*100:.1f}%)")
    print(f"  דחיסה רגילה (איכות {NORMAL_QUALITY}): {normal_blocks_cr} ({normal_blocks_cr/total_blocks_cr*100:.1f}%)")

    print(f"\nמסכה נטענה מ: {mask_path}")
    print("נשמרו קבצים:")
    print(f"- {compressed_path}")
    print(f"- {quantization_map_path}")