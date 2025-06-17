"""
import numpy as np
from PIL import Image
from math import cos, pi, sqrt

# =======================
# קבועים – טבלאות כימות סטנדרטיות של JPEG
# =======================
LUMINANCE_Q = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
], dtype=np.float32)

CHROMINANCE_Q = np.array([
    [17,18,24,47,99,99,99,99],
    [18,21,26,66,99,99,99,99],
    [24,26,56,99,99,99,99,99],
    [47,66,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99]
], dtype=np.float32)

# =======================
# פונקציות עזר
# =======================

def rgb_to_ycbcr(img):
    """המרת תמונה מ-RGB ל-YCbCr לפי התקן של JPEG"""
    img = img.astype(np.float32)
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    Y  =  0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
    Cr =  0.5 * R - 0.4187 * G - 0.0813 * B + 128
    return np.stack([Y, Cb, Cr], axis=-1)

def block_split(channel):
    """
#חלוקה לבלוקים של 8x8

"""
    h, w = channel.shape
    assert h % 8 == 0 and w % 8 == 0
    return channel.reshape(h//8, 8, w//8, 8).transpose(0,2,1,3).reshape(-1, 8, 8)

def dct_2d(block):
    """
#חישוב DCT דו-ממדי בגודל 8x8

"""
    N = 8
    result = np.zeros((N, N), dtype=np.float32)
    for u in range(N):
        for v in range(N):
            sum_ = 0
            for x in range(N):
                for y in range(N):
                    sum_ += block[x][y] * \
                            cos((2*x + 1)*u*pi / (2*N)) * \
                            cos((2*y + 1)*v*pi / (2*N))
            cu = sqrt(1/2) if u == 0 else 1
            cv = sqrt(1/2) if v == 0 else 1
            result[u][v] = 0.25 * cu * cv * sum_
    return result

def quantize_block(dct_block, q_table):
    """
#כימות בלוק DCT לפי טבלת כימות
"""
    return np.round(dct_block / q_table).astype(np.int32)

# =======================
# תהליך מלא על תמונה
# =======================

def jpeg_quantization_process(image_path):
    # שלב 1: קריאת תמונה
    image = Image.open(image_path).convert('RGB')
    image = image.resize((256, 256))  # גודל קבוע לצורך הדגמה
    img_np = np.array(image)

    # שלב 2: המרה ל-YCbCr
    ycbcr_img = rgb_to_ycbcr(img_np) - 128  # הורדת DC bias

    # שלב 3: חלוקה לבלוקים וחישוב DCT + כימות
    result_channels = []
    for ch in range(3):
        channel = ycbcr_img[:, :, ch]
        blocks = block_split(channel)
        dct_blocks = [dct_2d(block) for block in blocks]
        q_table = LUMINANCE_Q if ch == 0 else CHROMINANCE_Q
        quantized_blocks = [quantize_block(block, q_table) for block in dct_blocks]
        result_channels.append(quantized_blocks)

    return result_channels  # רשימות של בלוקים כמותים לכל ערוץ

# =======================
# דוגמה להרצה
# =======================

if __name__ == '__main__':
    path = rf"C:
    \Pictures\WIN_20250604_00_09_30_Pro.jpg"
    # שנה לשם קובץ קיים
    quantized = jpeg_quantization_process(path)
    print(f"מספר בלוקים בכימות לערוץ Y: {len(quantized[0])}")
"""
from PIL import Image
import numpy as np
import os
from math import cos, pi, sqrt

# ========== (הטבלאות הקיימות נשמרות ללא שינוי - LUMINANCE_Q, CHROMINANCE_Q, וכו') ==========

# פונקציה הפוכה ל-DCT
def idct_2d(block):
    N = 8
    result = np.zeros((N, N), dtype=np.float32)
    for x in range(N):
        for y in range(N):
            sum_ = 0
            for u in range(N):
                for v in range(N):
                    cu = sqrt(1/2) if u == 0 else 1
                    cv = sqrt(1/2) if v == 0 else 1
                    sum_ += cu * cv * block[u][v] * \
                            cos((2*x + 1)*u*pi / (2*N)) * \
                            cos((2*y + 1)*v*pi / (2*N))
            result[x][y] = 0.25 * sum_
    return result

# חיבור בלוקים לגודל תמונה
def merge_blocks(blocks, height, width):
    h_blocks = height // 8
    w_blocks = width // 8
    merged = np.zeros((height, width), dtype=np.float32)
    idx = 0
    for i in range(h_blocks):
        for j in range(w_blocks):
            merged[i*8:(i+1)*8, j*8:(j+1)*8] = blocks[idx]
            idx += 1
    return merged

# הפונקציה המורחבת שלך:
def jpeg_quantization_and_reconstruction(image_path, output_path='output'):
    os.makedirs(output_path, exist_ok=True)

    image = Image.open(image_path).convert('RGB')
    image = image.resize((256, 256))
    img_np = np.array(image)
    ycbcr_img = rgb_to_ycbcr(img_np) - 128

    quantized_blocks_all = []
    dequantized_blocks_all = []
    reconstructed_channels = []

    for ch in range(3):
        channel = ycbcr_img[:, :, ch]
        blocks = block_split(channel)
        q_table = LUMINANCE_Q if ch == 0 else CHROMINANCE_Q

        # כימות ו־דיס-כימות
        quantized_blocks = [quantize_block(dct_2d(b), q_table) for b in blocks]
        dequantized_blocks = [block * q_table for block in quantized_blocks]
        recon_blocks = [idct_2d(b) for b in dequantized_blocks]

        # שחזור התמונה
        merged_channel = merge_blocks(recon_blocks, 256, 256)
        reconstructed_channels.append(merged_channel + 128)

        quantized_blocks_all.append(quantized_blocks)
        dequantized_blocks_all.append(dequantized_blocks)

    # שחזור RGB
    recon_ycbcr = np.stack(reconstructed_channels, axis=-1)
    recon_rgb = ycbcr_to_rgb(recon_ycbcr)
    recon_rgb = np.clip(recon_rgb, 0, 255).astype(np.uint8)

    Image.fromarray(recon_rgb).save(os.path.join(output_path, 'reconstructed_after_quantization.jpg'))
    print("✅ נשמר: reconstructed_after_quantization.jpg")

    # מסכה – איפה שיש ערכים לא אפסיים (AC שונים מ־0)
    mask_y = np.array([
        (block != 0).astype(np.uint8)
        for block in quantized_blocks_all[0]
    ])
    mask_sum = merge_blocks([np.sum(block) * np.ones((8, 8)) for block in mask_y], 256, 256)
    mask_img = np.clip((mask_sum > 0) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(mask_img).save(os.path.join(output_path, 'activity_mask_y_channel.jpg'))
    print("✅ נשמר: activity_mask_y_channel.jpg")

# המרה הפוכה YCbCr ל־RGB
def ycbcr_to_rgb(img):
    Y, Cb, Cr = img[:,:,0], img[:,:,1], img[:,:,2]
    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)
    return np.stack([R, G, B], axis=-1)

# להפעלה:
if __name__ == '__main__':
    jpeg_quantization_and_reconstruction("image.jpg", "output")
"""
import numpy as np
import cv2
import sys
sys.path.append(rf"C:\Users\user1\PycharmProjects\Image_segmentation")
from Image_segmentationTRY import segment_image


# === טבלאות קוונטיזציה ===
q_table_Y = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103,  77],
    [24, 35, 55, 64, 81, 104, 113,  92],
    [49, 64, 78, 87,103, 121, 120, 101],
    [72, 92, 95, 98,112, 100, 103,  99]
])

q_table_C = np.full((8, 8), 99)


# === פונקציות עזר ===

def Creating_a_mask(image_path, output_dir):
     mask_path = segment_image(image_path,output_dir)
     return mask_path
def rgb_to_ycbcr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

def ycbcr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)

def split_into_blocks(channel, block_size=8):
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
    padded_image = np.zeros(padded_shape, dtype=blocks[0].dtype)
    idx = 0
    for i in range(0, padded_shape[0], block_size):
        for j in range(0, padded_shape[1], block_size):
            padded_image[i:i+block_size, j:j+block_size] = blocks[idx]
            idx += 1
    return padded_image[:original_shape[0], :original_shape[1]]


# === טרנספורמציית DCT ו-IDCT ===

def dct2(block):
    return cv2.dct(block.astype(np.float32))

def idct2(block):
    return cv2.idct(block.astype(np.float32))


# === שלבי קידוד ===

def process_channel(channel, q_table):
    blocks, original_shape, padded_shape = split_into_blocks(channel)
    dct_blocks = [dct2(b) for b in blocks]
    quantized_blocks = [np.round(b / q_table).astype(np.int32) for b in dct_blocks]
    return quantized_blocks, original_shape, padded_shape

# === שלבי שחזור ===

def reconstruct_channel(quantized_blocks, q_table, original_shape, padded_shape):
    dequantized = [b * q_table for b in quantized_blocks]
    idct_blocks = [idct2(b) for b in dequantized]
    channel = blocks_to_image(idct_blocks, original_shape, padded_shape)
    return np.clip(channel, 0, 255).astype(np.uint8)


# === עיבוד תמונה שלמה ===

def jpeg_compress_and_reconstruct(image_rgb):
    image_ycbcr = rgb_to_ycbcr(image_rgb)
    y, cb, cr = cv2.split(image_ycbcr)

    q_y, orig_y, pad_y = process_channel(y, q_table_Y)
    q_cb, orig_cb, pad_cb = process_channel(cb, q_table_C)
    q_cr, orig_cr, pad_cr = process_channel(cr, q_table_C)

    y_rec = reconstruct_channel(q_y, q_table_Y, orig_y, pad_y)
    cb_rec = reconstruct_channel(q_cb, q_table_C, orig_cb, pad_cb)
    cr_rec = reconstruct_channel(q_cr, q_table_C, orig_cr, pad_cr)

    rec_ycbcr = cv2.merge([y_rec, cb_rec, cr_rec])
    rec_rgb = ycbcr_to_rgb(rec_ycbcr)
    return rec_rgb


# === דוגמת שימוש ===

if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt

    image_path = rf"C:\Users\user1\Pictures\WIN_20250604_00_09_30_Pro.jpg"
    output_dir = rf"C:\Users\user1\Pictures"
    mask_path = Creating_a_mask(image_path,output_dir)
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    compressed = jpeg_compress_and_reconstruct(image_np)

    # שמירה והשוואה
    Image.fromarray(compressed).save("compressed_output.jpg")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(image_np)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Compressed (Lossy)")
    plt.imshow(compressed)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
"""