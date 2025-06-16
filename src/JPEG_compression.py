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
    """חלוקה לבלוקים של 8x8"""
    h, w = channel.shape
    assert h % 8 == 0 and w % 8 == 0
    return channel.reshape(h//8, 8, w//8, 8).transpose(0,2,1,3).reshape(-1, 8, 8)

def dct_2d(block):
    """חישוב DCT דו-ממדי בגודל 8x8"""
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
    """כימות בלוק DCT לפי טבלת כימות"""
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
    path = rf"C:\Users\user1\Pictures\28022.jpg"
    # שנה לשם קובץ קיים
    quantized = jpeg_quantization_process(path)
    print(f"מספר בלוקים בכימות לערוץ Y: {len(quantized[0])}")
