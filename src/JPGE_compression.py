"""
חלק 1: טעינת תמונה והמרת צבע
"""
import sys
sys.path.append(rf"C:\Users\user1\PycharmProjects\Image_segmentation")
import numpy as np
import cv2
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
from Image_segmentationTRY import segment_image


from PIL import Image
BLOCK_SIZE = 8

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

"""
חלק 2: חלוקת התמונה לבלוקים 8x8
"""

def split_into_blocks(image,mask, block_size=8):
    """חלוקת התמונה לבלוקים שאינם חופפים"""
    height, width = image.shape[:2]

    # חישוב כמות הרפידה הנדרשת
    pad_h = (block_size - height % block_size) % block_size
    pad_w = (block_size - width % block_size) % block_size

    # הוספת רפידה בהתאם לצורך
    if len(image.shape) == 3:  # תמונה צבעונית
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
        padded_mask = np.pad(mask, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')

    else:  # תמונה בגווני אפור
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='edge')
        padded_mask = np.pad(mask, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')


    print(f"Original size: {height}x{width}")
    print(f"Padded size: {padded_image.shape}")
    print(f"Padding added: {pad_h}x{pad_w}")

    blocks = []
    block_positions = []  # לשמירת מיקום הבלוקים
    blocks_mask = []
    block_positions_mask = []

    # חלוקה לבלוקים
    for i in range(0, padded_image.shape[0], block_size):
        for j in range(0, padded_image.shape[1], block_size):
            if len(image.shape) == 3:
                block = padded_image[i:i+block_size, j:j+block_size, :]
                block_mask = padded_mask[i:i+block_size, j:j+block_size, :]
            else:
                block = padded_image[i:i+block_size, j:j+block_size]
                block_mask = padded_mask[i:i+block_size, j:j+block_size]


            blocks.append(block)
            block_positions.append((i, j))
            blocks_mask.append(block_mask)
            block_positions_mask.append((i, j))

    print(f"Total blocks created: {len(blocks),len(blocks_mask)}")
    print(f"Each block size: {block_size}x{block_size}")

    return blocks, padded_image.shape, block_positions, blocks_mask, padded_mask.shape, block_positions_mask
# TODO לשחזור להוסיף כאן חלוקת מסכה לבלוקים

def blocks_to_image(blocks, image_shape, block_size=8):
    """שחזור התמונה מהבלוקים"""
    if len(image_shape) == 3:
        height, width, channels = image_shape
        image = np.zeros((height, width, channels))
    else:
        height, width = image_shape
        image = np.zeros((height, width))

    block_idx = 0
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            if len(image_shape) == 3:
                image[i:i+block_size, j:j+block_size, :] = blocks[block_idx]
            else:
                image[i:i+block_size, j:j+block_size] = blocks[block_idx]
            block_idx += 1

    return image

def visualize_blocks(image, num_blocks_to_show=9):
    """הצגת כמה בלוקים לדמונסטרציה"""

    blocks, padded_shape, positions = split_into_blocks(image)

    # הצגת הבלוקים הראשונים
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(min(num_blocks_to_show, len(blocks))):
        if len(blocks[i].shape) == 3:
            axes[i].imshow(blocks[i])
        else:
            axes[i].imshow(blocks[i], cmap='gray')

        axes[i].set_title(f'Block {i+1}\nPos: {positions[i]}')
        axes[i].axis('off')

    plt.suptitle('First 9 Blocks (8x8 each)')
    plt.tight_layout()
    plt.show()

    return blocks, padded_shape

"""
חלק 3: טרנספורמציית DCT (Discrete Cosine Transform)
"""
def dct2(block):
    """החלת DCT דו-מימדי על בלוק 8x8"""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    """החלת DCT הפוך דו-מימדי על בלוק 8x8"""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def apply_dct_to_blocks(blocks):
    """החלת DCT על כל הבלוקים"""
    dct_blocks = []

    for i, block in enumerate(blocks):
        if len(block.shape) == 3:  # בלוק צבעוני
            dct_block = np.zeros_like(block, dtype=np.float32)
            for channel in range(block.shape[2]):
                # הסטת הערכים למרכז סביב 0
                shifted = block[:, :, channel].astype(np.float32) - 128
                # החלת DCT
                dct_block[:, :, channel] = dct2(shifted)
        else:  # בלוק בגווני אפור
            # הסטת הערכים למרכז סביב 0
            shifted = block.astype(np.float32) - 128
            # החלת DCT
            dct_block = dct2(shifted)

        dct_blocks.append(dct_block)

    return dct_blocks

def apply_idct_to_blocks(dct_blocks):
    """החלת DCT הפוך על כל הבלוקים"""
    reconstructed_blocks = []

    for dct_block in dct_blocks:
        if len(dct_block.shape) == 3:  # בלוק צבעוני
            reconstructed_block = np.zeros_like(dct_block, dtype=np.float32)
            for channel in range(dct_block.shape[2]):
                # החלת DCT הפוך
                idct_result = idct2(dct_block[:, :, channel])
                # הסטה חזרה וחיתוך
                reconstructed_block[:, :, channel] = np.clip(idct_result + 128, 0, 255)
        else:  # בלוק בגווני אפור
            # החלת DCT הפוך
            idct_result = idct2(dct_block)
            # הסטה חזרה וחיתוך
            reconstructed_block = np.clip(idct_result + 128, 0, 255)

        reconstructed_blocks.append(reconstructed_block.astype(np.uint8))

    return reconstructed_blocks
# ויזואלי
def visualize_dct_coefficients(block, title="DCT Analysis"):
    """הצגת מקדמי DCT של בלוק"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # בלוק מקורי
    if len(block.shape) == 3:
        axes[0].imshow(block)
        # נתמקד בערוץ אחד לניתוח DCT
        gray_block = np.mean(block, axis=2)
    else:
        axes[0].imshow(block, cmap='gray')
        gray_block = block

    axes[0].set_title('Original Block')
    axes[0].axis('off')

    # החלת DCT
    shifted_block = gray_block.astype(np.float32) - 128
    dct_coeffs = dct2(shifted_block)

    # הצגת מקדמי DCT
    im1 = axes[1].imshow(dct_coeffs, cmap='RdBu_r', vmin=-100, vmax=100)
    axes[1].set_title('DCT Coefficients')
    axes[1].set_xlabel('Frequency →')
    axes[1].set_ylabel('Frequency →')
    plt.colorbar(im1, ax=axes[1])

    # הצגת ערכים מוחלטים
    im2 = axes[2].imshow(np.abs(dct_coeffs), cmap='hot')
    axes[2].set_title('DCT Coefficients (Magnitude)')
    axes[2].set_xlabel('Frequency →')
    axes[2].set_ylabel('Frequency →')
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

    return dct_coeffs

def analyze_frequency_distribution(dct_coeffs):
    """ניתוח התפלגות התדרים"""
    print("=== DCT Frequency Analysis ===")
    print(f"DC coefficient (0,0): {dct_coeffs[0,0]:.2f}")
    print(f"Max coefficient: {np.max(dct_coeffs):.2f}")
    print(f"Min coefficient: {np.min(dct_coeffs):.2f}")
    print(f"Mean absolute value: {np.mean(np.abs(dct_coeffs)):.2f}")

    # ספירת מקדמים קרובים לאפס
    threshold = 1.0
    near_zero = np.sum(np.abs(dct_coeffs) < threshold)
    print(f"Coefficients near zero (<{threshold}): {near_zero}/64 ({near_zero/64*100:.1f}%)")

    # ניתוח לפי אזורים
    print("\n=== Regional Analysis ===")
    print(f"Top-left 2x2 (low freq): {np.mean(np.abs(dct_coeffs[:2, :2])):.2f}")
    print(f"Top-right 2x2: {np.mean(np.abs(dct_coeffs[:2, -2:])):.2f}")
    print(f"Bottom-left 2x2: {np.mean(np.abs(dct_coeffs[-2:, :2])):.2f}")
    print(f"Bottom-right 2x2 (high freq): {np.mean(np.abs(dct_coeffs[-2:, -2:])):.2f}")
#TODO ולחבר את הפילוח תמונה להוסיף טבלת קוונטזציה 30 ו10
"""
חלק 4: קוונטיזציה ואיפוס מקדמים בתדר גבוה
"""
# איכות 50
# טבלאות קוונטיזציה סטנדרטיות של JPEG
QUANTIZATION_TABLE_Y = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

QUANTIZATION_TABLE_C = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])

def quantize_block(dct_block, quantization_table, quality_factor=1):
    """קוונטיזציה של בלוק DCT"""
    q_table = quantization_table * quality_factor
    return np.round(dct_block / q_table)

def dequantize_block(quantized_block, quantization_table, quality_factor=1):
    """ביטול קוונטיזציה"""
    q_table = quantization_table * quality_factor
    return quantized_block * q_table

def get_zigzag_order():
    """קבלת סדר הסריקה הזיגזג לבלוק 8x8"""
    return [
        (0,0), (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2),
        (2,1), (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5),
        (1,4), (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2),
        (3,3), (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4),
        (4,3), (5,2), (6,1), (7,0), (7,1), (6,2), (5,3), (4,4),
        (3,5), (2,6), (1,7), (2,7), (3,6), (4,5), (5,4), (6,3),
        (7,2), (7,3), (6,4), (5,5), (4,6), (3,7), (4,7), (5,6),
        (6,5), (7,4), (7,5), (6,6), (5,7), (6,7), (7,6), (7,7)
    ]

def zero_high_frequency_coefficients(dct_block, zero_percentage=0.5):
    """איפוס מקדמים בתדר גבוה לפי סדר זיגזג"""
    zigzag_order = get_zigzag_order()

    # חישוב כמות המקדמים לאיפוס
    num_coeffs_to_zero = int(64 * zero_percentage)

    # יצירת עותק של הבלוק
    modified_block = dct_block.copy()

    # איפוס מקדמי תדר גבוה (מסוף סדר הזיגזג)
    for i in range(64 - num_coeffs_to_zero, 64):
        row, col = zigzag_order[i]
        modified_block[row, col] = 0

    return modified_block

def visualize_quantization_process(dct_block, quantization_table, zero_percentage=0.5):
    """הצגת תהליך הקוונטיזציה והאיפוס"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # מקדמי DCT מקוריים
    im1 = axes[0,0].imshow(np.abs(dct_block), cmap='hot')
    axes[0,0].set_title('Original DCT Coefficients')
    plt.colorbar(im1, ax=axes[0,0])

    # טבלת קוונטיזציה
    im2 = axes[0,1].imshow(quantization_table, cmap='viridis')
    axes[0,1].set_title('Quantization Table')
    plt.colorbar(im2, ax=axes[0,1])

    # מקדמים מקוונטזים
    quantized = quantize_block(dct_block, quantization_table)
    im3 = axes[0,2].imshow(np.abs(quantized), cmap='hot')
    axes[0,2].set_title('Quantized Coefficients')
    plt.colorbar(im3, ax=axes[0,2])

    # מקדמים עם איפוס תדרים גבוהים
    zeroed = zero_high_frequency_coefficients(quantized, zero_percentage)
    im4 = axes[1,0].imshow(np.abs(zeroed), cmap='hot')
    axes[1,0].set_title(f'After Zeroing {zero_percentage*100}%')
    plt.colorbar(im4, ax=axes[1,0])

    # מקדמים לאחר ביטול קוונטיזציה
    dequantized = dequantize_block(zeroed, quantization_table)
    im5 = axes[1,1].imshow(np.abs(dequantized), cmap='hot')
    axes[1,1].set_title('Dequantized')
    plt.colorbar(im5, ax=axes[1,1])

    # השוואת מספר המקדמים שאינם אפס
    original_nonzero = np.count_nonzero(dct_block)
    quantized_nonzero = np.count_nonzero(quantized)
    zeroed_nonzero = np.count_nonzero(zeroed)

    stats_text = f'Non-zero coefficients:\nOriginal: {original_nonzero}/64\nQuantized: {quantized_nonzero}/64\nZeroed: {zeroed_nonzero}/64'
    axes[1,2].text(0.1, 0.5, stats_text, transform=axes[1,2].transAxes,
                   fontsize=12, verticalalignment='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[1,2].set_title('Compression Statistics')
    axes[1,2].axis('off')

    plt.tight_layout()
    plt.show()

    return quantized, zeroed, dequantized

def analyze_compression_effect(dct_block, quantization_table, zero_percentages):
    """ניתוח השפעת רמות דחיסה שונות"""
    print("=== Compression Analysis ===")

    results = []

    for zero_perc in zero_percentages:
        # קוונטיזציה
        quantized = quantize_block(dct_block, quantization_table)

        # איפוס מקדמים
        zeroed = zero_high_frequency_coefficients(quantized, zero_perc)

        # חישוב כמות המידע
        nonzero_count = np.count_nonzero(zeroed)
        compression_ratio = 64 / nonzero_count if nonzero_count > 0 else float('inf')

        # שחזור
        dequantized = dequantize_block(zeroed, quantization_table)

        # חישוב שגיאה
        error = np.mean(np.abs(dct_block - dequantized))

        results.append({
            'zero_percentage': zero_perc,
            'nonzero_coeffs': nonzero_count,
            'compression_ratio': compression_ratio,
            'error': error
        })

        print(f"Zero {zero_perc*100:3.0f}%: {nonzero_count:2d} coeffs, "
              f"ratio {compression_ratio:5.1f}:1, error {error:6.2f}")

    return results

def visualize_zigzag_pattern():
    """הצגת תבנית הסריקה הזיגזג"""
    zigzag_order = get_zigzag_order()

    # יצירת מטריצה עם מספרי הסדר
    zigzag_matrix = np.zeros((8, 8))
    for i, (row, col) in enumerate(zigzag_order):
        zigzag_matrix[row, col] = i

    plt.figure(figsize=(10, 8))
    plt.imshow(zigzag_matrix, cmap='viridis')
    plt.colorbar(label='Zigzag Order')
    plt.title('Zigzag Scanning Pattern\n(Lower numbers = lower frequency)')

    # הוספת מספרים על המטריצה
    for i in range(8):
        for j in range(8):
            plt.text(j, i, f'{int(zigzag_matrix[i,j])}',
                     ha='center', va='center', color='white', fontweight='bold')

    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.show()

if __name__ == "__main__":
    image_path =RF"C:\Users\user1\Pictures\2802.jpg"
    mask_path = RF"C:\Users\user1\Pictures"
    mask = segment_image(image_path,mask_path)
    image = cv2.imread(image_path)  # קורא את התמונה כ-BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ycbcr_image = rgb_to_ycbcr(image)
    print("YCbCr image shape:", ycbcr_image.shape)
    blocks, padded_image_shape, block_positions, blocks_mask, padded_mask_shape, block_positions_mask = split_into_blocks(image, mask)
    dct_blocks=apply_dct_to_blocks(blocks)
    for i, block in enumerate(dct_blocks):
        mask_block = blocks_mask[i]
        if (np.all(mask_block == 0))




# דוגמה לשימוש
if __name__ == "__main__":
    # יצירת בלוק DCT לדוגמה
    np.random.seed(42)
    sample_dct = np.random.randn(8, 8) * 50
    sample_dct[0, 0] = 1000  # DC component גדול

    print("=== Quantization and Zeroing Demo ===")

    # הצגת תהליך הקוונטיזציה
    quantized, zeroed, dequantized = visualize_quantization_process(
        sample_dct, QUANTIZATION_TABLE_Y, zero_percentage=0.5
    )

    # ניתוח השפעת רמות דחיסה שונות
    zero_percentages = [0.0, 0.3, 0.5, 0.7, 0.9]
    results = analyze_compression_effect(sample_dct, QUANTIZATION_TABLE_Y, zero_percentages)

    # הצגת תבנית הזיגזג
    print("\n=== Zigzag Pattern ===")
    visualize_zigzag_pattern()

    # השוואת טבלאות קוונטיזציה
    print("\n=== Quantization Tables Comparison ===")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im1 = axes[0].imshow(QUANTIZATION_TABLE_Y, cmap='viridis')
    axes[0].set_title('Luminance (Y) Quantization Table')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(QUANTIZATION_TABLE_C, cmap='viridis')
    axes[1].set_title('Chrominance (Cb/Cr) Quantization Table')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()

    print(f"Y table range: {QUANTIZATION_TABLE_Y.min()}-{QUANTIZATION_TABLE_Y.max()}")
    print(f"C table range: {QUANTIZATION_TABLE_C.min()}-{QUANTIZATION_TABLE_C.max()}")
# דוגמה לשימוש
if __name__ == "__main__":
    # יצירת בלוק דוגמה עם תבנית
    x = np.linspace(0, 2*np.pi, 8)
    y = np.linspace(0, 2*np.pi, 8)
    X, Y = np.meshgrid(x, y)

    # בלוק עם תבנית גלים
    pattern_block = 128 + 127 * np.sin(X) * np.cos(Y)
    pattern_block = pattern_block.astype(np.uint8)

    print("=== DCT Transform Demo ===")

    # הצגת ניתוח DCT
    dct_coeffs = visualize_dct_coefficients(pattern_block, "Pattern Block DCT")
    analyze_frequency_distribution(dct_coeffs)

    # בדיקת שחזור מושלם
    print("\n=== Perfect Reconstruction Test ===")
    shifted = pattern_block.astype(np.float32) - 128
    dct_result = dct2(shifted)
    reconstructed = idct2(dct_result) + 128

    error = np.mean(np.abs(pattern_block - reconstructed))
    print(f"Reconstruction error: {error:.6f}")
    print(f"Perfect reconstruction: {error < 1e-10}")

    # דוגמה עם בלוק אקראי
    print("\n=== Random Block Analysis ===")
    random_block = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
    random_dct = visualize_dct_coefficients(random_block, "Random Block DCT")
    analyze_frequency_distribution(random_dct)

# דוגמה לשימוש
if __name__ == "__main__":
    # יצירת תמונת דוגמה
    sample_image = np.random.randint(0, 256, (50, 70, 3), dtype=np.uint8)

    print("=== Block Splitting Demo ===")

    # חלוקה לבלוקים
    blocks, padded_shape, positions = split_into_blocks(sample_image)

    # שחזור התמונה
    reconstructed = blocks_to_image(blocks, padded_shape)

    print(f"\nReconstruction check:")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Original padded shape: {padded_shape}")

    # בדיקה שהשחזור מדויק
    print(f"Reconstruction is perfect: {np.allclose(reconstructed[:sample_image.shape[0], :sample_image.shape[1]], sample_image)}")

    # הצגת סטטיסטיקות
    print(f"\n=== Statistics ===")
    print(f"Number of blocks: {len(blocks)}")
    print(f"Blocks per row: {padded_shape[1] // BLOCK_SIZE}")
    print(f"Blocks per column: {padded_shape[0] // BLOCK_SIZE}")

    # דוגמה לבלוק בודד
    if len(blocks) > 0:
        print(f"\n=== First Block Example ===")
        print(f"Block shape: {blocks[0].shape}")
        print(f"Block position: {positions[0]}")
        if len(blocks[0].shape) == 2:  # grayscale
            print("Block values:")
            print(blocks[0])

# דוגמה לשימוש
if __name__ == "__main__":

    image_path = rf"C:\Users\user1\Pictures\28022.jpg"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Image shape: {image.shape}")


    # יצירת תמונת דוגמה

    print("Original RGB image shape:", image.shape)

    # המרה ל-YCbCr
    ycbcr_image = rgb_to_ycbcr(image)
    print("YCbCr image shape:", ycbcr_image.shape)

    # המרה חזרה ל-RGB
    rgb_reconstructed = ycbcr_to_rgb(ycbcr_image)
    print("Reconstructed RGB shape:", rgb_reconstructed.shape)

    # בדיקת הפרש
    diff = np.mean(np.abs(image.astype(float) - rgb_reconstructed.astype(float)))
    print(f"Mean difference after conversion: {diff:.2f}")
    # מציג את התמונה בחלון חדש
    cv2.imshow('תמונה', ycbcr_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ##