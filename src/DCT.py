from PIL import Image
import os
import time

def compress_image(input_path, output_path, quality=75):
    """
    Compress an image using JPEG compression

    Args:
        input_path (str): Path to the input image
        output_path (str): Path where compressed image will be saved
        quality (int): JPEG quality (1-95, higher means better quality but larger file)

    Returns:
        tuple: Original file size and compressed file size in KB
    """
    try:
        # Open the image
        with Image.open(input_path) as img:
            # Get original size
            original_size = os.path.getsize(input_path) / 1024  # KB

            # Make sure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save with JPEG compression
            img.save(output_path, 'JPEG', quality=quality, optimize=True)

            # Get compressed size
            compressed_size = os.path.getsize(output_path) / 1024  # KB

            return original_size, compressed_size
    except Exception as e:
        print(f"שגיאה בעיבוד הקובץ {input_path}: {e}")
        return 0, 0

def process_directory(input_dir, output_dir, quality=75):
    """
    Process all image files in a directory

    Args:
        input_dir (str): Directory containing image files
        output_dir (str): Directory to save compressed images
        quality (int): JPEG compression quality
    """
    # ודא שתיקיית הפלט קיימת
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # רשימת סיומות קבצים נתמכות
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

    # סטטיסטיקה
    total_files = 0
    total_original_size = 0
    total_compressed_size = 0
    start_time = time.time()

    print(f"מעבד תמונות מהתיקייה: {input_dir}")
    print(f"שומר תמונות דחוסות בתיקייה: {output_dir}")
    print(f"איכות דחיסה: {quality}")
    print("-" * 50)

    # עבור על כל הקבצים בתיקייה
    for filename in os.listdir(input_dir):
        # בדוק אם הקובץ הוא תמונה
        if filename.lower().endswith(image_extensions):
            # נתיבים מלאים
            input_file_path = os.path.join(input_dir, filename)
            # שמור את הקובץ בפורמט JPEG עם שם זהה אך סיומת jpg
            output_filename = os.path.splitext(filename)[0] + '.jpg'
            output_file_path = os.path.join(output_dir, output_filename)

            print(f"מעבד: {filename}")

            # דחס את התמונה
            orig_size, comp_size = compress_image(input_file_path, output_file_path, quality)

            if orig_size > 0:  # רק אם העיבוד הצליח
                reduction = ((orig_size - comp_size) / orig_size) * 100
                print(f"  מקורי: {orig_size:.2f} KB, דחוס: {comp_size:.2f} KB, חיסכון: {reduction:.2f}%")

                # עדכון סטטיסטיקה
                total_files += 1
                total_original_size += orig_size
                total_compressed_size += comp_size

    # הצג סיכום
    elapsed_time = time.time() - start_time

    print("\n" + "=" * 50)
    print("סיכום דחיסה:")
    print(f"קבצים שעובדו: {total_files}")
    print(f"זמן עיבוד: {elapsed_time:.2f} שניות")

    if total_files > 0:
        print(f"גודל מקורי כולל: {total_original_size:.2f} KB")
        print(f"גודל דחוס כולל: {total_compressed_size:.2f} KB")
        overall_reduction = ((total_original_size - total_compressed_size) / total_original_size) * 100
        print(f"חיסכון כולל: {overall_reduction:.2f}%")
    else:
        print("לא נמצאו קבצי תמונה בתיקייה.")

def main():
    # נתיבי התיקיות - שנה אותם לפי הצורך
    input_directory = r"C:\Users\user1\Pictures\try9"  # תיקיית התמונות המקוריות
    output_directory = r"C:\Users\user1\Pictures\try1111"  # תיקייה לשמירת התמונות הדחוסות

    # איכות דחיסה (1-95, כאשר 95 היא האיכות הגבוהה ביותר)
    compression_quality = 50

    # עבד את כל התמונות בתיקייה
    process_directory(input_directory, output_directory, compression_quality)

if __name__ == "__main__":
    main()
