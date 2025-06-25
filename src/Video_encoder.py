import cv2
import os
from Mask_creation import Creating_a_mask, prepare_mask_for_compression
from Lossless_frame_compression import jpeg_compress_and_reconstruct_with_mask
from PIL import Image
import numpy as np
import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


def process_single_frame(args):
    """
    פונקציה לעיבוד פריים יחיד - מיועדת לריצה מקבילה
    """
    image_path, output_mask, frame_index = args

    try:
        # טעינת התמונה
        image_rgb = np.array(Image.open(image_path).convert("RGB"))
        height, width, _ = image_rgb.shape
        size = (width, height)

        # יצירת מסכה
        mask_path = Creating_a_mask(image_path, output_mask, size)
        mask_gray = np.array(Image.open(mask_path).convert("L"))
        mask_gray = (mask_gray > 128).astype(np.uint8) * 255

        # דחיסה
        compressed_frame, _ = jpeg_compress_and_reconstruct_with_mask(
            image_rgb, mask_gray, normal_quality=70, strong_quality=45
        )

        # שמירת התמונה המדחסת
        Image.fromarray(compressed_frame).save(image_path)

        return f"Frame {frame_index} processed successfully"

    except Exception as e:
        return f"Error processing frame {frame_index}: {str(e)}"


def RemovingFrames_Parallel(output_folder, num_processes=None):
    """
    גרסה מקבילה של RemovingFrames
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    print(f"Using {num_processes} processes for frame processing")

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    output_mask = rf"C:\Users\user1\PycharmProjects\FrameInterpolationModel\src\Mask"

    # קבלת רשימת הקבצים ומחיקת הפריימים האי-זוגיים
    files_to_remove = []
    frames_to_process = []

    count = 0
    for filename in sorted(os.listdir(output_folder)):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(output_folder, filename)

            if count % 2 == 1:  # פריימים אי-זוגיים - למחיקה
                files_to_remove.append(image_path)
            else:  # פריימים זוגיים - לעיבוד
                frames_to_process.append((image_path, output_mask, count))

            count += 1

    # מחיקת פריימים אי-זוגיים
    print(f"Removing {len(files_to_remove)} odd frames...")
    for file_path in files_to_remove:
        os.remove(file_path)

    # עיבוד מקבילי של הפריימים הזוגיים
    print(f"Processing {len(frames_to_process)} even frames in parallel...")

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # שליחת כל המשימות לביצוע
        future_to_frame = {
            executor.submit(process_single_frame, frame_data): frame_data[2]
            for frame_data in frames_to_process
        }

        # מעקב אחר התקדמות
        completed = 0
        total = len(frames_to_process)

        for future in as_completed(future_to_frame):
            frame_index = future_to_frame[future]
            try:
                result = future.result()
                completed += 1
                print(f"Progress: {completed}/{total} - {result}")
            except Exception as exc:
                print(f"Frame {frame_index} generated an exception: {exc}")

    end_time = time.time()
    print(f"Parallel processing completed in {end_time - start_time:.2f} seconds")


def RemovingFrames_Sequential(output_folder):
    """
    הגרסה הרצפתית המקורית - לצורך השוואה
    """
    print("Processing frames sequentially...")
    start_time = time.time()

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    count = 0
    output_mask = rf"C:\Users\user1\PycharmProjects\FrameInterpolationModel\src\Mask"

    for filename in sorted(os.listdir(output_folder)):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(output_folder, filename)
            if count % 2 == 1:
                os.remove(image_path)
            else:
                image_rgb = np.array(Image.open(image_path).convert("RGB"))
                height, width, _ = image_rgb.shape
                size = (width, height)
                mask_path = Creating_a_mask(image_path, output_mask, size)
                mask_gray = np.array(Image.open(mask_path).convert("L"))
                mask_gray = (mask_gray > 128).astype(np.uint8) * 255
                compressed_frame, _ = jpeg_compress_and_reconstruct_with_mask(
                    image_rgb, mask_gray, normal_quality=70, strong_quality=45
                )
                Image.fromarray(compressed_frame).save(image_path)
                print(f"Processed frame {count}")

            count += 1

    end_time = time.time()
    print(f"Sequential processing completed in {end_time - start_time:.2f} seconds")


def SplitVideoToFrames(video_path, output_folder):
    """
    פונקציה ללא שינוי
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"שגיאה: לא ניתן לפתוח את קובץ הווידאו בנתיב: {video_path}")
        return
    frame_count = 0
    success, frame = video_capture.read()
    while success:
        frame_filename = os.path.join(output_folder, f"{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        success, frame = video_capture.read()
        frame_count += 1

    video_capture.release()
    print(f"הסרטון חולק בהצלחה ל-{frame_count} פריימים בתיקייה: {output_folder}")


def get_video_fps(video_path):
    """
    פונקציה ללא שינוי
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def encode_video_h264(frames_dir, output_path, framerate=24, quality=23):
    """
    פונקציה ללא שינוי
    """
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"Directory not found: {frames_dir}")

    command = [
        'ffmpeg',
        '-y',
        '-framerate', str(framerate),
        '-i', os.path.join(frames_dir, '%04d.png'),
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', str(quality),
        '-pix_fmt', 'yuv420p',
        output_path
    ]

    subprocess.run(command, check=True)
    print(f"✔️ Video saved to: {output_path}")


def renumber_frames(folder):
    """
    פונקציה ללא שינוי
    """
    images = sorted([f for f in os.listdir(folder) if f.endswith('.png')])
    for i, filename in enumerate(images):
        src = os.path.join(folder, filename)
        dst = os.path.join(folder, f"{i:04d}.png")
        os.rename(src, dst)


if __name__ == "__main__":
    # הגדרת נתיבים
    video_file = rf"C:\Users\user1\Videos\video1\גזירות נייר.mp4"
        #rf"C:\Users\user1\Videos\video1\אבן נשברת.mp4"
    output_directory = r"C:\Users\user1\Pictures\Experiment_with_compression_25.6.25_Paper"
    video_file_1 = rf"C:\Users\user1\Videos\video1\25.6.25_Paper.mp4"

    # חלוקת הווידאו לפריימים
    SplitVideoToFrames(video_file, output_directory)

    # קבלת מספר הליבות
    num_cores = multiprocessing.cpu_count()
    print(f"Detected {num_cores} CPU cores")

    # בחירה בין עיבוד מקבילי ורצפתי
    use_parallel = True  # שנה ל-False אם אתה רוצה לבדוק את הביצועים של הגרסה הרצפתית

    if use_parallel:
        # עיבוד מקבילי
        RemovingFrames_Parallel(output_directory, num_processes=num_cores)
    else:
        # עיבוד רצפתי (המקורי)
        RemovingFrames_Sequential(output_directory)

    # המשך התהליך
    renumber_frames(output_directory)
    fpss = get_video_fps(video_file)
    encode_video_h264(output_directory, video_file_1, fpss)