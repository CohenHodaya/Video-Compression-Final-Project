from src.Model.CNN import predict_and_save
import os
import cv2
import tempfile
import shutil
import multiprocessing
from tensorflow.keras.models import load_model
import math


def SplitVideoToFrames(video_path, output_folder):
    """פיצול וידאו לפריימים"""
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


def renumber_frames_to_even(folder):
    """מספור מחדש של פריימים למספרים זוגיים בלבד"""
    images = sorted([f for f in os.listdir(folder) if f.endswith('.png')])

    if not images:
        print("No PNG files found in the folder.")
        return

    temp_dir = tempfile.mkdtemp()

    try:
        for i, filename in enumerate(images):
            src = os.path.join(folder, filename)
            new_name = f"{i * 2:04d}.png"
            temp_dst = os.path.join(temp_dir, new_name)
            shutil.move(src, temp_dst)

        temp_files = os.listdir(temp_dir)
        for temp_file in temp_files:
            temp_src = os.path.join(temp_dir, temp_file)
            final_dst = os.path.join(folder, temp_file)
            shutil.move(temp_src, final_dst)

        print(f"Successfully renumbered {len(images)} files to even numbers.")

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def process_frame_interpolation(output_directory, model_path, image_extensions, size, start_idx, end_idx, process_id):
    """פונקציה לעיבוד אינטרפולציה של פריימים לתהליך ספציפי"""
    print(f"Process {process_id}: Starting interpolation for frames {start_idx} to {end_idx}")

    # טעינת המודל לכל תהליך
    model = load_model(model_path)

    list_frame = sorted([f for f in os.listdir(output_directory) if f.endswith(image_extensions)])

    # עיבוד הפריימים בטווח שהוקצה לתהליך זה
    processed_count = 0
    for i in range(start_idx, min(end_idx, len(list_frame) - 1)):
        filename1 = list_frame[i]
        filename2 = list_frame[i + 1]

        image_path1 = os.path.join(output_directory, filename1)
        image_path2 = os.path.join(output_directory, filename2)

        # חישוב מספר הפריים החדש
        frame_number = int(os.path.splitext(filename1)[0]) + 1
        output_path = os.path.join(output_directory, f"{frame_number:04d}.png")

        try:
            predict_and_save(model, image_path1, image_path2, output_path, size)
            processed_count += 1
        except Exception as e:
            print(f"Process {process_id}: Error processing frame {frame_number}: {e}")

    print(f"Process {process_id}: Completed. Processed {processed_count} frames.")


def run_multiprocess_interpolation(output_directory, model_path, image_extensions, size):
    """הרצת אינטרפולציה רב-תהליכית"""
    # קבלת מספר הליבות
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores for processing")

    # קבלת רשימת הפריימים
    list_frame = sorted([f for f in os.listdir(output_directory) if f.endswith(image_extensions)])
    total_frames = len(list_frame) - 1  # מספר זוגות הפריימים לעיבוד

    if total_frames <= 0:
        print("No frames to process")
        return

    # חלוקת העבודה בין התהליכים
    frames_per_process = math.ceil(total_frames / num_cores)
    processes = []

    for i in range(num_cores):
        start_idx = i * frames_per_process
        end_idx = min((i + 1) * frames_per_process, total_frames)

        if start_idx >= total_frames:
            break

        process = multiprocessing.Process(
            target=process_frame_interpolation,
            args=(output_directory, model_path, image_extensions, size, start_idx, end_idx, i)
        )
        processes.append(process)
        process.start()

    # המתנה לסיום כל התהליכים
    for process in processes:
        process.join()

    print("All processes completed successfully!")


def generate_video(output_directory, fps):
    """יצירת וידאו מהפריימים"""
    image_folder = output_directory
    video_name = 'mygeneratedvideo.mp4'

    images = sorted([img for img in os.listdir(image_folder) if img.endswith((".jpg", ".jpeg", ".png"))])
    print(f"Found {len(images)} images for video generation")

    if not images:
        print("No images found for video generation")
        return

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()
    cv2.destroyAllWindows()
    print("Video generated successfully!")


if __name__ == "__main__":
    # הגדרת נתיבים
    video_file = r"C:\Users\user1\Videos\video1\25.6.25_Paper.mp4"
    output_directory = rf"C:\Users\user1\Pictures\Experiment_with_compression_25.6.25"
    model_path = rf"C:\Users\user1\PycharmProjects\FrameInterpolationModel\src\Model\best_model.keras"

    # הגדרות
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

    print("Starting video processing pipeline...")

    # שלב 1: פיצול הוידאו לפריימים
    print("Step 1: Splitting video to frames...")
    SplitVideoToFrames(video_file, output_directory)

    # שלב 2: מספור מחדש לפריימים זוגיים
    print("Step 2: Renumbering frames to even numbers...")
    renumber_frames_to_even(output_directory)

    # שלב 3: קבלת גודל התמונה
    print("Step 3: Getting image dimensions...")
    images = sorted([f for f in os.listdir(output_directory) if f.endswith('.png')])
    if images:
        image = cv2.imread(os.path.join(output_directory, images[0]))
        h, w, _ = image.shape
        size = (w, h)
        print(f"Image size: {size}")
    else:
        print("No images found!")
        exit()

    # שלב 4: אינטרפולציה רב-תהליכית
    print("Step 4: Running multi-process frame interpolation...")
    run_multiprocess_interpolation(output_directory, model_path, image_extensions, size)

    # שלב 5: יצירת וידאו
    print("Step 5: Generating final video...")
    cam = cv2.VideoCapture(video_file)
    fps = cam.get(cv2.CAP_PROP_FPS)
    cam.release()
    print(f"Original video FPS: {fps}")

    generate_video(output_directory, fps)  # FPS כפול בגלל האינטרפולציה

    print("Pipeline completed successfully!")