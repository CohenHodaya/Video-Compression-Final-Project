import cv2
import os
from Mask_creation import Creating_a_mask ,prepare_mask_for_compression
from Lossless_frame_compression import jpeg_compress_and_reconstruct_with_mask
from PIL import Image
import numpy as np
import subprocess



def RemovingFrames(output_folder):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    count = 0
    output_mask = rf"C:\Users\user1\PycharmProjects\FrameInterpolationModel\src\Mask"
    for filename in os.listdir(output_folder):
        if filename.lower().endswith(image_extensions):
           image_path = os.path.join(output_folder, filename)
           if count % 2 == 1:
             os.remove(image_path)
           else:

              image_rgb = np.array(Image.open(image_path).convert("RGB"))
              height, width, _ = image_rgb.shape
              size = (width, height)
              mask_path = Creating_a_mask(image_path, output_mask,size)
              mask_gray = np.array(Image.open(mask_path).convert("L"))
              mask_gray = (mask_gray > 128).astype(np.uint8) * 255
              compressed_frame, _ = jpeg_compress_and_reconstruct_with_mask(
                  image_rgb, mask_gray, normal_quality=70, strong_quality=45
              )
              Image.fromarray(compressed_frame).save(image_path)

           count = count+1

def SplitVideoToFrames(video_path, output_folder):

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
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def encode_video_h264(frames_dir, output_path, framerate=24, quality=23):
    # לוודא שהתיקייה קיימת
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"Directory not found: {frames_dir}")
    # הפקודה עצמה
    command = [
        'ffmpeg',
        '-y',  # תמיד תדרוס קובץ קיים
        '-framerate', str(framerate),
        '-i', os.path.join(frames_dir, '%04d.png'),  # התאמה לשמות frame_0001.png וכו'
        '-c:v', 'libx264',
        '-preset', 'slow',        # איזון בין מהירות לאיכות
        '-crf', str(quality),     # איכות – 18 טוב מאוד, 23 סביר
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    # הרצה
    subprocess.run(command, check=True)
    print(f"✔️ Video saved to: {output_path}")

def renumber_frames(folder):
    images = sorted([f for f in os.listdir(folder) if f.endswith('.png')])
    for i, filename in enumerate(images):
        src = os.path.join(folder, filename)
        dst = os.path.join(folder, f"{i:04d}.png")
        os.rename(src, dst)

if __name__ == "__main__":
    video_file =rf"C:\Users\user1\Videos\video1\אבן נשברת.mp4"#rf"C:\Users\user1\PycharmProjects\FrameInterpolationModel\src\my_video_h264.mp4"  #rf"C:\Users\user1\Downloads\nn.mp4" # input("אנא הזן את הנתיב המלא של קובץ הווידאו: ")
    output_directory = r"C:\Users\user1\Pictures\Experiment_with_compression_2025"
    video_file_1 =rf"C:\Users\user1\Videos\video1\24.6.25.mp4"
    #input("אנא הזן את הנתיב של התיקייה שבה תרצה לשמור את הפריימים: ")
    SplitVideoToFrames(video_file, output_directory)
    RemovingFrames(output_directory)
    renumber_frames(output_directory)
    fpss = (get_video_fps(video_file))
    encode_video_h264(output_directory,video_file_1,fpss)
#לסדר את כמות הפרימים בסרטון ארוך
#לסדר את הראשון והאחרון
#לסדר שלא יחלק ואז ימחק



