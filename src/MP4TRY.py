import cv2
import os
from glob import glob

def images_to_video(folder_path, output_path='output.mp4', fps=30):
    # קבל את כל קבצי התמונות (jpg / png וכו') ממוינים לפי שם
    image_files = sorted(glob(os.path.join(folder_path, '*.*')))
    image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        raise ValueError("לא נמצאו תמונות בתיקייה")

    # טען את התמונה הראשונה כדי לקבל מימדים
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape

    # צור כותב וידאו
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # קודק ל-MP4
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_path in image_files:
        img = cv2.imread(img_path)
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        video.write(img)

    video.release()
    print(f"נשמר בהצלחה לקובץ: {output_path}")

if __name__ == "__main__":
    #video_file = rf"C:\Users\user1\Videos\video1\אבן נשברת.mp4" #rf"C:\Users\user1\Downloads\nn.mp4" # input("אנא הזן את הנתיב המלא של קובץ הווידאו: ")
    output_directory = r"C:\Users\user1\Pictures\try1111" #input("אנא הזן את הנתיב של התיקייה שבה תרצה לשמור את הפריימים: ")

    images_to_video(output_directory, 'my_video1.mp4', fps=40)

