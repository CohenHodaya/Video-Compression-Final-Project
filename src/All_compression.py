import cv2
import os

def list_files_in_directory(directory_path):
    """
    מדפיסה את כל שמות הקבצים בספריה הנתונה.

    :param directory_path: נתיב לספריה
    """
    try:
        files = os.listdir(directory_path)
        for filename in files:
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                print(filename)
    except FileNotFoundError:
        print("הספריה לא נמצאה:", directory_path)
    except Exception as e:
        print("אירעה שגיאה:", e)

frames_folder = 'frames'  # תיקיית פריימים
output_path = 'output.mp4'

# קרא את אחד הקבצים כדי לדעת את הגודל
first = cv2.imread(f"{frames_folder}/frame_000.jpg")
h, w, _ = first.shape

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))

for i in range(len(os.listdir(frames_folder))):
    frame = cv2.imread(f"{frames_folder}/frame_{i:03}.jpg")
    out.write(frame)

out.release()
