import cv2
import os

def RemovingFrames(output_folder):

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    count = 0
    # מעבר על כל הקבצים בתיקייה
    for filename in os.listdir(output_folder):
        if filename.lower().endswith(image_extensions):
          if count % 2 == 1:

            image_path = os.path.join(output_folder, filename)
            os.remove(image_path)
          count=count+1
            #print(f"פותח: {image_path}")
            #image = Image.open(image_path)
            #image.show()






#def Input_to_the_model(output_directory)



def SplitVideoToFrames(video_path, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # פתח את קובץ הווידאו
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

if __name__ == "__main__":
    video_file =rf"C:\Users\user1\Videos\video1\אבן נשברת.mp4"#rf"C:\Users\user1\PycharmProjects\FrameInterpolationModel\src\my_video_h264.mp4"  #rf"C:\Users\user1\Downloads\nn.mp4" # input("אנא הזן את הנתיב המלא של קובץ הווידאו: ")
    output_directory = r"C:\Users\user1\Pictures\tryTry" #input("אנא הזן את הנתיב של התיקייה שבה תרצה לשמור את הפריימים: ")
    SplitVideoToFrames(video_file, output_directory)
    #RemovingFrames(output_directory)

#לסדר את כמות הפרימים בסרטון ארוך
#לסדר את הראשון והאחרון
#לסדר שלא יחלק ואז ימחק