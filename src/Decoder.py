from src.Model.CNN import predict_and_save
import os
import cv2
import tempfile
import shutil
#import threading
from multiprocessing import Process
import multiprocessing
from tensorflow.keras.models import load_model
from src.Closing_the_video import encode_video_with_audio
r"""
def SendingToModel(output_directory):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    count = 1
    #save_path = rf"{output_directory}\pred17.png"
    
    model = load_model(rf"C:\Users\user1\PycharmProjects\FrameInterpolationModel\src\my_model7.keras")
    """
"""
    # מעבר על כל הקבצים בתיקייה
    for filename in os.listdir(output_directory):
        filename1 = filename"""
"""
    list_frame = os.listdir(output_directory)
    print(list_frame)
    for filename1,filename2 in zip(list_frame[:1000:1], list_frame[1:1000:1]):
       if (filename1.lower().endswith(image_extensions)) and (filename2.lower().endswith(image_extensions)):
            image_path1 = os.path.join(output_directory, filename1)
            image_path2 = os.path.join(output_directory, filename2)
            predict_and_save(model,image_path1,image_path2,rf"{output_directory}\{count:04d}.png")

       count=count+2
"""
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
#model = load_model(rf"C:\Users\user1\PycharmProjects\FrameInterpolationModel\src\my_model7.keras")
#image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
def function_name1(output_directory,model,image_extensions,size):
    list_frame = os.listdir(output_directory)
    for filename1,filename2 in zip(list_frame[:998:1], list_frame[1:999:1]):
        if (filename1.lower().endswith(image_extensions)) and (filename2.lower().endswith(image_extensions)):
            image_path1 = os.path.join(output_directory, filename1)
            image_path2 = os.path.join(output_directory, filename2)
            frame_number = int(os.path.splitext(filename1)[0]) + 1
            predict_and_save(model,image_path1,image_path2,rf"{output_directory}\{frame_number:04d}.png",size)

        #count=count+2

def function_name2(output_directory,model,image_extensions,size):
    list_frame = os.listdir(output_directory)
    for filename1,filename2 in zip(list_frame[1000:1700:1], list_frame[1001:1701:1]):
      if (filename1.lower().endswith(image_extensions)) and (filename2.lower().endswith(image_extensions)):
        image_path1 = os.path.join(output_directory, filename1)
        image_path2 = os.path.join(output_directory, filename2)
        frame_number = int(os.path.splitext(filename1)[0]) + 1
        predict_and_save(model,image_path1,image_path2,rf"{output_directory}\{frame_number:04d}.png",size)

      #count=count+2
def function_name3(output_directory,model,image_extensions,size):
    list_frame = os.listdir(output_directory)
    for filename1,filename2 in zip(list_frame[1702::1], list_frame[1703::1]):
        if (filename1.lower().endswith(image_extensions)) and (filename2.lower().endswith(image_extensions)):
            image_path1 = os.path.join(output_directory, filename1)
            image_path2 = os.path.join(output_directory, filename2)
            frame_number = int(os.path.splitext(filename1)[0]) + 1
            predict_and_save(model,image_path1,image_path2,rf"{output_directory}\{frame_number:04d}.png",size)









"""
def SendingToModel1(output_directory):
    model_path = os.path.join("src", "my_model7.keras")

    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        model = load_model(model_path)
        SendingToModel(output_directory,model)
    else:
        print(f"Error: Model file not found at {model_path}")
"""
# Function to generate video

def generate_video(output_directory,fps):
    image_folder = output_directory
    video_name = 'mygeneratedvideo.mp4'

    images = [img for img in os.listdir(image_folder) if img.endswith((".jpg", ".jpeg", ".png"))]
    print("Images:", images)

    # Set frame from the first image
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Video writer to create .avi file
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Appending images to video
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # Release the video file
    video.release()
    cv2.destroyAllWindows()
    print("Video generated successfully!")


def renumber_frames_to_even(folder):
    """
    Renumbers PNG files in a folder to even numbers only.
    Original: 0000.png, 0001.png, 0002.png
    Result:   0000.png, 0002.png, 0004.png
    """
    # Get all PNG files and sort them
    images = sorted([f for f in os.listdir(folder) if f.endswith('.png')])

    if not images:
        print("No PNG files found in the folder.")
        return

    # Create temporary directory to avoid conflicts during renaming
    temp_dir = tempfile.mkdtemp()

    try:
        # First, move all files to temp directory with new even names
        for i, filename in enumerate(images):
            src = os.path.join(folder, filename)
            new_name = f"{i * 2:04d}.png"  # i * 2 gives us even numbers: 0, 2, 4, 6...
            temp_dst = os.path.join(temp_dir, new_name)
            shutil.move(src, temp_dst)

        # Then move them back to original folder
        temp_files = os.listdir(temp_dir)
        for temp_file in temp_files:
            temp_src = os.path.join(temp_dir, temp_file)
            final_dst = os.path.join(folder, temp_file)
            shutil.move(temp_src, final_dst)

        print(f"Successfully renumbered {len(images)} files to even numbers.")

    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# Example usage:
# renumber_frames_to_even("/path/to/your/folder")

if __name__ == "__main__":
    video_file = rf"C:\Users\user1\Videos\video1\try__22.6.25.mp4" #rf"C:\Users\user1\Videos"
    output_directory = rf"C:\Users\user1\Pictures\Experiment_with_compression_2025" #input("אנא הזן את הנתיב של התיקייה שבה תרצה לשמור את הפריימים: ")
    video_file_final = rf"C:\Users\user1\Videos\video1\try__22.6.25.mp4"

    SplitVideoToFrames(video_file,output_directory)
    renumber_frames_to_even(output_directory)
    image = cv2.imread(os.path.join(output_directory, sorted(os.listdir(output_directory))[0]))
    h, w, _ = image.shape
    size = (w, h)
    print(size)
    count = len([f for f in os.listdir(output_directory) if os.path.isfile(os.path.join(output_directory, f))])
    model = load_model(rf"C:\Users\user1\PycharmProjects\FrameInterpolationModel\src\Model\best_model.keras")
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    num_cores = multiprocessing.cpu_count()
    Average_frames = count/num_cores
    processes = []
    for i in range(num_cores):
        p = multiprocessing.Process(target=function_name1, args=(i,))
        processes.append(p)
        p.start()

    for p in processes:
       p.join()

    #SendingToModel(output_directory)
    cam = cv2.VideoCapture(video_file)
    fps = cam.get(cv2.CAP_PROP_FPS)
    print(fps)
    #generate_video(output_directory,fps)
    t1 = Process(target=function_name1, args=(output_directory,model,image_extensions,size))
    t2 = Process(target=function_name2, args=(output_directory,model,image_extensions,size))
    t3 = Process(target=function_name3, args=(output_directory,model,image_extensions,size))
    t1.start()
    t2.start()
    t3.start()

    encode_video_with_audio(output_directory,video_file,video_file_final)