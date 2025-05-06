from try2 import predict_and_save
import os
import cv2
#import threading
from multiprocessing import Process
import multiprocessing
from tensorflow.keras.models import load_model
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

#model = load_model(rf"C:\Users\user1\PycharmProjects\FrameInterpolationModel\src\my_model7.keras")
#image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
def function_name1(output_directory,model,image_extensions):
    list_frame = os.listdir(output_directory)
    for filename1,filename2 in zip(list_frame[:998:1], list_frame[1:999:1]):
        if (filename1.lower().endswith(image_extensions)) and (filename2.lower().endswith(image_extensions)):
            image_path1 = os.path.join(output_directory, filename1)
            image_path2 = os.path.join(output_directory, filename2)
            frame_number = int(os.path.splitext(filename1)[0]) + 1
            predict_and_save(model,image_path1,image_path2,rf"{output_directory}\{frame_number:04d}.png")

        #count=count+2

def function_name2(output_directory,model,image_extensions):
    list_frame = os.listdir(output_directory)
    for filename1,filename2 in zip(list_frame[1000:1700:1], list_frame[1001:1701:1]):
      if (filename1.lower().endswith(image_extensions)) and (filename2.lower().endswith(image_extensions)):
        image_path1 = os.path.join(output_directory, filename1)
        image_path2 = os.path.join(output_directory, filename2)
        frame_number = int(os.path.splitext(filename1)[0]) + 1
        predict_and_save(model,image_path1,image_path2,rf"{output_directory}\{frame_number:04d}.png")

      #count=count+2
def function_name3(output_directory,model,image_extensions):
    list_frame = os.listdir(output_directory)
    for filename1,filename2 in zip(list_frame[1702::1], list_frame[1703::1]):
        if (filename1.lower().endswith(image_extensions)) and (filename2.lower().endswith(image_extensions)):
            image_path1 = os.path.join(output_directory, filename1)
            image_path2 = os.path.join(output_directory, filename2)
            frame_number = int(os.path.splitext(filename1)[0]) + 1
            predict_and_save(model,image_path1,image_path2,rf"{output_directory}\{frame_number:04d}.png")









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

if __name__ == "__main__":
    video_file = rf"C:\Users\user1\Videos\video1\אבן נשברת.mp4" #rf"C:\Users\user1\Videos"
    output_directory = rf"C:\Users\user1\Pictures\try9" #input("אנא הזן את הנתיב של התיקייה שבה תרצה לשמור את הפריימים: ")
    count = output_directory.count()
    model = load_model(rf"C:\Users\user1\PycharmProjects\FrameInterpolationModel\src\my_model7.keras")
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
    t1 = Process(target=function_name1, args=(output_directory,model,image_extensions))
    t2 = Process(target=function_name2, args=(output_directory,model,image_extensions))
    t3 = Process(target=function_name3, args=(output_directory,model,image_extensions))
    t1.start()
    t2.start()
    t3.start()