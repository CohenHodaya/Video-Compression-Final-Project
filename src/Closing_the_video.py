import subprocess
import json
import os

def extract_video_info(video_path):
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,avg_frame_rate',
        '-of', 'json',
        video_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    info = json.loads(result.stdout)
    stream = info['streams'][0]

    codec = stream['codec_name']
    fps_str = stream['avg_frame_rate']

    # הפיכת קצב פריימים למספר (למשל "25/1" -> 25.0)
    if '/' in fps_str:
        num, denom = map(int, fps_str.split('/'))
        fps = num / denom
    else:
        fps = float(fps_str)

    return codec, fps
def encode_video_auto(frames_dir, original_video, output_video):
    codec, fps = extract_video_info(original_video)

    # התאמה בין שם codec לשם FFmpeg
    codec_map = {
        'h264': 'libx264',
        'hevc': 'libx265',
        'vp9': 'libvpx-vp9'
    }
    ffmpeg_codec = codec_map.get(codec, 'libx264')  # ברירת מחדל

    print(f"🔍 Original codec: {codec}, FPS: {fps}, Using FFmpeg codec: {ffmpeg_codec}")

    command = [
        'ffmpeg',
        '-y',
        '-framerate', str(fps),
        '-i', os.path.join(frames_dir, '%04d.png'),
        '-c:v', ffmpeg_codec,
        '-preset', 'slow',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        output_video
    ]
    subprocess.run(command, check=True)
    print(f"🎥 Compressed video saved to: {output_video}")
    import subprocess
import json
import os

def extract_video_info(video_path):
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,avg_frame_rate',
        '-of', 'json',
        video_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    info = json.loads(result.stdout)
    stream = info['streams'][0]

    codec = stream['codec_name']
    fps_str = stream['avg_frame_rate']

    # הפיכת קצב פריימים למספר (למשל "25/1" -> 25.0)
    if '/' in fps_str:
        num, denom = map(int, fps_str.split('/'))
        fps = num / denom
    else:
        fps = float(fps_str)

    return codec, fps

def has_audio_stream(video_path):
    """בדיקה אם הווידאו המקורי מכיל רצועת שמע"""
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=codec_name',
        '-of', 'json',
        video_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        info = json.loads(result.stdout)
        return len(info.get('streams', [])) > 0
    except:
        return False

def encode_video_with_audio(frames_dir, original_video, output_video):
    codec, fps = extract_video_info(original_video)
    has_audio = has_audio_stream(original_video)

    # התאמה בין שם codec לשם FFmpeg
    codec_map = {
        'h264': 'libx264',
        'hevc': 'libx265',
        'h265': 'libx265',
        'vp9': 'libvpx-vp9',
        'av1': 'libaom-av1'
    }
    ffmpeg_codec = codec_map.get(codec, 'libx264')  # ברירת מחדל

    print(f"🔍 Original codec: {codec}, FPS: {fps}, Using FFmpeg codec: {ffmpeg_codec}")
    print(f"🎵 Audio detected: {'Yes' if has_audio else 'No'}")

    # בניית פקודת FFmpeg
    command = [
        'ffmpeg',
        '-y',
        '-framerate', str(fps),
        '-i', os.path.join(frames_dir, '%04d.png'),  # קלט הפריימים
    ]

    # הוספת קלט השמע אם קיים
    if has_audio:
        command.extend(['-i', original_video])

    # הגדרות וידאו
    command.extend([
        '-c:v', ffmpeg_codec,
        '-preset', 'slow',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
    ])

    # הגדרות שמע
    if has_audio:
        command.extend([
            '-c:a', 'aac',          # קודק שמע
            '-b:a', '128k',         # ביטרייט שמע
            '-map', '0:v:0',        # מפה את הווידאו מהקלט הראשון (הפריימים)
            '-map', '1:a:0',        # מפה את השמע מהקלט השני (הווידאו המקורי)
            '-shortest'             # מסיים כשהקלט הקצר ביותר מסתיים
        ])

    command.append(output_video)

    print(f"🎬 Running FFmpeg command...")
    subprocess.run(command, check=True)
    print(f"🎥 Video with audio saved to: {output_video}")

if __name__ == "__main__":
    original_video = rf"C:\Users\user1\Videos\video1\אבן נשברת.mp4"
    frames_folder = rf"C:\Users\user1\Pictures\Experiment_with_compression"
    output_video = r"C:\Users\user1\Videos\video1\final__output.mp4"

    # אופציה 1: שימוש בשמע מהווידאו המקורי
    encode_video_with_audio(frames_folder, original_video, output_video)


