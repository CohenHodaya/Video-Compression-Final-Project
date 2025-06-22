import subprocess
import json
import os

def get_audio_info(video_path):
    """מחלץ מידע על רצועת השמע"""
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=codec_name,bit_rate,sample_rate,channels',
        '-of', 'json',
        video_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        info = json.loads(result.stdout)
        if info.get('streams'):
            stream = info['streams'][0]
            return {
                'codec': stream.get('codec_name', 'unknown'),
                'bitrate': stream.get('bit_rate', 'unknown'),
                'sample_rate': stream.get('sample_rate', 'unknown'),
                'channels': stream.get('channels', 'unknown')
            }
        else:
            return None
    except:
        return None

def extract_audio_original_format(video_path, output_audio):
    """מחלץ שמע בפורמט המקורי (ללא המרה - מהיר יותר)"""
    audio_info = get_audio_info(video_path)

    if not audio_info:
        print("❌ לא נמצאה רצועת שמע בווידאו")
        return False

    print(f"🎵 Audio info: {audio_info['codec']} | {audio_info['bitrate']} bps | {audio_info['sample_rate']} Hz | {audio_info['channels']} channels")

    # קביעת סיומת קובץ לפי קודק
    codec_extensions = {
        'aac': '.aac',
        'mp3': '.mp3',
        'ac3': '.ac3',
        'flac': '.flac',
        'opus': '.opus',
        'vorbis': '.ogg'
    }

    codec = audio_info['codec']
    if codec in codec_extensions:
        base_name = os.path.splitext(output_audio)[0]
        output_audio = base_name + codec_extensions[codec]

    command = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-vn',              # ללא וידאו
        '-acodec', 'copy',  # העתקת קודק מקורי (ללא המרה)
        output_audio
    ]

    print(f"🚀 Extracting audio in original format...")
    subprocess.run(command, check=True)
    print(f"✅ Audio extracted to: {output_audio}")
    return True

def extract_audio_mp3(video_path, output_audio, bitrate='192k'):
    """מחלץ שמע וממיר ל-MP3"""
    audio_info = get_audio_info(video_path)

    if not audio_info:
        print("❌ לא נמצאה רצועת שמע בווידאו")
        return False

    print(f"🎵 Converting audio to MP3 ({bitrate})")

    # וידוא שהקובץ מסתיים ב-.mp3
    if not output_audio.lower().endswith('.mp3'):
        output_audio = os.path.splitext(output_audio)[0] + '.mp3'

    command = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-vn',              # ללא וידאו
        '-acodec', 'mp3',   # המרה ל-MP3
        '-ab', bitrate,     # ביטרייט
        output_audio
    ]

    print(f"🚀 Converting audio to MP3...")
    subprocess.run(command, check=True)
    print(f"✅ Audio extracted to: {output_audio}")
    return True

def extract_audio_wav(video_path, output_audio):
    """מחלץ שמע וממיר ל-WAV (איכות מקסימלית)"""
    audio_info = get_audio_info(video_path)

    if not audio_info:
        print("❌ לא נמצאה רצועת שמע בווידאו")
        return False

    print(f"🎵 Converting audio to WAV (uncompressed)")

    # וידוא שהקובץ מסתיים ב-.wav
    if not output_audio.lower().endswith('.wav'):
        output_audio = os.path.splitext(output_audio)[0] + '.wav'

    command = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-vn',              # ללא וידאו
        '-acodec', 'pcm_s16le',  # WAV לא דחוס
        output_audio
    ]

    print(f"🚀 Converting audio to WAV...")
    subprocess.run(command, check=True)
    print(f"✅ Audio extracted to: {output_audio}")
    return True

def extract_audio_custom(video_path, output_audio, codec='aac', bitrate='128k', sample_rate=None):
    """מחלץ שמע עם הגדרות מותאמות אישית"""
    audio_info = get_audio_info(video_path)

    if not audio_info:
        print("❌ לא נמצאה רצועת שמע בווידאו")
        return False

    print(f"🎵 Custom audio extraction: {codec} | {bitrate}")

    command = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-vn',              # ללא וידאו
        '-acodec', codec,   # קודק מותאם
        '-ab', bitrate,     # ביטרייט
    ]

    # הוספת sample rate אם צוין
    if sample_rate:
        command.extend(['-ar', str(sample_rate)])

    command.append(output_audio)

    print(f"🚀 Extracting audio with custom settings...")
    subprocess.run(command, check=True)
    print(f"✅ Audio extracted to: {output_audio}")
    return True

if __name__ == "__main__":
    video_file = r"C:\Users\user1\Videos\video1\אבן נשברת.mp4"

    # אופציה 1: חילוץ בפורמט המקורי (הכי מהיר)
    extract_audio_original_format(video_file, r"C:\Users\user1\Music\audio")

    # אופציה 2: המרה ל-MP3
    extract_audio_mp3(video_file, r"C:\Users\user1\Music\audio\extracted.mp3", bitrate='192k')

    # אופציה 3: המרה ל-WAV (איכות מקסימלית)
    extract_audio_wav(video_file, r"C:\Users\user1\Music\audio\TRY_audio.wav")

    # אופציה 4: הגדרות מותאמות אישית
    # extract_audio_custom(video_file, r"C:\Users\user1\Audio\custom.aac",
    #                     codec='aac', bitrate='256k', sample_rate=48000)
    # חילוץ מהיר בפורמט המקורי
    extract_audio_original_format("video.mp4", "audio_original")

# MP3 באיכות גבוהה
    extract_audio_mp3("video.mp4", "audio.mp3", bitrate='320k')

# WAV לעריכה מקצועית
    extract_audio_wav("video.mp4", "audio.wav")