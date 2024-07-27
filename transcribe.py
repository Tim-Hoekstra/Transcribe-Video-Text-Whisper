import os
import re
import subprocess
import whisper
import shutil
import torch
def extract_audio_from_video(video_file, audio_file):
    """Extract audio from video file."""
    command = [
        'ffmpeg',
        '-i', video_file,
        '-q:a', '0',
        '-map', 'a',
        audio_file
    ]
    subprocess.run(command, check=True)

def parse_vtt(vtt_file):
    """Parse VTT file and return list of (start_time, end_time) tuples."""
    with open(vtt_file, 'r') as file:
        content = file.read()

    # Regex to find timestamps in VTT format
    pattern = re.compile(r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{02}:\d{2}\.\d{3})')
    matches = pattern.findall(content)

    segments = []
    for match in matches:
        start, end = match
        start_seconds = convert_to_seconds(start)
        end_seconds = convert_to_seconds(end)
        segments.append((start_seconds, end_seconds))

    return segments

def convert_to_seconds(timestamp):
    """Convert VTT timestamp to seconds."""
    h, m, s = timestamp.split(':')
    s, ms = s.split('.')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

def split_audio(audio_file, segments):
    """Split audio file based on segments."""
    split_files = []
    for i, (start, end) in enumerate(segments):
        start_str = format_time(start)
        end_str = format_time(end)
        output_file = f"segment_{i}.mp3"
        split_files.append(output_file)

        command = [
            'ffmpeg',
            '-i', audio_file,
            '-ss', start_str,
            '-to', end_str,
            '-c', 'copy',
            output_file
        ]
        subprocess.run(command, check=True)
    return split_files

def format_time(seconds):
    """Format seconds to HH:MM:SS.mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    milliseconds = int((secs - int(secs)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(secs):02}.{milliseconds:03}"

def process_video(video_file, vtt_file, model):
    audio_file = video_file.replace(".mp4", ".mp3")

    # Step 1: Extract audio from video
    print(f"Extracting audio from {video_file}...")
    extract_audio_from_video(video_file, audio_file)

    # Step 2: Parse VTT file
    print(f"Parsing VTT file {vtt_file}...")
    segments = parse_vtt(vtt_file)

    # Step 3: Split audio based on segments
    print(f"Splitting audio {audio_file}...")
    split_files = split_audio(audio_file, segments)

    # Step 4: Transcribe audio segments
    print(f"Transcribing audio segments for {video_file}...")

    transcriptions = []
    for split_file in split_files:
        audio = whisper.load_audio(split_file)
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)
        transcriptions.append(result.text)

    full_transcription = ' '.join(transcriptions)
    full_transcription = full_transcription.replace('.', ' ')
    full_transcription = full_transcription.replace('  ', ' ')

    transcription_file = video_file.replace(".mp4", "_transcription.txt")
    with open(transcription_file, 'w') as file:
        file.write(full_transcription)

    print(f"Transcription saved to {transcription_file}")

    # Cleanup
    os.remove(audio_file)
    for split_file in split_files:
        os.remove(split_file)

def main():
    #torch.cuda.is_available()
    #torch.cuda.init()
    model = whisper.load_model("base")
    #model = model.to('cuda')
    for root, dirs, files in os.walk("/content/"):
        for file in files:
            if file.endswith(".mp4"):
                video_file = os.path.join(root, file)
                vtt_file = video_file.replace(".mp4", ".vtt")
                if os.path.exists(vtt_file):
                    process_video(video_file, vtt_file, model)

if __name__ == "__main__":
    main()
