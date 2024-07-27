# Transcribe-Video-Text-Whisper

This program performs the following tasks:

1. **Searches for MP4 Files**: It identifies and processes MP4 video files.
2. **Converts MP4 to MP3**: Converts the video files to MP3 audio format.
3. **Splits Audio Based on VTT Contents**: Splits the MP3 audio into chunks based on VTT (WebVTT) subtitle timings. Each chunk should be no longer than 30 seconds, as the Whisper model supports only 30-second audio chunks.
4. **Translates Audio Chunks**: Uses the Whisper model to transcribe each audio chunk.
5. **Combines Transcriptions**: Merges the transcriptions from all chunks into a single text file.

The program is designed to handle the transcription of audio extracted from video files, ensuring that each segment adheres to the Whisper model's requirements for chunk length.  

**Requirements**:    
```
!pip install git+https://github.com/openai/whisper.git
!apt-get install ffmpeg
```
