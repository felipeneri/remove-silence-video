# Video Transcription with WhisperX

This document provides instructions on how to use the WhisperX transcription scripts to generate accurate transcriptions for your videos after removing silence.

## Installation

Before using the scripts, make sure you have the required dependencies installed:

```bash
# Install WhisperX
pip install whisperx

# Make sure ffmpeg is installed
# On macOS:
brew install ffmpeg

# On Ubuntu/Debian:
# sudo apt update && sudo apt install ffmpeg
```

For speaker diarization, you'll need a Hugging Face token with access to the required models.

## Available Scripts

There are two main scripts:

1. `transcribe_video.py` - Only transcribes a video
2. `process_video.py` - Removes silence from a video and then transcribes it

## Usage Examples

### Transcribing a Video

To transcribe a video directly:

```bash
python transcribe_video.py path/to/video.mp4 --model medium
```

### Processing and Transcribing a Video

To remove silence and then transcribe:

```bash
python process_video.py path/to/video.mp4
```

## Command Line Options

### Common Options (both scripts)

- `--model [tiny/base/small/medium/large-v1/large-v2/large-v3]`: Whisper model to use (default: medium)
- `--compute-type [float16/int8]`: Compute type (use int8 for CPU-only systems)
- `--language [code]`: Language code (e.g., 'en', 'pt', 'fr')
- `-v/--verbose`: Show debug information
- `-o/--output-dir`: Directory for output files

### Transcription Options (both scripts)

- `--diarize`: Enable speaker diarization
- `--min-speakers [n]`: Minimum number of speakers
- `--max-speakers [n]`: Maximum number of speakers
- `--hf-token [token]`: Hugging Face token (required for diarization)
- `--batch-size [n]`: Batch size for processing

### Silence Removal Options (process_video.py only)

- `-t/--threshold [dB]`: Silence threshold (default: -30dB)
- `-d/--duration [seconds]`: Maximum silence duration to keep (default: 1.0s)
- `-p/--padding [ms]`: Padding in milliseconds (default: 200ms)
- `--codec [codec]`: Specific video codec
- `--bitrate [bitrate]`: Video bitrate (e.g., "5M")
- `--quality [crf]`: CRF quality setting (18-28, lower is better)
- `--preset [preset]`: Encoding preset (e.g., medium, fast)
- `--no-transcribe`: Skip transcription step

## Output Files

Both scripts generate the following files:

- `.srt`: Subtitle file for video players
- `.vtt`: WebVTT subtitle file for web videos
- `.txt`: Plain text transcript
- `.json`: Detailed JSON with timestamps and confidence scores

## Examples

### Basic Transcription (Small Model)

```bash
python transcribe_video.py my_video.mp4 --model small --compute-type int8
```

### Remove Silence and Transcribe with Speaker Identification

```bash
python process_video.py my_video.mp4 --model medium --diarize --min-speakers 2 --max-speakers 4 --hf-token YOUR_HF_TOKEN
```

### Custom Silence Removal with Higher Quality Transcription

```bash
python process_video.py my_video.mp4 --threshold -35dB --duration 0.5 --padding 300 --model large-v2 --language en
```

## Troubleshooting

- If you encounter memory issues, try a smaller model or use `--compute-type int8`
- For CPU-only systems, always use `--compute-type int8`
- For missing language support, check WhisperX documentation for supported languages
- If diarization fails, ensure your Hugging Face token has access to the required models
