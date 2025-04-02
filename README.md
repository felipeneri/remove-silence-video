# Video Silence Remover

This application allows you to automatically remove silence from video files while maintaining good video quality with minimal file size.

## Features

- Detect and remove silent portions from video files
- Preserve smooth transitions with customizable padding
- Choose optimal compression settings for smaller file sizes
- Both command line and graphical user interfaces
- Supports hardware acceleration on compatible systems

## Usage Guide

### Basic Usage

```
python remove_silence.py input_video.mp4
```

This will create a new file named `input_video_silence.mp4` with silent parts removed.

### Command Line Options

```
python remove_silence.py input_video.mp4 [options]
```

#### Basic Options

- `-o, --output FILE` - Specify output file path
- `-t, --threshold VALUE` - Set silence threshold (default: -30dB)
- `-d, --duration SECONDS` - Set minimum silence duration to keep (default: 1.0s)
- `-p, --padding MILLISECONDS` - Set padding around non-silent parts (default: 200ms)

#### Compression Options

- `--codec CODEC` - Select a specific video codec (e.g., libx264, h264_videotoolbox)
- `--bitrate VALUE` - Set video bitrate (e.g., 5M for 5 megabits/s)
- `--quality VALUE` - Set video quality/CRF (18-28, lower is better quality)
- `--preset VALUE` - Set encoding preset (e.g., medium, fast, slow)

### Understanding Compression Options

#### Video Codec (`--codec`)

- `libx264` - Software H.264 encoder (works on all systems, best quality control)
- `h264_videotoolbox` - Hardware accelerated encoder for macOS (faster, but less quality control)
- `h264_nvenc` - NVIDIA GPU acceleration (for systems with NVIDIA graphics)

#### Quality vs Bitrate

You can control video quality in two ways:

1. **Quality-based encoding** (`--quality`):

   - Values range from 18 (high quality, large file) to 28 (lower quality, small file)
   - Recommended values: 23-25 for good balance
   - Example: `--codec libx264 --quality 23`

2. **Bitrate-based encoding** (`--bitrate`):
   - Specify exact bitrate like 2M (2 Mbps) or 500k (500 Kbps)
   - Example: `--codec h264_videotoolbox --bitrate 2M`

#### Encoding Presets (`--preset`)

For software encoding (`libx264`):

- `ultrafast` - Fastest encoding, largest files
- `veryfast` - Very quick encoding, larger files
- `faster` - Quick encoding, moderate file size
- `medium` - Good balance (default)
- `slow` - Better compression, slower encoding
- `veryslow` - Best compression, slowest encoding

## Recommended Settings

### Best Quality with Reasonable Size

```
python remove_silence.py video.mp4 --codec libx264 --quality 23 --preset medium
```

### Smaller Files with Good Quality

```
python remove_silence.py video.mp4 --codec libx264 --quality 26 --preset medium
```

### Fastest Processing (with hardware acceleration)

```
python remove_silence.py video.mp4 --codec h264_videotoolbox --bitrate 2M
```

### Optimal for Sharing Online

```
python remove_silence.py video.mp4 --codec libx264 --quality 28 --preset medium
```

## Testing Results

After extensive testing, we found that the `libx264` codec with quality (CRF) settings provides the best balance of quality and file size:

| Original File | Duration | Size   | Silent File | Duration | Size with Default | Size with Optimization |
| ------------- | -------- | ------ | ----------- | -------- | ----------------- | ---------------------- |
| Test video    | 18.0s    | 4.7 MB | Optimized   | 11.4s    | 3.0 MB            | 1.6 MB                 |

The optimized settings (`--codec libx264 --quality 28 --preset medium`) reduced the file size by ~65% compared to the original, while still maintaining good visual quality.

## Graphical User Interface

For a more user-friendly experience, you can use the graphical interface:

```
python remove_silence_ui.py
```

The user interface provides the same options as the command line version, allowing you to:

- Select input and output files
- Adjust silence detection parameters
- Choose compression settings
- Monitor progress in real-time

## Requirements

- Python 3.8 or higher
- FFmpeg (must be installed and available in your PATH)
- Required Python packages: `textual`, `rich`

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/video-silence-remover.git
cd video-silence-remover
```

2. Install the required packages:

```bash
# Using conda
conda create -n remove-video-silence python=3.10
conda activate remove-video-silence
conda install -c conda-forge ffmpeg textual rich

# Or using pip
pip install textual rich
```

3. Ensure FFmpeg is installed on your system:

```bash
# Check if FFmpeg is installed
ffmpeg -version
```

## How It Works

1. The program uses FFmpeg to detect silence in the video based on the specified threshold and duration
2. It creates a list of non-silent segments with padding
3. It processes the segments, either directly or in chunks for large videos
4. It generates a new video with only the non-silent parts

## Troubleshooting

- If you encounter the "Unknown encoder 'libx264'" error, ensure FFmpeg is compiled with x264 support
- For large videos, the program will automatically process in chunks to avoid memory issues
- If you see "Non-monotonic DTS" warnings, these are normal and don't affect the output quality
- Log files are saved in the `logs` directory for troubleshooting

## License

This project is licensed under the MIT License - see the LICENSE file for details.
