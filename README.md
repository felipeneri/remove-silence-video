# Video Silence Remover

A Python-based tool for removing silent parts from videos, with both command-line and terminal UI interfaces.

## Features

- Remove silence from videos, keeping only the parts with audio
- Adjustable silence threshold and duration parameters
- Terminal UI with real-time progress tracking
- Command-line interface for easy scripting
- Hardware acceleration for faster processing on supported systems
- Detailed logging of the process

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

## Usage

### Terminal UI

Run the program with the Terminal UI for a more interactive experience:

```bash
python remove_silence_ui.py
```

You can pre-fill the UI with command-line arguments:

```bash
python remove_silence_ui.py --input path/to/video.mp4 --output path/to/output.mp4
```

### Command Line Interface

For quick usage or scripting, use the command-line interface:

```bash
python cli.py path/to/video.mp4
```

Options:

- `-o, --output`: Specify output file name (default: "no*silence*" + input filename)
- `-t, --threshold`: Silence threshold in dB (default: -30dB)
- `-d, --duration`: Maximum silence duration to keep in seconds (default: 1.0)
- `-p, --padding`: Padding in milliseconds to preserve around non-silent parts (default: 200ms)
- `-v, --verbose`: Show more detailed output

Examples:

```bash
# Process a video with default settings
python cli.py my_video.mp4

# Custom silence threshold and duration
python cli.py my_video.mp4 -t -35dB -d 0.8

# Specify output file and padding
python cli.py my_video.mp4 -o output.mp4 -p 150
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
