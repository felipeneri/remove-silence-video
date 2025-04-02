import argparse
import sys
import logging
import os
from silence_remover import SilenceRemover

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger("remove_silence")
file_handler = logging.FileHandler("remove_silence.log", mode="a")
file_handler.setLevel(logging.DEBUG)
log.addHandler(file_handler)

def log_callback(message, level="info"):
    """Callback for the SilenceRemover to log messages"""
    print(message)

def progress_callback(progress, message):
    """Callback for the SilenceRemover to update progress"""
    pass  # We don't need visual progress in CLI mode

def remove_silence(input_file, output_file=None, silence_threshold="-30dB", min_silence_duration=1, padding_ms=200, 
                  codec=None, bitrate=None, quality=None, preset=None, verbose=False):
    """
    Remove silence from a video file using the SilenceRemover class.
    
    Args:
        input_file: Path to the input video file
        output_file: Path to the output video file (if None, will be named "input_file_silence.ext")
        silence_threshold: The threshold below which audio is considered silence (in dB)
        min_silence_duration: Maximum silence duration to keep in seconds
        padding_ms: Amount of milliseconds to keep before and after non-silent parts
        codec: Specific video codec to use (None for auto-detection)
        bitrate: Video bitrate (e.g., "5M" for 5 megabits)
        quality: CRF value for quality (lower is better quality, 18-28 is typical)
        preset: Encoding preset (e.g., "medium", "fast", "veryfast")
        verbose: Whether to show debug information
    """
    # Set up the logger based on verbosity
    if verbose:
        log.setLevel(logging.DEBUG)
        
    # Create a SilenceRemover instance
    remover = SilenceRemover(
        input_file=input_file,
        output_file=output_file,
        silence_threshold=silence_threshold,
        min_silence_duration=min_silence_duration,
        padding_ms=padding_ms,
        codec=codec,
        bitrate=bitrate,
        quality=quality,
        preset=preset,
        on_progress=progress_callback,
        on_log=log_callback
    )
    
    # Run the silence removal process
    result = remover.remove_silence()
    return result

def main():
    parser = argparse.ArgumentParser(description='Remove silence from a video file.')
    parser.add_argument('input_file', help='Input video file path')
    parser.add_argument('-o', '--output', help='Output video file path')
    parser.add_argument('-t', '--threshold', default='-30dB', help='Silence threshold (default: -30dB)')
    parser.add_argument('-d', '--duration', type=float, default=1.0, 
                       help='Maximum silence duration to keep in seconds (default: 1.0)')
    parser.add_argument('-p', '--padding', type=int, default=200,
                       help='Padding in milliseconds before and after non-silent parts (default: 200ms)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show more detailed output')
    parser.add_argument('--codec', help='Specific video codec to use (e.g., h264_videotoolbox, libx264, h264_nvenc)')
    parser.add_argument('--bitrate', help='Video bitrate (e.g., 5M for 5 megabits/s)')
    parser.add_argument('--quality', type=int, help='Quality setting (CRF, lower is better quality, 18-28 is typical)')
    parser.add_argument('--preset', help='Encoding preset (e.g., medium, fast, slow for libx264)')
    
    args = parser.parse_args()
    
    print("\n=== Video Silence Remover ===")
    print(f"Processing file: {args.input_file}")
    print(f"Parameters: threshold={args.threshold}, max_silence={args.duration}s, padding={args.padding}ms")
    
    if args.codec:
        print(f"Using codec: {args.codec}")
    if args.bitrate:
        print(f"Using bitrate: {args.bitrate}")
    if args.quality:
        print(f"Using quality setting: {args.quality}")
    if args.preset:
        print(f"Using preset: {args.preset}")
    
    success = remove_silence(
        args.input_file, 
        args.output, 
        args.threshold, 
        args.duration,
        args.padding,
        args.codec,
        args.bitrate,
        args.quality,
        args.preset,
        args.verbose
    )
    
    if success:
        print("\nOperation completed successfully!")
    else:
        print("\nOperation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 