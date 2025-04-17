#!/usr/bin/env python3
import argparse
import os
import sys
import logging
from remove_silence import remove_silence
from transcribe_video import transcribe_video, ensure_dir

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger("process_video")
ensure_dir("logs")
file_handler = logging.FileHandler("logs/process_video.log", mode="a", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
log.addHandler(file_handler)

def process_video(
    input_file,
    output_dir=None,
    # Silence removal parameters
    silence_threshold="-30dB",
    min_silence_duration=1.0,
    padding_ms=200,
    codec=None,
    bitrate=None,
    quality=None,
    preset=None,
    # Transcription parameters
    transcribe=True,
    model_name="medium",
    language=None,
    align_model=None,
    batch_size=16,
    compute_type="float16",
    diarize=False,
    min_speakers=None,
    max_speakers=None,
    hf_token=None,
    verbose=False
):
    """
    Process a video by removing silence and optionally transcribing it
    
    Args:
        input_file: Path to the input video file
        output_dir: Directory where output files should be saved
        silence_threshold: The threshold below which audio is considered silence (in dB)
        min_silence_duration: Maximum silence duration to keep in seconds
        padding_ms: Amount of milliseconds to keep before and after non-silent parts
        codec: Specific video codec to use
        bitrate: Video bitrate
        quality: CRF value for quality
        preset: Encoding preset
        transcribe: Whether to transcribe the video after removing silence
        model_name: WhisperX model to use
        language: Language code (if None, auto-detected)
        align_model: Specific alignment model to use
        batch_size: Batch size for transcription
        compute_type: Compute precision for transcription
        diarize: Whether to perform speaker diarization
        min_speakers: Minimum number of speakers for diarization
        max_speakers: Maximum number of speakers for diarization
        hf_token: HuggingFace token for diarization
        verbose: Whether to show debug information
    
    Returns:
        Tuple of (processed_video_path, transcription_path)
    """
    if verbose:
        log.setLevel(logging.DEBUG)
    
    # Check if file exists
    if not os.path.exists(input_file):
        log.error(f"Input file '{input_file}' does not exist.")
        return None, None
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    ensure_dir(output_dir)
    
    # Remove silence from the video
    log.info("=== STEP 1: REMOVING SILENCE ===")
    
    # Determine output filename for processed video
    filename = os.path.basename(input_file)
    base_name, ext = os.path.splitext(filename)
    processed_video = os.path.join(output_dir, f"{base_name}_silence{ext}")
    
    silence_result = remove_silence(
        input_file=input_file,
        output_file=processed_video,
        silence_threshold=silence_threshold,
        min_silence_duration=min_silence_duration,
        padding_ms=padding_ms,
        codec=codec,
        bitrate=bitrate,
        quality=quality,
        preset=preset,
        verbose=verbose
    )
    
    if not silence_result:
        log.error("Failed to remove silence from the video.")
        return None, None
    
    log.info(f"Successfully removed silence: {processed_video}")
    
    # Transcribe the processed video if requested
    transcription_path = None
    if transcribe:
        log.info("\n=== STEP 2: TRANSCRIBING VIDEO ===")
        
        transcription_path = transcribe_video(
            input_file=processed_video,
            output_dir=output_dir,
            model_name=model_name,
            language=language,
            align_model=align_model,
            batch_size=batch_size,
            compute_type=compute_type,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            diarize=diarize,
            hf_token=hf_token,
            verbose=verbose
        )
        
        if not transcription_path:
            log.error("Failed to transcribe the video.")
        else:
            log.info(f"Successfully transcribed: {transcription_path}")
    
    return processed_video, transcription_path

def main():
    parser = argparse.ArgumentParser(description='Process a video by removing silence and optionally transcribing it.')
    parser.add_argument('input_file', help='Input video file path')
    parser.add_argument('-o', '--output-dir', help='Output directory for processed files')
    
    # Silence removal options
    silence_group = parser.add_argument_group('Silence Removal Options')
    silence_group.add_argument('-t', '--threshold', default='-30dB', 
                              help='Silence threshold (default: -30dB)')
    silence_group.add_argument('-d', '--duration', type=float, default=1.0,
                              help='Maximum silence duration to keep in seconds (default: 1.0)')
    silence_group.add_argument('-p', '--padding', type=int, default=200,
                              help='Padding in milliseconds before and after non-silent parts (default: 200ms)')
    silence_group.add_argument('--codec', help='Specific video codec to use')
    silence_group.add_argument('--bitrate', help='Video bitrate (e.g., 5M for 5 megabits/s)')
    silence_group.add_argument('--quality', type=int, 
                              help='Quality setting (CRF, lower is better quality, 18-28 is typical)')
    silence_group.add_argument('--preset', 
                              help='Encoding preset (e.g., medium, fast, slow for libx264)')
    
    # Transcription options
    trans_group = parser.add_argument_group('Transcription Options')
    trans_group.add_argument('--no-transcribe', action='store_true',
                           help='Skip transcription step')
    trans_group.add_argument('-m', '--model', default='medium',
                           choices=['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3'],
                           help='Whisper model to use (default: medium)')
    trans_group.add_argument('-l', '--language', 
                           help='Language code (auto-detect if not specified)')
    trans_group.add_argument('-a', '--align-model',
                           help='Alignment model to use (auto-selected if not specified)')
    trans_group.add_argument('-b', '--batch-size', type=int, default=16,
                           help='Batch size for processing (default: 16)')
    trans_group.add_argument('-c', '--compute-type', default='float16', 
                           choices=['float16', 'int8'],
                           help='Compute precision (default: float16)')
    trans_group.add_argument('--diarize', action='store_true',
                           help='Perform speaker diarization')
    trans_group.add_argument('--min-speakers', type=int, 
                           help='Minimum number of speakers for diarization')
    trans_group.add_argument('--max-speakers', type=int,
                           help='Maximum number of speakers for diarization')
    trans_group.add_argument('--hf-token',
                           help='HuggingFace token for diarization')
    
    # General options
    parser.add_argument('-v', '--verbose', action='store_true',
                      help='Show more detailed output')
    
    args = parser.parse_args()
    
    print("\n=== Video Processing Pipeline ===")
    print(f"Input file: {args.input_file}")
    print("Step 1: Silence Removal")
    print(f"  Silence threshold: {args.threshold}")
    print(f"  Max silence duration: {args.duration}s")
    print(f"  Padding: {args.padding}ms")
    
    if not args.no_transcribe:
        print("Step 2: Transcription")
        print(f"  Model: {args.model}")
        if args.language:
            print(f"  Language: {args.language}")
        if args.diarize:
            print("  Speaker diarization: Enabled")
            if args.min_speakers:
                print(f"  Minimum speakers: {args.min_speakers}")
            if args.max_speakers:
                print(f"  Maximum speakers: {args.max_speakers}")
    else:
        print("Step 2: Transcription (SKIPPED)")
    
    processed_video, transcription = process_video(
        input_file=args.input_file,
        output_dir=args.output_dir,
        # Silence removal parameters
        silence_threshold=args.threshold,
        min_silence_duration=args.duration,
        padding_ms=args.padding,
        codec=args.codec,
        bitrate=args.bitrate,
        quality=args.quality,
        preset=args.preset,
        # Transcription parameters
        transcribe=not args.no_transcribe,
        model_name=args.model,
        language=args.language,
        align_model=args.align_model,
        batch_size=args.batch_size,
        compute_type=args.compute_type,
        diarize=args.diarize,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        hf_token=args.hf_token,
        verbose=args.verbose
    )
    
    if processed_video:
        print("\nVideo processing completed successfully!")
        print(f"Processed video: {processed_video}")
        if transcription:
            print(f"Transcription: {transcription}")
    else:
        print("\nVideo processing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 