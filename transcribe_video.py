#!/usr/bin/env python3
import argparse
import os
import sys
import logging
import whisperx
import torch
import gc
import json
from datetime import timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger("transcribe_video")
file_handler = logging.FileHandler("logs/transcribe_video.log", mode="a", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
log.addHandler(file_handler)

def ensure_dir(directory):
    """Ensure that a directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def format_timestamp(seconds, always_include_hours=False, decimal_marker='.'):
    """Format timestamp for SRT file."""
    hours = int(seconds / 3600)
    seconds = seconds % 3600
    minutes = int(seconds / 60)
    seconds = seconds % 60
    milliseconds = int(seconds * 1000) % 1000
    
    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{int(seconds):02d}{decimal_marker}{milliseconds:03d}"

def save_as_srt(segments, file):
    """Save segments as SRT file"""
    for i, segment in enumerate(segments, start=1):
        print(f"{i}", file=file)
        start = format_timestamp(segment["start"], always_include_hours=True, decimal_marker=',')
        end = format_timestamp(segment["end"], always_include_hours=True, decimal_marker=',')
        print(f"{start} --> {end}", file=file)
        
        text = segment["text"].strip()
        if "speaker" in segment:
            text = f"[Speaker {segment['speaker']}] {text}"
        
        print(f"{text}\n", file=file)

def save_as_vtt(segments, file):
    """Save segments as WebVTT file"""
    print("WEBVTT\n", file=file)
    
    for i, segment in enumerate(segments, start=1):
        start = format_timestamp(segment["start"], always_include_hours=True)
        end = format_timestamp(segment["end"], always_include_hours=True)
        print(f"{start} --> {end}", file=file)
        
        text = segment["text"].strip()
        if "speaker" in segment:
            text = f"[Speaker {segment['speaker']}] {text}"
        
        print(f"{text}\n", file=file)

def transcribe_video(
    input_file, 
    output_dir=None,
    model_name="medium",
    language=None,
    align_model=None,
    batch_size=16,
    compute_type="float16",
    min_speakers=None,
    max_speakers=None,
    diarize=False,
    hf_token=None,
    verbose=False,
):
    """
    Transcribe a video file using WhisperX
    
    Args:
        input_file: Path to the input video file
        output_dir: Directory where transcription files should be saved
        model_name: WhisperX model to use (tiny, base, small, medium, large-v1, large-v2, large-v3)
        language: Language code (e.g., 'en', 'pt', 'fr', etc. If None, auto-detected)
        align_model: Specific alignment model to use (if None, auto-selected based on language)
        batch_size: Batch size for processing
        compute_type: Compute precision (float16, int8)
        min_speakers: Minimum number of speakers for diarization
        max_speakers: Maximum number of speakers for diarization
        diarize: Whether to perform speaker diarization
        hf_token: HuggingFace token for diarization (required if diarize=True)
        verbose: Whether to show debug information
    
    Returns:
        Path to the .srt file
    """
    if verbose:
        log.setLevel(logging.DEBUG)
    
    # Check if file exists
    if not os.path.exists(input_file):
        log.error(f"Input file '{input_file}' does not exist.")
        return None
    
    # Create output directory if needed
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    ensure_dir(output_dir)
    
    # Determine output file names
    filename = os.path.basename(input_file)
    base_name, _ = os.path.splitext(filename)
    output_srt = os.path.join(output_dir, f"{base_name}.srt")
    output_txt = os.path.join(output_dir, f"{base_name}.txt")
    output_vtt = os.path.join(output_dir, f"{base_name}.vtt")
    output_json = os.path.join(output_dir, f"{base_name}.json")
    
    try:
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Using device: {device}")
        
        # Load audio
        log.info(f"Loading audio from: {input_file}")
        audio = whisperx.load_audio(input_file)
        
        # Load model
        log.info(f"Loading Whisper model: {model_name}")
        model = whisperx.load_model(model_name, device, compute_type=compute_type)
        
        # Transcribe with Whisper
        log.info("Transcribing with Whisper...")
        result = model.transcribe(audio, batch_size=batch_size, language=language)
        
        detected_language = result["language"]
        log.info(f"Detected language: {detected_language}")
        
        # Clear GPU memory for Whisper model
        del model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Align with WAV2VEC2
        log.info("Loading alignment model...")
        align_model, align_metadata = whisperx.load_align_model(
            language_code=detected_language, 
            device=device,
            model_name=align_model
        )
        
        log.info("Aligning the transcript...")
        result = whisperx.align(
            result["segments"], 
            align_model, 
            align_metadata, 
            audio, 
            device, 
            return_char_alignments=False
        )
        
        # Clear GPU memory for alignment model
        del align_model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Perform diarization if requested
        if diarize:
            if not hf_token:
                log.warning("Diarization requested but no Hugging Face token provided. Skipping diarization.")
            else:
                log.info("Loading diarization model...")
                try:
                    diarize_model = whisperx.DiarizationPipeline(
                        use_auth_token=hf_token, 
                        device=device
                    )
                    
                    diarization_options = {}
                    if min_speakers is not None:
                        diarization_options['min_speakers'] = min_speakers
                    if max_speakers is not None:
                        diarization_options['max_speakers'] = max_speakers
                    
                    log.info("Performing speaker diarization...")
                    diarize_segments = diarize_model(audio, **diarization_options)
                    
                    log.info("Assigning speakers to transcript...")
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    
                    # Clear GPU memory for diarization model
                    del diarize_model
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                except Exception as e:
                    log.error(f"Error during diarization: {e}")
        
        # Save results
        log.info("Saving transcription files...")
        
        # Save SRT
        with open(output_srt, 'w', encoding='utf-8') as f:
            save_as_srt(result["segments"], file=f)
        
        # Save plain text
        with open(output_txt, 'w', encoding='utf-8') as f:
            for segment in result["segments"]:
                if "speaker" in segment:
                    f.write(f"[Speaker {segment['speaker']}] {segment['text']}\n")
                else:
                    f.write(f"{segment['text']}\n")
        
        # Save VTT
        with open(output_vtt, 'w', encoding='utf-8') as f:
            save_as_vtt(result["segments"], file=f)
        
        # Save JSON
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        
        log.info(f"Transcription completed successfully! Files saved:")
        log.info(f"  SRT: {output_srt}")
        log.info(f"  TXT: {output_txt}")
        log.info(f"  VTT: {output_vtt}")
        log.info(f"  JSON: {output_json}")
        
        return output_srt
    
    except Exception as e:
        log.error(f"Error transcribing video: {e}")
        import traceback
        log.error(traceback.format_exc())
        return None

def main():
    parser = argparse.ArgumentParser(description='Transcribe a video file using WhisperX.')
    parser.add_argument('input_file', help='Input video file path')
    parser.add_argument('-o', '--output-dir', help='Output directory for transcription files')
    parser.add_argument('-m', '--model', default='medium', 
                        choices=['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3'],
                        help='Whisper model to use (default: medium)')
    parser.add_argument('-l', '--language', help='Language code (auto-detect if not specified)')
    parser.add_argument('-a', '--align-model', help='Alignment model to use (auto-selected if not specified)')
    parser.add_argument('-b', '--batch-size', type=int, default=16, 
                        help='Batch size for processing (default: 16)')
    parser.add_argument('-c', '--compute-type', default='float16', choices=['float16', 'int8'],
                        help='Compute precision (default: float16)')
    parser.add_argument('-d', '--diarize', action='store_true',
                        help='Perform speaker diarization')
    parser.add_argument('--min-speakers', type=int, help='Minimum number of speakers for diarization')
    parser.add_argument('--max-speakers', type=int, help='Maximum number of speakers for diarization')
    parser.add_argument('--hf-token', help='HuggingFace token for diarization')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show more detailed output')
    
    args = parser.parse_args()
    
    print("\n=== Video Transcription with WhisperX ===")
    print(f"Input file: {args.input_file}")
    print(f"Model: {args.model}")
    if args.language:
        print(f"Language: {args.language}")
    if args.diarize:
        print("Speaker diarization: Enabled")
        if args.min_speakers:
            print(f"Minimum speakers: {args.min_speakers}")
        if args.max_speakers:
            print(f"Maximum speakers: {args.max_speakers}")
    
    result = transcribe_video(
        input_file=args.input_file,
        output_dir=args.output_dir,
        model_name=args.model,
        language=args.language,
        align_model=args.align_model,
        batch_size=args.batch_size,
        compute_type=args.compute_type,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        diarize=args.diarize,
        hf_token=args.hf_token,
        verbose=args.verbose
    )
    
    if result:
        print("\nTranscription completed successfully!")
    else:
        print("\nTranscription failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 