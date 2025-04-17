#!/usr/bin/env python3
import argparse
import os
import sys
import logging
import whisperx
import torch
import gc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger("transcribe_audio")
ensure_dir = lambda d: os.makedirs(d, exist_ok=True)
ensure_dir("logs")
file_handler = logging.FileHandler("logs/transcribe_audio.log", mode="a", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
log.addHandler(file_handler)

def transcribe_audio(
    input_file, 
    output_dir=None,
    model_name="medium",
    language=None,
    batch_size=16,
    compute_type="float16",
    verbose=False,
):
    """
    Transcribe an audio file using WhisperX
    
    Args:
        input_file: Path to the input audio file
        output_dir: Directory where transcription file should be saved
        model_name: WhisperX model to use (tiny, base, small, medium, large-v1, large-v2, large-v3)
        language: Language code (e.g., 'en', 'pt', 'fr', etc. If None, auto-detected)
        batch_size: Batch size for processing
        compute_type: Compute precision (float16, int8)
        verbose: Whether to show debug information
    
    Returns:
        Path to the .txt file
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
    
    # Determine output file name
    filename = os.path.basename(input_file)
    base_name, _ = os.path.splitext(filename)
    output_txt = os.path.join(output_dir, f"{base_name}.txt")
    
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
            device=device
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
        
        # Save plain text
        log.info("Saving transcription file...")
        with open(output_txt, 'w', encoding='utf-8') as f:
            for segment in result["segments"]:
                f.write(f"{segment['text']}\n")
        
        log.info(f"Transcription completed successfully! File saved: {output_txt}")
        
        return output_txt
    
    except Exception as e:
        log.error(f"Error transcribing audio: {e}")
        import traceback
        log.error(traceback.format_exc())
        return None

def main():
    parser = argparse.ArgumentParser(description='Transcribe an audio file using WhisperX.')
    parser.add_argument('input_file', help='Input audio file path')
    parser.add_argument('-o', '--output-dir', help='Output directory for transcription file')
    parser.add_argument('-m', '--model', default='medium', 
                        choices=['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3'],
                        help='Whisper model to use (default: medium)')
    parser.add_argument('-l', '--language', help='Language code (auto-detect if not specified)')
    parser.add_argument('-b', '--batch-size', type=int, default=16, 
                        help='Batch size for processing (default: 16)')
    parser.add_argument('-c', '--compute-type', default='float16', choices=['float16', 'int8'],
                        help='Compute precision (default: float16)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show more detailed output')
    
    args = parser.parse_args()
    
    print("\n=== Audio Transcription with WhisperX ===")
    print(f"Input file: {args.input_file}")
    print(f"Model: {args.model}")
    if args.language:
        print(f"Language: {args.language}")
    
    result = transcribe_audio(
        input_file=args.input_file,
        output_dir=args.output_dir,
        model_name=args.model,
        language=args.language,
        batch_size=args.batch_size,
        compute_type=args.compute_type,
        verbose=args.verbose
    )
    
    if result:
        print("\nTranscription completed successfully!")
    else:
        print("\nTranscription failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 