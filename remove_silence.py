import argparse
import subprocess
import os
import sys
import time
import tempfile
import shutil
import json
import re

def remove_silence(input_file, output_file=None, silence_threshold="-30dB", min_silence_duration=1, padding_ms=200):
    """
    Remove silence from a video file using FFmpeg.
    
    Args:
        input_file: Path to the input video file
        output_file: Path to the output video file (if None, will be named "no_silence_" + input_file)
        silence_threshold: The threshold below which audio is considered silence (in dB)
        min_silence_duration: Maximum silence duration to keep in seconds
        padding_ms: Amount of milliseconds to keep before and after non-silent parts
    """
    print(f"Step 1: Checking if input file exists...")
    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' does not exist.")
        return False
    
    # Convert to absolute path
    input_file = os.path.abspath(input_file)
    
    if output_file is None:
        filename = os.path.basename(input_file)
        output_file = "no_silence_" + filename
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        print(f"Step 2: Detecting silence in '{input_file}'...")
        print(f"Using silence threshold: {silence_threshold}, minimum silence duration: {min_silence_duration}s")
        
        # Create a silencedetect file
        silence_file = os.path.join(temp_dir, "silence_data.txt")
        
        try:
            # Run the silence detection in a more efficient way
            subprocess.run([
                'ffmpeg', 
                '-i', input_file, 
                '-af', f'silencedetect=noise={silence_threshold}:d={min_silence_duration}', 
                '-f', 'null', 
                '-'
            ], stderr=open(silence_file, 'w'), check=True)
            
            print("Silence detection completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error during silence detection: {e}")
            return False
        
        # Parse the silence detection output
        print(f"Step 3: Parsing silence detection results...")
        silence_starts = []
        silence_ends = []
        
        with open(silence_file, 'r') as f:
            silence_data = f.read()
            for line in silence_data.split('\n'):
                if 'silence_start' in line:
                    try:
                        start_time = float(line.split('silence_start:')[1].strip().split(' ')[0])
                        silence_starts.append(start_time)
                        print(f"Found silence starting at: {start_time:.2f}s")
                    except (IndexError, ValueError):
                        pass
                elif 'silence_end' in line:
                    try:
                        parts = line.split('silence_end:')[1].strip().split(' ')
                        end_time = float(parts[0])
                        silence_ends.append(end_time)
                        print(f"Found silence ending at: {end_time:.2f}s")
                    except (IndexError, ValueError):
                        pass
        
        print(f"Found {len(silence_starts)} silence start points and {len(silence_ends)} silence end points.")
        
        # If there's no silence detected or error in parsing
        if not silence_starts and not silence_ends:
            print("No silence detected or error in parsing FFmpeg output.")
            return False
        
        # Get video duration
        print(f"Step 4: Getting video duration...")
        try:
            duration_output = subprocess.check_output([
                'ffprobe', 
                '-v', 'error', 
                '-show_entries', 'format=duration', 
                '-of', 'default=noprint_wrappers=1:nokey=1', 
                input_file
            ], universal_newlines=True).strip()
            
            duration = float(duration_output)
            print(f"Video duration: {duration:.2f}s")
        except (ValueError, subprocess.CalledProcessError) as e:
            print(f"Error getting video duration: {e}")
            return False
            
        # Process the silence intervals to create non-silent segments
        print(f"Step 5: Creating segments list of non-silent parts...")
        segments = []
        padding_sec = padding_ms / 1000  # Convert to seconds
        
        # Check if video starts with non-silence
        if not silence_starts or silence_starts[0] > 0:
            start_time = 0
            end_time = silence_starts[0] if silence_starts else duration
            segments.append((start_time, end_time))
            print(f"Adding initial non-silent segment: {start_time:.2f}s to {end_time:.2f}s")
        
        # Process the silence intervals
        for i in range(len(silence_starts)):
            # Add segment from end of current silence to start of next silence
            current_silence_end = silence_ends[i] if i < len(silence_ends) else duration
            next_silence_start = silence_starts[i+1] if i+1 < len(silence_starts) else duration
            
            if current_silence_end < next_silence_start:
                # Apply padding
                start_time = max(0, current_silence_end - padding_sec)
                end_time = min(duration, next_silence_start + padding_sec)
                segments.append((start_time, end_time))
                print(f"Adding non-silent segment: {start_time:.2f}s to {end_time:.2f}s")
        
        print(f"Total non-silent segments: {len(segments)}")
        
        # Since concat demuxer might be having issues with the segments file,
        # let's try an alternative approach using the filter_complex method
        print(f"Step 6: Creating filter complex for trimming and concatenating...")
        
        # Check if there are too many segments for filter_complex
        if len(segments) > 80:  # Filter complex has limitations
            print(f"Warning: Large number of segments ({len(segments)}). Processing in chunks...")
            
            # Process in chunks by extracting segments to temporary files
            temp_files = []
            chunk_size = 50  # Number of segments per chunk
            
            for chunk_idx in range(0, len(segments), chunk_size):
                chunk_segments = segments[chunk_idx:chunk_idx + chunk_size]
                chunk_file = os.path.join(temp_dir, f"chunk_{chunk_idx}.mp4")
                temp_files.append(chunk_file)
                
                # Create filter complex for this chunk
                filter_complex = create_filter_complex(chunk_segments)
                
                print(f"Processing chunk {chunk_idx//chunk_size + 1}/{(len(segments) + chunk_size - 1)//chunk_size}...")
                
                cmd = [
                    'ffmpeg',
                    '-y',
                    '-i', input_file,
                    '-filter_complex', filter_complex,
                    '-map', '[v]',
                    '-map', '[a]',
                    '-c:v', 'h264_videotoolbox',  # Use hardware acceleration
                    '-c:a', 'aac',  # Re-encode audio for compatibility
                    '-b:a', '128k',
                    '-pix_fmt', 'yuv420p',
                    chunk_file
                ]
                
                subprocess.run(cmd, check=True)
                print(f"Chunk {chunk_idx//chunk_size + 1} completed.")
            
            # Create a file list for concatenation
            concat_list = os.path.join(temp_dir, "concat_list.txt")
            with open(concat_list, 'w') as f:
                for temp_file in temp_files:
                    f.write(f"file '{temp_file}'\n")
            
            # Final concatenation
            print("Joining chunks into final output...")
            subprocess.run([
                'ffmpeg',
                '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_list,
                '-c', 'copy',
                output_file
            ], check=True)
            
        else:
            # Create filter complex string directly
            filter_complex = create_filter_complex(segments)
            
            # Check for available hardware acceleration
            use_hardware_accel = check_hardware_acceleration()
            
            # Execute FFmpeg to create the output file without silences
            print(f"Step 7: Processing video and removing silence, saving to {output_file}...")
            
            try:
                start_time = time.time()
                
                # Use filter_complex approach
                cmd = [
                    'ffmpeg',
                    '-y',  # Overwrite output files
                    '-i', input_file,
                    '-filter_complex', filter_complex,
                    '-map', '[v]',
                    '-map', '[a]'
                ]
                
                # Add encoding parameters based on hardware acceleration
                if use_hardware_accel == 'videotoolbox':
                    cmd.extend([
                        '-c:v', 'h264_videotoolbox',
                        '-pix_fmt', 'yuv420p',
                        '-c:a', 'aac',
                        '-b:a', '128k'
                    ])
                elif use_hardware_accel == 'nvenc':
                    cmd.extend([
                        '-c:v', 'h264_nvenc',
                        '-preset', 'p1',
                        '-c:a', 'aac',
                        '-b:a', '128k'
                    ])
                else:
                    cmd.extend([
                        '-c:v', 'libx264',
                        '-preset', 'ultrafast',  # Use fastest preset for encoding speed
                        '-crf', '23',            # Balance quality and size
                        '-c:a', 'aac',
                        '-b:a', '128k'
                    ])
                
                cmd.append(output_file)
                
                # Print the command for debugging
                print("Executing command: " + " ".join(cmd))
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
                
                # Print progress updates
                while process.poll() is None:
                    line = process.stdout.readline()
                    if line:
                        if 'time=' in line:
                            print(f"Progress: {line.strip()}")
                        elif 'Error' in line or 'error' in line.lower():
                            print(f"Error detected: {line.strip()}")
                    time.sleep(0.5)  # Reduce output frequency
                    
                rc = process.wait()
                
                elapsed_time = time.time() - start_time
                if rc == 0:
                    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                        print(f"Video processing completed successfully in {elapsed_time:.2f} seconds")
                        print(f"Output file saved to: {output_file}")
                        print(f"Output file size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
                        return True
                    else:
                        print(f"Error: Output file not created or empty: {output_file}")
                        return False
                else:
                    print(f"Error: FFmpeg exited with code {rc}")
                    return False
            except subprocess.CalledProcessError as e:
                print(f"Error during video processing: {e}")
                if hasattr(e, 'output'):
                    print(f"Error output: {e.output}")
                return False
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)

def create_filter_complex(segments):
    """Create a filter_complex string for FFmpeg to extract and concatenate segments."""
    trim_parts = []
    concat_parts = []
    
    for i, (start, end) in enumerate(segments):
        # Create trim filters for video and audio
        trim_parts.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}];")
        trim_parts.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}];")
        
        # Add to concat list
        concat_parts.append(f"[v{i}]")
        concat_parts.append(f"[a{i}]")
    
    # Create concatenation part
    concat_v = ''.join(concat_parts[0::2])  # Even indices are video parts
    concat_a = ''.join(concat_parts[1::2])  # Odd indices are audio parts
    n = len(segments)
    
    trim_str = ''.join(trim_parts)
    concat_str = f"{concat_v}concat=n={n}:v=1:a=0[v];{concat_a}concat=n={n}:v=0:a=1[a]"
    
    return trim_str + concat_str

def check_hardware_acceleration():
    """Check for available hardware acceleration options."""
    try:
        # Check for hardware acceleration
        hwaccel_output = subprocess.check_output(
            ['ffmpeg', '-hide_banner', '-hwaccels'], 
            stderr=subprocess.STDOUT, 
            universal_newlines=True
        )
        
        if 'videotoolbox' in hwaccel_output.lower():
            print("Using macOS VideoToolbox hardware acceleration")
            return 'videotoolbox'
        elif 'cuda' in hwaccel_output.lower():
            print("Using NVIDIA CUDA hardware acceleration")
            return 'nvenc'
        else:
            print("No suitable hardware acceleration found, using libx264")
            return None
    except subprocess.CalledProcessError:
        print("Could not check hardware acceleration. Using libx264.")
        return None

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
    
    args = parser.parse_args()
    
    print("\n=== Video Silence Remover ===")
    print(f"Processing file: {args.input_file}")
    print(f"Parameters: threshold={args.threshold}, max_silence={args.duration}s, padding={args.padding}ms")
    
    success = remove_silence(
        args.input_file, 
        args.output, 
        args.threshold, 
        args.duration,
        args.padding
    )
    
    if success:
        print("\nOperation completed successfully!")
    else:
        print("\nOperation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 