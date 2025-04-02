import subprocess
import os
import time
import tempfile
import shutil
import re
import logging
from typing import List, Tuple, Optional, Callable

# Set up logging
logger = logging.getLogger("silence_remover")

class SilenceRemover:
    """Core functionality for removing silence from videos"""
    
    def __init__(self, input_file: str, output_file: Optional[str] = None, 
                 silence_threshold: str = "-30dB", min_silence_duration: float = 1.0,
                 padding_ms: int = 200, on_progress: Optional[Callable] = None, 
                 on_log: Optional[Callable] = None):
        """
        Initialize the silence remover
        
        Args:
            input_file: Path to the input video file
            output_file: Path to the output video file (if None, will be named "no_silence_" + input_file)
            silence_threshold: The threshold below which audio is considered silence (in dB)
            min_silence_duration: Maximum silence duration to keep in seconds
            padding_ms: Amount of milliseconds to keep before and after non-silent parts
            on_progress: Callback for progress updates
            on_log: Callback for log messages
        """
        self.input_file = os.path.abspath(input_file)
        
        if output_file is None:
            filename = os.path.basename(input_file)
            self.output_file = "no_silence_" + filename
        else:
            self.output_file = output_file
            
        self.silence_threshold = silence_threshold
        self.min_silence_duration = min_silence_duration
        self.padding_ms = padding_ms
        self.temp_dir = None
        self.on_progress = on_progress
        self.on_log = on_log
        self.process = None
        self.cancelled = False
    
    def log_message(self, message: str, level: str = "info"):
        """Log a message to the console and to the callback if provided"""
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        elif level == "debug":
            logger.debug(message)
            
        if self.on_log:
            self.on_log(message, level)
    
    def update_progress(self, progress: float, message: str = ""):
        """Update progress with the callback if provided"""
        if self.on_progress:
            self.on_progress(progress, message)
    
    def check_input_file(self) -> bool:
        """Check if the input file exists and is valid"""
        self.log_message(f"Checking if input file exists: {self.input_file}")
        if not os.path.exists(self.input_file):
            self.log_message(f"Input file '{self.input_file}' does not exist.", "error")
            return False
        return True
    
    def detect_silence(self, silence_file: str) -> bool:
        """Detect silence in the input file and write results to silence_file"""
        self.log_message(f"Detecting silence in '{self.input_file}'...")
        self.log_message(f"Using silence threshold: {self.silence_threshold}, minimum silence duration: {self.min_silence_duration}s")
        
        try:
            self.process = subprocess.Popen([
                'ffmpeg', 
                '-i', self.input_file, 
                '-af', f'silencedetect=noise={self.silence_threshold}:d={self.min_silence_duration}', 
                '-f', 'null', 
                '-'
            ], stderr=open(silence_file, 'w'), stdout=subprocess.PIPE, universal_newlines=True)
            
            # Monitor the process
            while self.process.poll() is None:
                if self.cancelled:
                    self.process.terminate()
                    self.log_message("Silence detection cancelled.", "warning")
                    return False
                time.sleep(0.1)
            
            if self.process.returncode != 0:
                self.log_message(f"Error during silence detection, return code: {self.process.returncode}", "error")
                return False
                
            self.log_message("Silence detection completed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            self.log_message(f"Error during silence detection: {e}", "error")
            return False
        except Exception as e:
            self.log_message(f"Unexpected error during silence detection: {e}", "error")
            return False
    
    def parse_silence_data(self, silence_file: str) -> Tuple[List[float], List[float]]:
        """Parse the silence detection output file"""
        self.log_message("Parsing silence detection results...")
        silence_starts = []
        silence_ends = []
        
        try:
            with open(silence_file, 'r') as f:
                silence_data = f.read()
                for line in silence_data.split('\n'):
                    if 'silence_start' in line:
                        try:
                            start_time = float(line.split('silence_start:')[1].strip().split(' ')[0])
                            silence_starts.append(start_time)
                            self.log_message(f"Found silence starting at: {start_time:.2f}s", "debug")
                        except (IndexError, ValueError):
                            pass
                    elif 'silence_end' in line:
                        try:
                            parts = line.split('silence_end:')[1].strip().split(' ')
                            end_time = float(parts[0])
                            silence_ends.append(end_time)
                            self.log_message(f"Found silence ending at: {end_time:.2f}s", "debug")
                        except (IndexError, ValueError):
                            pass
            
            self.log_message(f"Found {len(silence_starts)} silence start points and {len(silence_ends)} silence end points.")
            return silence_starts, silence_ends
        except Exception as e:
            self.log_message(f"Error parsing silence data: {e}", "error")
            return [], []
    
    def get_video_duration(self) -> float:
        """Get the duration of the input video"""
        self.log_message("Getting video duration...")
        try:
            duration_output = subprocess.check_output([
                'ffprobe', 
                '-v', 'error', 
                '-show_entries', 'format=duration', 
                '-of', 'default=noprint_wrappers=1:nokey=1', 
                self.input_file
            ], universal_newlines=True).strip()
            
            duration = float(duration_output)
            self.log_message(f"Video duration: {duration:.2f}s")
            return duration
        except (ValueError, subprocess.CalledProcessError) as e:
            self.log_message(f"Error getting video duration: {e}", "error")
            return 0
    
    def create_segments(self, silence_starts: List[float], silence_ends: List[float], duration: float) -> List[Tuple[float, float]]:
        """Create a list of non-silent segments"""
        self.log_message("Creating segments list of non-silent parts...")
        segments = []
        padding_sec = self.padding_ms / 1000  # Convert to seconds
        
        # Check if video starts with non-silence
        if not silence_starts or silence_starts[0] > 0:
            start_time = 0
            end_time = silence_starts[0] if silence_starts else duration
            segments.append((start_time, end_time))
            self.log_message(f"Adding initial non-silent segment: {start_time:.2f}s to {end_time:.2f}s", "debug")
        
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
                self.log_message(f"Adding non-silent segment: {start_time:.2f}s to {end_time:.2f}s", "debug")
        
        self.log_message(f"Total non-silent segments: {len(segments)}")
        return segments
    
    def create_filter_complex(self, segments: List[Tuple[float, float]]) -> str:
        """Create a filter_complex string for FFmpeg to extract and concatenate segments"""
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
    
    def check_hardware_acceleration(self) -> Optional[str]:
        """Check for available hardware acceleration options"""
        try:
            # Check for hardware acceleration
            hwaccel_output = subprocess.check_output(
                ['ffmpeg', '-hide_banner', '-hwaccels'], 
                stderr=subprocess.STDOUT, 
                universal_newlines=True
            )
            
            if 'videotoolbox' in hwaccel_output.lower():
                self.log_message("Using macOS VideoToolbox hardware acceleration")
                return 'videotoolbox'
            elif 'cuda' in hwaccel_output.lower():
                self.log_message("Using NVIDIA CUDA hardware acceleration")
                return 'nvenc'
            else:
                self.log_message("No hardware acceleration found, using libx264")
                return None
        except subprocess.CalledProcessError:
            self.log_message("Could not check hardware acceleration. Using libx264.", "warning")
            return None
    
    def process_video(self, segments: List[Tuple[float, float]]) -> bool:
        """Process the video to remove silence"""
        # Since we have a lot of segments, we'll process in chunks
        if len(segments) > 80:  # Filter complex has limitations
            return self._process_video_in_chunks(segments)
        else:
            return self._process_video_direct(segments)
    
    def _process_video_direct(self, segments: List[Tuple[float, float]]) -> bool:
        """Process the video directly with filter_complex"""
        filter_complex = self.create_filter_complex(segments)
        use_hardware_accel = self.check_hardware_acceleration()
        
        self.log_message(f"Processing video and removing silence, saving to {self.output_file}...")
        
        try:
            start_time = time.time()
            
            # Use filter_complex approach
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output files
                '-i', self.input_file,
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
            
            cmd.append(self.output_file)
            
            # Print the command for debugging
            self.log_message("Executing command: " + " ".join(cmd), "debug")
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Track progress
            duration_regex = r"Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})"
            time_regex = r"time=(\d{2}):(\d{2}):(\d{2}\.\d{2})"
            total_duration = 0
            
            # Print progress updates
            while self.process.poll() is None:
                if self.cancelled:
                    self.process.terminate()
                    self.log_message("Processing cancelled.", "warning")
                    return False
                
                line = self.process.stdout.readline()
                if line:
                    # Extract duration
                    duration_match = re.search(duration_regex, line)
                    if duration_match and total_duration == 0:
                        h, m, s = duration_match.groups()
                        total_duration = float(h) * 3600 + float(m) * 60 + float(s)
                    
                    # Extract current time
                    time_match = re.search(time_regex, line)
                    if time_match and total_duration > 0:
                        h, m, s = time_match.groups()
                        current_time = float(h) * 3600 + float(m) * 60 + float(s)
                        progress = min(current_time / total_duration, 1.0) if total_duration > 0 else 0
                        self.update_progress(progress, line.strip())
                    
                    if 'time=' in line:
                        self.log_message(f"Progress: {line.strip()}")
                    elif 'Error' in line or 'error' in line.lower():
                        self.log_message(f"Error detected: {line.strip()}", "error")
            
            rc = self.process.wait()
            
            elapsed_time = time.time() - start_time
            # Non-monotonic DTS warnings are common and don't affect the output
            if rc == 0 or (rc == 254 and "Non-monotonic DTS" in line):
                if os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0:
                    self.log_message(f"Video processing completed successfully in {elapsed_time:.2f} seconds")
                    self.log_message(f"Output file saved to: {self.output_file}")
                    size_mb = os.path.getsize(self.output_file) / (1024*1024)
                    self.log_message(f"Output file size: {size_mb:.2f} MB")
                    self.update_progress(1.0, "Complete")
                    return True
                else:
                    self.log_message(f"Error: Output file not created or empty: {self.output_file}", "error")
                    return False
            else:
                self.log_message(f"Error: FFmpeg exited with code {rc}", "error")
                # Check if the file was created despite the error
                if os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0:
                    self.log_message(f"Output file was created despite error, size: {os.path.getsize(self.output_file) / (1024*1024):.2f} MB", "warning")
                    self.update_progress(1.0, "Complete with warnings")
                    return True
                return False
        except Exception as e:
            self.log_message(f"Error during video processing: {e}", "error")
            return False
    
    def _process_video_in_chunks(self, segments: List[Tuple[float, float]]) -> bool:
        """Process the video in chunks to avoid filter_complex limitations"""
        self.log_message(f"Processing {len(segments)} segments in chunks...")
        
        # Process in chunks by extracting segments to temporary files
        temp_files = []
        chunk_size = 50  # Number of segments per chunk
        total_chunks = (len(segments) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(0, len(segments), chunk_size):
            if self.cancelled:
                self.log_message("Processing cancelled.", "warning")
                return False
                
            chunk_segments = segments[chunk_idx:chunk_idx + chunk_size]
            chunk_file = os.path.join(self.temp_dir, f"chunk_{chunk_idx}.mp4")
            temp_files.append(chunk_file)
            
            # Create filter complex for this chunk
            filter_complex = self.create_filter_complex(chunk_segments)
            
            current_chunk = chunk_idx // chunk_size + 1
            self.log_message(f"Processing chunk {current_chunk}/{total_chunks}...")
            self.update_progress(current_chunk / total_chunks * 0.8, f"Processing chunk {current_chunk}/{total_chunks}")
            
            try:
                cmd = [
                    'ffmpeg',
                    '-y',
                    '-i', self.input_file,
                    '-filter_complex', filter_complex,
                    '-map', '[v]',
                    '-map', '[a]',
                    '-c:v', 'h264_videotoolbox',  # Use hardware acceleration
                    '-c:a', 'aac',  # Re-encode audio for compatibility
                    '-b:a', '128k',
                    '-pix_fmt', 'yuv420p',
                    chunk_file
                ]
                
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
                
                # Print progress updates
                while self.process.poll() is None:
                    if self.cancelled:
                        self.process.terminate()
                        self.log_message("Processing cancelled.", "warning")
                        return False
                        
                    line = self.process.stdout.readline()
                    if line:
                        if 'time=' in line:
                            self.log_message(f"Chunk {current_chunk} progress: {line.strip()}")
                        elif 'Error' in line or 'error' in line.lower():
                            self.log_message(f"Error detected: {line.strip()}", "error")
                
                rc = self.process.wait()
                if rc != 0 and rc != 254:
                    self.log_message(f"Error processing chunk {current_chunk}: FFmpeg exited with code {rc}", "error")
                    return False
                
                self.log_message(f"Chunk {current_chunk} completed.")
            except Exception as e:
                self.log_message(f"Error processing chunk {current_chunk}: {e}", "error")
                return False
        
        # Create a file list for concatenation
        concat_list = os.path.join(self.temp_dir, "concat_list.txt")
        with open(concat_list, 'w') as f:
            for temp_file in temp_files:
                f.write(f"file '{temp_file}'\n")
        
        # Final concatenation
        self.log_message("Joining chunks into final output...")
        self.update_progress(0.9, "Joining chunks into final output")
        
        try:
            self.process = subprocess.Popen([
                'ffmpeg',
                '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_list,
                '-c', 'copy',
                self.output_file
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            
            # Print progress updates
            while self.process.poll() is None:
                if self.cancelled:
                    self.process.terminate()
                    self.log_message("Processing cancelled.", "warning")
                    return False
                    
                line = self.process.stdout.readline()
                if line:
                    if 'time=' in line:
                        self.log_message(f"Joining progress: {line.strip()}")
                    elif 'Error' in line or 'error' in line.lower():
                        self.log_message(f"Error detected: {line.strip()}", "error")
            
            rc = self.process.wait()
            if rc != 0 and rc != 254:
                self.log_message(f"Error joining chunks: FFmpeg exited with code {rc}", "error")
                return False
            
            if os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0:
                size_mb = os.path.getsize(self.output_file) / (1024*1024)
                self.log_message(f"Video processing completed successfully")
                self.log_message(f"Output file saved to: {self.output_file}")
                self.log_message(f"Output file size: {size_mb:.2f} MB")
                self.update_progress(1.0, "Complete")
                return True
            else:
                self.log_message(f"Error: Output file not created or empty: {self.output_file}", "error")
                return False
        except Exception as e:
            self.log_message(f"Error joining chunks: {e}", "error")
            return False
    
    def cancel(self):
        """Cancel the current operation"""
        self.cancelled = True
        if self.process and self.process.poll() is None:
            self.process.terminate()
    
    def remove_silence(self) -> bool:
        """Main method to remove silence from a video"""
        self.cancelled = False
        
        try:
            # Create a temporary directory for intermediate files
            self.temp_dir = tempfile.mkdtemp()
            
            # Step 1: Check if input file exists
            if not self.check_input_file():
                return False
            
            self.update_progress(0.1, "Detecting silence")
            
            # Step 2: Detect silence
            silence_file = os.path.join(self.temp_dir, "silence_data.txt")
            if not self.detect_silence(silence_file):
                return False
                
            self.update_progress(0.2, "Parsing silence data")
            
            # Step 3: Parse silence detection output
            silence_starts, silence_ends = self.parse_silence_data(silence_file)
            if not silence_starts and not silence_ends:
                self.log_message("No silence detected or error in parsing FFmpeg output.", "error")
                return False
            
            self.update_progress(0.3, "Getting video duration")
            
            # Step 4: Get video duration
            duration = self.get_video_duration()
            if duration <= 0:
                return False
            
            self.update_progress(0.4, "Creating segments")
            
            # Step 5: Create segment list
            segments = self.create_segments(silence_starts, silence_ends, duration)
            
            self.update_progress(0.5, "Processing video")
            
            # Step 6: Process the video
            return self.process_video(segments)
        except Exception as e:
            self.log_message(f"Unexpected error: {e}", "error")
            import traceback
            self.log_message(traceback.format_exc(), "error")
            return False
        finally:
            # Clean up temp directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            
            # Ensure the progress is complete
            if not self.cancelled:
                self.update_progress(1.0, "Complete") 