import argparse
import os
import sys
import logging
import time
from datetime import datetime
from typing import List, Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Static, Button, Input, Label, ProgressBar, Log
from textual.binding import Binding
from rich.text import Text
from rich.logging import RichHandler

# Import the silence remover class
from silence_remover import SilenceRemover

# Set up logging
FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger("silence_remover_ui")

# Create a log file handler
if not os.path.exists("logs"):
    os.makedirs("logs")
log_file = os.path.join("logs", f"silence_remover_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

class SilenceRemoverApp(App):
    """A Textual app to remove silence from videos."""
    
    CSS = """
    #app-grid {
        layout: grid;
        grid-size: 2;
        grid-rows: 2fr auto auto;
        grid-columns: 1fr 1fr;
        height: 100%;
    }
    
    #input-container {
        grid-columns: 1;
        grid-rows: 1;
        width: 100%;
        height: 100%;
        border: round $primary;
        padding: 1;
        margin: 1;
        overflow-y: auto;
    }
    
    #log-container {
        grid-columns: 2;
        grid-rows: 1-3;
        width: 100%;
        height: 100%;
        border: round $primary;
        padding: 1;
        margin: 1;
    }

    #log-title {
        height: auto;
        content-align: center middle;
        text-style: bold;
        margin-bottom: 1;
    }
    
    #log-viewer-container {
        width: 100%;
        height: 1fr;
        overflow-y: auto;
        overflow-x: auto;
    }
    
    #progress-container {
        grid-columns: 1;
        grid-rows: 3;
        width: 100%;
        height: auto;
        border: round $primary;
        padding: 1;
        margin: 1;
    }
    
    #button-container {
        grid-columns: 1;
        grid-rows: 2;
        width: 100%;
        height: auto;
        border: round $primary;
        padding: 1;
        margin: 1;
        align: center middle;
    }
    
    .input-label {
        width: 30%;
        height: 3;
        content-align: left middle;
        padding: 0 1;
    }
    
    .input-field {
        width: 70%;
        height: 3;
    }
    
    .input-row {
        width: 100%;
        height: auto;
        align: left middle;
        padding: 0 0 1 0;
    }
    
    #threshold-container, #duration-container, #padding-container,
    #codec-container, #bitrate-container, #quality-container, #preset-container {
        width: 100%;
        layout: horizontal;
    }
    
    #title {
        content-align: center middle;
        text-style: bold;
        height: 3;
    }
    
    #progress-label {
        width: 100%;
        height: 2;
        content-align: center middle;
    }
    
    #time-info {
        width: 100%;
        height: 2;
        content-align: center middle;
        color: $text-muted;
    }
    
    #progress-bar {
        width: 100%;
        height: 2;
    }
    
    #start-button {
        min-width: 15;
        margin: 0 2;
    }
    
    #cancel-button {
        min-width: 15;
        margin: 0 2;
    }
    
    #log-viewer {
        width: 100%;
        height: 1fr;
        background: $surface;
        color: $text;
        overflow-x: auto;
        text-wrap: wrap;
    }
    
    Log {
        background: $surface;
        color: $text;
        overflow-x: auto;
        text-wrap: wrap;
    }
    
    #compression-title {
        content-align: center middle;
        text-style: bold;
        margin-top: 1;
        height: 2;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("s", "start", "Start", show=True),
        Binding("c", "cancel", "Cancel", show=False),
    ]
    
    # Add command line arguments as class attributes
    input_file = ""
    output_file = ""
    threshold = "-30dB"
    duration = "1.0"
    padding = "200"
    codec = ""
    bitrate = ""
    quality = ""
    preset = ""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.silence_remover = None
        self.processing = False
        self.start_time = None
        self.last_progress = 0
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        
        with Container(id="app-grid"):
            with Container(id="input-container"):
                yield Static("Video Silence Remover", id="title")
                
                with Horizontal(classes="input-row"):
                    yield Label("Input Video:", classes="input-label")
                    yield Input(placeholder="Path to input video file", id="input-file", classes="input-field", value=self.input_file)
                
                with Horizontal(classes="input-row"):
                    yield Label("Output Video:", classes="input-label")
                    yield Input(placeholder="Path to output video file (optional)", id="output-file", classes="input-field", value=self.output_file)
                
                with Horizontal(classes="input-row", id="threshold-container"):
                    yield Label("Silence Threshold:", classes="input-label")
                    yield Input(value=self.threshold, placeholder="-30dB", id="threshold", classes="input-field")
                
                with Horizontal(classes="input-row", id="duration-container"):
                    yield Label("Max Silence Duration:", classes="input-label")
                    yield Input(value=self.duration, placeholder="1.0 seconds", id="duration", classes="input-field")
                
                with Horizontal(classes="input-row", id="padding-container"):
                    yield Label("Padding:", classes="input-label")
                    yield Input(value=self.padding, placeholder="200 milliseconds", id="padding", classes="input-field")
                
                yield Static("Compression Settings", id="compression-title")
                
                with Horizontal(classes="input-row", id="codec-container"):
                    yield Label("Video Codec:", classes="input-label")
                    yield Input(value=self.codec, placeholder="libx264, h264_videotoolbox", id="codec", classes="input-field")
                
                with Horizontal(classes="input-row", id="bitrate-container"):
                    yield Label("Bitrate:", classes="input-label")
                    yield Input(value=self.bitrate, placeholder="5M, 500k", id="bitrate", classes="input-field")
                
                with Horizontal(classes="input-row", id="quality-container"):
                    yield Label("Quality (CRF):", classes="input-label")
                    yield Input(value=self.quality, placeholder="23 (18-28, lower is better)", id="quality", classes="input-field")
                
                with Horizontal(classes="input-row", id="preset-container"):
                    yield Label("Encoding Preset:", classes="input-label")
                    yield Input(value=self.preset, placeholder="medium, fast, slow", id="preset", classes="input-field")
            
            with Container(id="log-container"):
                yield Static("Welcome to Video Silence Remover", id="log-title")
                with Container(id="log-viewer-container"):
                    yield Log(highlight=True, id="log-viewer")
            
            with Horizontal(id="button-container"):
                yield Button("Start Processing", variant="primary", id="start-button")
                yield Button("Cancel", variant="error", id="cancel-button", disabled=True)
            
            with Container(id="progress-container"):
                yield Label("Ready to start...", id="progress-label")
                yield ProgressBar(total=100, id="progress-bar")
                yield Label("Time: --:--:-- (est. --:--:--)", id="time-info")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.title = "Video Silence Remover"
        self.add_log("Welcome to Video Silence Remover")
        self.add_log("Enter the path to a video file and click 'Start Processing'")
        self.add_log("TIP: Use compression settings for smaller output files.")
    
    def add_log(self, message: str, level: str = "info") -> None:
        """Add a message to the log viewer."""
        # Add timestamp
        timestamp = time.strftime("[%H:%M:%S]")
        
        # Get log widget
        log_widget = self.query_one("#log-viewer", Log)
        
        # Format each log message and ensure it's treated as a separate entry
        formatted_message = f"{timestamp} "
        
        # Set message style based on level and write to log
        if level == "error":
            formatted_message += f"[red]{message}[/red]"
            log.error(message)
        elif level == "warning":
            formatted_message += f"[yellow]{message}[/yellow]"
            log.warning(message)
        elif level == "debug":
            formatted_message += f"[dim]{message}[/dim]"
            log.debug(message)
        else:
            formatted_message += message
            log.info(message)
        
        # Write the formatted text to the log, ensuring it's a separate entry
        log_widget.write(formatted_message + "\n")
        
        # Ensure scroll to bottom happens after the UI updates
        def ensure_scroll():
            log_widget.scroll_end(animate=False)
        
        self.call_later(ensure_scroll)
    
    def update_progress(self, progress: float, message: str = "") -> None:
        """Update the progress bar and label."""
        progress_bar = self.query_one("#progress-bar", ProgressBar)
        progress_label = self.query_one("#progress-label", Label)
        time_info = self.query_one("#time-info", Label)
        
        # Initialize start time if this is the first progress update
        if self.start_time is None and progress > 0:
            self.start_time = time.time()
        
        # Update progress bar (0-100)
        progress_bar.progress = int(progress * 100)
        self.last_progress = progress
        
        # Update label with percentage
        percentage = int(progress * 100)
        if message:
            # Trim message if too long
            if len(message) > 30:
                short_message = message[:27] + "..."
            else:
                short_message = message
            progress_label.update(f"{short_message} - {percentage}%")
        else:
            progress_label.update(f"Progress: {percentage}%")
            
        # Update time information
        if self.start_time is not None and progress > 0:
            # Calculate elapsed time
            elapsed_seconds = time.time() - self.start_time
            elapsed_str = self._format_time(elapsed_seconds)
            
            # Calculate estimated total time and remaining time
            if progress > 0.01:  # Avoid division by very small numbers
                estimated_total = elapsed_seconds / progress
                remaining_seconds = estimated_total - elapsed_seconds
                remaining_str = self._format_time(remaining_seconds)
                time_info.update(f"Elapsed: {elapsed_str} (est. remaining: {remaining_str})")
            else:
                time_info.update(f"Elapsed: {elapsed_str} (est. remaining: calculating...)")
        else:
            time_info.update("Time: --:--:-- (est. --:--:--)")
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS format."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Called when a button is pressed."""
        if event.button.id == "start-button":
            self.action_start()
        elif event.button.id == "cancel-button":
            self.action_cancel()
    
    def action_start(self) -> None:
        """Start processing the video."""
        if self.processing:
            return
        
        # Get input values
        input_file = self.query_one("#input-file", Input).value
        output_file = self.query_one("#output-file", Input).value or None
        threshold = self.query_one("#threshold", Input).value
        duration = self.query_one("#duration", Input).value
        padding = self.query_one("#padding", Input).value
        
        # Get compression values
        codec = self.query_one("#codec", Input).value
        bitrate = self.query_one("#bitrate", Input).value
        quality = self.query_one("#quality", Input).value
        preset = self.query_one("#preset", Input).value
        
        # Validate input
        if not input_file:
            self.add_log("Please enter an input file path", "error")
            return
        
        if not os.path.exists(input_file):
            self.add_log(f"Input file does not exist: {input_file}", "error")
            return
        
        try:
            duration = float(duration)
        except ValueError:
            self.add_log(f"Invalid duration value: {duration}", "error")
            return
        
        try:
            padding = int(padding)
        except ValueError:
            self.add_log(f"Invalid padding value: {padding}", "error")
            return
        
        # Validate quality if provided
        if quality:
            try:
                quality = int(quality)
                if quality < 0:
                    self.add_log(f"Quality value must be positive: {quality}", "error")
                    return
            except ValueError:
                self.add_log(f"Invalid quality value: {quality}", "error")
                return
        
        # Update UI
        self.processing = True
        start_button = self.query_one("#start-button", Button)
        cancel_button = self.query_one("#cancel-button", Button)
        start_button.disabled = True
        cancel_button.disabled = False
        
        # Reset progress and timer
        self.start_time = None
        self.last_progress = 0
        self.update_progress(0, "Starting...")
        
        # Create silence remover
        self.silence_remover = SilenceRemover(
            input_file=input_file,
            output_file=output_file,
            silence_threshold=threshold,
            min_silence_duration=duration,
            padding_ms=padding,
            codec=codec,
            bitrate=bitrate,
            quality=quality,
            preset=preset,
            on_progress=self.update_progress,
            on_log=self.add_log
        )
        
        # Start processing in a worker thread
        self.add_log(f"Starting silence removal for {input_file}")
        self.add_log(f"Parameters: threshold={threshold}, max_silence={duration}s, padding={padding}ms")
        
        # Log compression settings if provided
        if codec:
            self.add_log(f"Using codec: {codec}")
        if bitrate:
            self.add_log(f"Using bitrate: {bitrate}")
        if quality:
            self.add_log(f"Using quality setting: {quality}")
        if preset:
            self.add_log(f"Using preset: {preset}")
        
        def process_complete(worker) -> None:
            result = worker.result
            self.processing = False
            start_button.disabled = False
            cancel_button.disabled = True
            
            # Calculate and show total processing time
            if self.start_time is not None:
                total_time = time.time() - self.start_time
                self.add_log(f"Total processing time: {self._format_time(total_time)}")
            
            if result:
                self.add_log("Silence removal completed successfully!")
                if self.silence_remover and self.silence_remover.output_file:
                    self.add_log(f"Output saved to: {self.silence_remover.output_file}")
            else:
                self.add_log("Silence removal failed. See log for details.", "error")
        
        worker = self.run_worker(
            self.silence_remover.remove_silence, 
            process_complete,
            thread=True
        )
    
    def action_cancel(self) -> None:
        """Cancel the current operation."""
        if self.processing and self.silence_remover:
            self.add_log("Cancelling operation...", "warning")
            self.silence_remover.cancel()
            self.processing = False
            self.query_one("#start-button", Button).disabled = False
            self.query_one("#cancel-button", Button).disabled = True
            
            # Reset timer and progress
            self.start_time = None
            self.last_progress = 0
            self.update_progress(0, "Cancelled")
            self.query_one("#time-info", Label).update("Time: --:--:-- (est. --:--:--)")
    
    def action_quit(self) -> None:
        """Quit the application."""
        if self.processing and self.silence_remover:
            self.silence_remover.cancel()
        self.exit()

def main():
    """Run the app."""
    parser = argparse.ArgumentParser(description="Video Silence Remover")
    parser.add_argument("--input", help="Input video file path")
    parser.add_argument("--output", help="Output video file path")
    parser.add_argument("--threshold", default="-30dB", help="Silence threshold (default: -30dB)")
    parser.add_argument("--duration", type=float, default=1.0, 
                        help="Maximum silence duration to keep in seconds (default: 1.0)")
    parser.add_argument("--padding", type=int, default=200,
                        help="Padding in milliseconds before and after non-silent parts (default: 200ms)")
    parser.add_argument("--codec", help="Specific video codec to use (e.g., h264_videotoolbox, libx264)")
    parser.add_argument("--bitrate", help="Video bitrate (e.g., 5M for 5 megabits/s)")
    parser.add_argument("--quality", help="Quality setting (CRF, lower is better quality, 18-28 is typical)")
    parser.add_argument("--preset", help="Encoding preset (e.g., medium, fast, slow for libx264)")
    
    args = parser.parse_args()
    
    # Create app class with command line arguments as attributes
    app = SilenceRemoverApp()
    
    # Set the command line arguments as class attributes
    if args.input:
        app.input_file = args.input
    if args.output:
        app.output_file = args.output
    if args.threshold:
        app.threshold = args.threshold
    if args.duration:
        app.duration = str(args.duration)
    if args.padding:
        app.padding = str(args.padding)
    if args.codec:
        app.codec = args.codec
    if args.bitrate:
        app.bitrate = args.bitrate
    if args.quality:
        app.quality = args.quality
    if args.preset:
        app.preset = args.preset
    
    app.run()

if __name__ == "__main__":
    main() 