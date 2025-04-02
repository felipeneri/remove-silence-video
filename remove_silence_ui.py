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
        grid-columns: 2fr 1fr;
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
    
    #threshold-container, #duration-container, #padding-container {
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
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.silence_remover = None
        self.processing = False
    
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
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.title = "Video Silence Remover"
        self.add_log("Welcome to Video Silence Remover")
        self.add_log("Enter the path to a video file and click 'Start Processing'")
    
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
        
        # Update progress bar (0-100)
        progress_bar.progress = int(progress * 100)
        
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
        
        # Update UI
        self.processing = True
        start_button = self.query_one("#start-button", Button)
        cancel_button = self.query_one("#cancel-button", Button)
        start_button.disabled = True
        cancel_button.disabled = False
        
        # Reset progress
        self.update_progress(0, "Starting...")
        
        # Create silence remover
        self.silence_remover = SilenceRemover(
            input_file=input_file,
            output_file=output_file,
            silence_threshold=threshold,
            min_silence_duration=duration,
            padding_ms=padding,
            on_progress=self.update_progress,
            on_log=self.add_log
        )
        
        # Start processing in a worker thread
        self.add_log(f"Starting silence removal for {input_file}")
        self.add_log(f"Parameters: threshold={threshold}, max_silence={duration}s, padding={padding}ms")
        
        def process_complete(worker) -> None:
            result = worker.result
            self.processing = False
            start_button.disabled = False
            cancel_button.disabled = True
            
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
            self.update_progress(0, "Cancelled")
    
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
    
    app.run()

if __name__ == "__main__":
    main() 