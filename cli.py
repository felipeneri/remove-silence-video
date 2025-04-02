#!/usr/bin/env python3
import argparse
import os
import sys
import logging
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.logging import RichHandler

# Import the silence remover class
from silence_remover import SilenceRemover

# Set up logging
FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger("silence_remover_cli")

console = Console()

def main():
    """Run the silence remover from the command line."""
    parser = argparse.ArgumentParser(description="Video Silence Remover")
    parser.add_argument("input", help="Input video file path")
    parser.add_argument("-o", "--output", help="Output video file path")
    parser.add_argument("-t", "--threshold", default="-30dB", help="Silence threshold (default: -30dB)")
    parser.add_argument("-d", "--duration", type=float, default=1.0, 
                        help="Maximum silence duration to keep in seconds (default: 1.0)")
    parser.add_argument("-p", "--padding", type=int, default=200,
                        help="Padding in milliseconds before and after non-silent parts (default: 200ms)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show more detailed output")
    
    args = parser.parse_args()
    
    # Configure logging based on verbosity
    if args.verbose:
        log.setLevel(logging.DEBUG)
    
    # Create a progress object
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    )
    
    # Function to update progress
    task_id = None
    
    def update_progress(progress_value, message):
        nonlocal task_id
        if task_id is None:
            task_id = progress.add_task("[cyan]Processing video...", total=100)
        progress.update(task_id, completed=progress_value * 100, description=message)
    
    # Function to log messages
    def log_message(message, level="info"):
        if level == "info":
            log.info(message)
        elif level == "warning":
            log.warning(message)
        elif level == "error":
            log.error(message)
        elif level == "debug":
            log.debug(message)
    
    console.print(f"[bold green]Video Silence Remover[/bold green]")
    console.print(f"Processing file: [cyan]{args.input}[/cyan]")
    console.print(f"Parameters: threshold=[yellow]{args.threshold}[/yellow], "
                 f"max_silence=[yellow]{args.duration}s[/yellow], "
                 f"padding=[yellow]{args.padding}ms[/yellow]")
    
    # Create silence remover
    remover = SilenceRemover(
        input_file=args.input,
        output_file=args.output,
        silence_threshold=args.threshold,
        min_silence_duration=args.duration,
        padding_ms=args.padding,
        on_progress=update_progress,
        on_log=log_message
    )
    
    # Process the video
    with progress:
        success = remover.remove_silence()
    
    if success:
        console.print(f"[bold green]Operation completed successfully![/bold green]")
        console.print(f"Output saved to: [cyan]{remover.output_file}[/cyan]")
        return 0
    else:
        console.print(f"[bold red]Operation failed![/bold red]")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 