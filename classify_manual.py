import os
import sys
import random
import argparse
from copy import deepcopy
from pathlib import Path
import time
from datetime import timedelta

import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.progress import Progress

from utils import Data, ProcessedData, read_cached_avro

console = Console()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Manually classify comments with configurable options.")
    parser.add_argument("--default", choices=["y", "n"], default="n", 
                        help="Default classification (y=bot/spam, n=normal)")
    parser.add_argument("--filter", type=str, default=None,
                        help="Filter comments containing this text")
    parser.add_argument("--author-filter", type=str, default=None,
                        help="Filter comments by author name containing this text")
    parser.add_argument("--auto", choices=["y", "n"], default=None,
                        help="Auto mode: automatically classify all comments as specified (y=bot/spam, n=normal)")
    parser.add_argument("--modify-processed", action="store_true",
                        help="Modify already processed comments instead of unprocessed ones")
    return parser.parse_args()

def display_comment(data: Data, index: int, total: int, default_choice="n", auto_choice=None, eta_info=None, current_classification=None):
    """Display a comment with rich formatting for manual classification."""
    console.clear()
    
    # Create a table for video information
    video_table = Table(show_header=False, box=None)
    video_table.add_row("Video Title:", data.video_title)
    video_table.add_row("Video Author:", data.video_author)
    
    # Create a table for comment information
    comment_table = Table(show_header=False, box=None)
    comment_table.add_row("Author:", data.author_name)
    comment_table.add_row("Comment Length:", str(len(data.content)))
    
    # If there's a parent comment, try to show it
    parent_info = ""
    if data.parent_id:
        try:
            parent = original_comments.filter(pl.col('comment_id') == data.parent_id).to_dicts()[0]
            parent = Data.model_validate(parent)
            parent_info = f"\n[dim]Replying to {parent.author_name}:[/dim]\n[dim]{parent.content}[/dim]"
        except (IndexError, KeyError):
            parent_info = "\n[dim]Replying to unknown comment[/dim]"
    
    # Display all information
    console.print(Panel(video_table, title="Video Information"))
    console.print(Panel(comment_table, title="Comment Information"))
    console.print(Panel(f"{parent_info}\n{data.content}", title="Content"))
    
    # Display progress and ETA
    progress_text = f"Comment {index} of {total}"
    if eta_info:
        progress_text += f" | {eta_info}"
    if current_classification is not None:
        classification_text = "bot/spam" if current_classification else "normal"
        progress_text += f" | Current classification: [bold]{classification_text}[/bold]"
    console.print(Panel(progress_text, title="Progress"))
    
    # If in auto mode, use the auto choice
    if auto_choice:
        console.print(f"[bold]Auto classifying as {'bot/spam' if auto_choice == 'y' else 'normal'}[/bold]")
        return auto_choice
    
    # Ask for classification
    choice = Prompt.ask(
        "Is this a bot/spam comment?",
        choices=["y", "n", "s", "q"],
        default=default_choice
    )
    
    return choice

def append(df: pl.DataFrame, data: ProcessedData) -> pl.DataFrame:
    """Append a new processed comment to the DataFrame."""
    return pl.concat([df, pl.DataFrame([data.model_dump()])], how="vertical", rechunk=True)

def update_processed(df: pl.DataFrame, data: ProcessedData) -> pl.DataFrame:
    """Update an existing processed comment in the DataFrame."""
    # Filter out the existing entry
    filtered_df = df.filter(pl.col("comment_id") != data.comment_id)
    # Add the updated entry
    return append(filtered_df, data)

def apply_filters(comments, args):
    """Apply filters to the comments DataFrame based on command line arguments."""
    filtered_comments = comments
    
    if args.filter:
        filtered_comments = filtered_comments.filter(pl.col("content").str.contains(args.filter))
    
    if args.author_filter:
        filtered_comments = filtered_comments.filter(pl.col("author_name").str.contains(args.author_filter))
    
    return filtered_comments

def main():
    args = parse_args()
    
    # Set random seed
    random.seed()
    
    # Load data
    comments = read_cached_avro("comments.avro")
    global original_comments
    original_comments = deepcopy(comments)
    processed_df = read_cached_avro("processed.avro")
    
    if args.modify_processed:
        # We're modifying processed comments
        if len(processed_df) == 0:
            console.print("[bold red]No processed comments found to modify![/bold red]")
            return
        
        # Apply filters to processed comments
        working_df = apply_filters(processed_df, args)
        
        if len(working_df) == 0:
            console.print("[bold red]No processed comments match your filter criteria![/bold red]")
            return
        
        console.print(f"[bold]Found {len(working_df)} processed comments matching your criteria for modification[/bold]")
    else:
        # We're processing new comments
        # Remove already processed comments from the comments DataFrame
        if len(processed_df) > 0:
            processed_ids = set(processed_df.select("comment_id").to_series().to_list())
            comments = comments.filter(~pl.col("comment_id").is_in(processed_ids))
        
        # Apply filters
        working_df = apply_filters(comments, args)
        
        if len(working_df) == 0:
            console.print("[bold red]No comments match your filter criteria or all comments have been processed![/bold red]")
            return
        
        console.print(f"[bold]Found {len(working_df)} unprocessed comments matching your criteria[/bold]")
    
    # Auto mode message
    if args.auto:
        console.print(f"[bold cyan]AUTO MODE: All comments will be classified as {'bot/spam' if args.auto == 'y' else 'normal'}[/bold cyan]")
        console.print(f"[bold cyan]Press Ctrl+C to stop at any time[/bold cyan]")
        console.print(f"[bold cyan]Processing will begin in 3 seconds...[/bold cyan]")
        time.sleep(3)
    
    # Convert to list of dictionaries for random access
    comments_list = working_df.to_dicts()
    random.shuffle(comments_list)
    
    # Process comments in random order
    processed_count = 0
    start_time = time.time()
    seconds_per_comment = 0
    eta_info = None
    
    try:
        for i, row in enumerate(comments_list, 1):
            comment_start_time = time.time()
            
            if args.modify_processed:
                # For modifying existing entries, we already have ProcessedData
                processed_data = ProcessedData.model_validate(row)
                data = Data.model_validate(row)
                current_classification = processed_data.is_bot_comment
            else:
                # For new entries, we have Data
                data = Data.model_validate(row)
                current_classification = None
            
            # Calculate ETA if we have processed at least one comment
            if processed_count > 0:
                seconds_per_comment = (time.time() - start_time) / processed_count
                remaining_comments = len(comments_list) - i + 1
                eta_seconds = int(seconds_per_comment * remaining_comments)
                eta = str(timedelta(seconds=eta_seconds))
                eta_info = f"[cyan]Avg: {seconds_per_comment:.2f}s per comment | ETA: {eta}[/cyan]"
                
                # Calculate completion percentage
                percent_complete = (i - 1) / len(comments_list) * 100
                eta_info += f" | {percent_complete:.1f}% complete"
            
            # Set default choice based on current classification if modifying
            display_default = args.default
            if args.modify_processed and current_classification is not None:
                display_default = "y" if current_classification else "n"
            
            choice = display_comment(
                data, 
                i, 
                len(comments_list), 
                default_choice=display_default, 
                auto_choice=args.auto,
                eta_info=eta_info,
                current_classification=current_classification if args.modify_processed else None
            )
            
            if choice == "q":
                console.print("[yellow]Quitting...[/yellow]")
                break
            elif choice == "s" and not args.auto:
                console.print("[yellow]Skipping...[/yellow]")
                continue
            
            # Process the comment
            new_processed_data = ProcessedData(
                is_bot_comment=(choice == "y"),
                **data.model_dump()
            )
            
            if args.modify_processed:
                # Update existing entry
                processed_df = update_processed(processed_df, new_processed_data)
                action_text = "Updated"
            else:
                # Add new entry
                processed_df = append(processed_df, new_processed_data)
                action_text = "Saved"
            
            processed_df.write_avro("processed.avro")
            
            processed_count += 1
            comment_time = time.time() - comment_start_time
            console.print(f"[green]{action_text} classification for comment {i} (Total modified: {processed_count}, Time: {comment_time:.2f}s)[/green]")
            
            # Show estimated completion time after each comment
            if processed_count > 0:
                completion_time = start_time + (seconds_per_comment * len(comments_list))
                estimated_finish = time.strftime("%H:%M:%S", time.localtime(completion_time))
                console.print(f"[cyan]Estimated completion time: {estimated_finish}[/cyan]")
    except KeyboardInterrupt:
        console.print("[bold red]Process interrupted by user[/bold red]")
    
    # Show summary
    total_processed = len(processed_df)
    bot_comments = processed_df.filter(pl.col("is_bot_comment") == True).height
    
    summary_table = Table(title="Classification Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Total Processed", str(total_processed))
    summary_table.add_row("Bot Comments", str(bot_comments))
    summary_table.add_row("Regular Comments", str(total_processed - bot_comments))
    summary_table.add_row(f"{'Modified' if args.modify_processed else 'Classified'} This Session", str(processed_count))
    
    # Add time statistics to summary
    if processed_count > 0:
        total_time = time.time() - start_time
        seconds_per_comment = total_time / processed_count
        
        # Format time as HH:MM:SS
        formatted_time = str(timedelta(seconds=int(total_time)))
        
        summary_table.add_row("Total Time", formatted_time)
        summary_table.add_row("Seconds per Comment", f"{seconds_per_comment:.2f}s")
        
        # Add projected time for all remaining comments
        if not args.modify_processed:
            remaining_all = len(comments) - processed_count
            if remaining_all > 0:
                projected_time = remaining_all * seconds_per_comment
                projected_formatted = str(timedelta(seconds=int(projected_time)))
                summary_table.add_row("Projected Time for All Remaining", projected_formatted)
    
    # Show mode and filter information
    console.print("\n[bold]Session Settings:[/bold]")
    console.print(f"Target: {'Modifying existing entries' if args.modify_processed else 'Processing new entries'}")
    
    if args.auto:
        console.print(f"Mode: Auto (forced '{args.auto}' classification)")
    else:
        console.print(f"Mode: Manual (default: '{args.default}')")
        
    if any([args.filter, args.author_filter]):
        if args.filter:
            console.print(f"Content filter: '{args.filter}'")
        if args.author_filter:
            console.print(f"Author filter: '{args.author_filter}'")
    
    console.print(summary_table)

if __name__ == "__main__":
    main() 