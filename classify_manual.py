import os
import sys
import random
from copy import deepcopy
from pathlib import Path

import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from utils import Data, ProcessedData, read_cached_avro

console = Console()

def display_comment(data: Data, index: int, total: int):
    """Display a comment with rich formatting for manual classification."""
    console.clear()
    
    # Create a table for video information
    video_table = Table(show_header=False, box=None)
    video_table.add_row("Video Title:", data.video_title)
    video_table.add_row("Video Author:", data.video_author)
    
    # Create a table for comment information
    comment_table = Table(show_header=False, box=None)
    comment_table.add_row("Author:", data.author_name)
    
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
    console.print(f"Comment {index} of {total}")
    
    # Ask for classification
    choice = Prompt.ask(
        "Is this a bot/spam comment?",
        choices=["y", "n", "s", "q"],
        default="n"
    )
    
    return choice

def append(df: pl.DataFrame, data: ProcessedData) -> pl.DataFrame:
    """Append a new processed comment to the DataFrame."""
    return pl.concat([df, pl.DataFrame([data.model_dump()])], how="vertical", rechunk=True)

def main():
    # Load data
    comments = read_cached_avro("comments.avro")
    global original_comments
    original_comments = deepcopy(comments)
    processed_df = read_cached_avro("processed.avro")
    
    # Remove already processed comments from the comments DataFrame
    if len(processed_df) > 0:
        processed_ids = set(processed_df.select("comment_id").to_series().to_list())
        comments = comments.filter(~pl.col("comment_id").is_in(processed_ids))
    
    if len(comments) == 0:
        console.print("[bold green]All comments have been processed![/bold green]")
        return
    
    console.print(f"[bold]Found {len(comments)} unprocessed comments[/bold]")
    
    # Convert to list of dictionaries for random access
    comments_list = comments.to_dicts()
    random.shuffle(comments_list)
    
    # Process comments in random order
    processed_count = 0
    for i, row in enumerate(comments_list, 1):
        data = Data.model_validate(row)
        choice = display_comment(data, i, len(comments_list))
        
        if choice == "q":
            console.print("[yellow]Quitting...[/yellow]")
            break
        elif choice == "s":
            console.print("[yellow]Skipping...[/yellow]")
            continue
        
        # Process the comment
        processed_data = ProcessedData(
            is_bot_comment=(choice == "y"),
            **data.model_dump()
        )
        
        processed_df = append(processed_df, processed_data)
        processed_df.write_avro("processed.avro")
        
        processed_count += 1
        console.print(f"[green]Saved classification for comment {i} (Total classified: {processed_count})[/green]")
    
    # Show summary
    total_processed = len(processed_df)
    bot_comments = processed_df.filter(pl.col("is_bot_comment") == True).height
    
    summary_table = Table(title="Classification Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Count", style="green")
    
    summary_table.add_row("Total Processed", str(total_processed))
    summary_table.add_row("Bot Comments", str(bot_comments))
    summary_table.add_row("Regular Comments", str(total_processed - bot_comments))
    summary_table.add_row("Classified This Session", str(processed_count))
    
    console.print(summary_table)

if __name__ == "__main__":
    # Set random seed
    random.seed()
    main() 