import sys
import random
import re
from copy import deepcopy
import time
from datetime import timedelta
from enum import Enum
from pathlib import Path

import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
import typer

from utils import Data, ProcessedData, read_cached_avro

console = Console()
app = typer.Typer(help="Tool for manually classifying YouTube comments as bot/spam or normal.")

def display_comment(
  data: Data,
  index: int,
  total: int,
  default_choice: str = "n",
  auto_choice: str | None = None,
  eta_info: str | None = None,
  current_classification: bool | None = None,
):
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

  return Prompt.ask(
    "Is this a bot/spam comment?",
    choices=["y", "n", "s", "q"],
    default=default_choice,
  )

def append(df: pl.DataFrame, data: ProcessedData) -> pl.DataFrame:
  """Append a new processed comment to the DataFrame."""
  return pl.concat([df, pl.DataFrame([data.model_dump()])], how="vertical", rechunk=True)

def update_processed(df: pl.DataFrame, data: ProcessedData) -> pl.DataFrame:
  """Update an existing processed comment in the DataFrame."""
  # Filter out the existing entry
  filtered_df = df.filter(pl.col("comment_id") != data.comment_id)
  # Add the updated entry
  return append(filtered_df, data)

def apply_filters(
  comments: pl.DataFrame,
  filter_text: str | None = None,
  author_filter: str | None = None,
  author_regex: str | None = None,
):
  """Apply filters to the comments DataFrame based on command line arguments."""
  filtered_comments = comments
  
  if filter_text:
    filtered_comments = filtered_comments.filter(pl.col("content").str.contains(filter_text))
  
  if author_filter:
    filtered_comments = filtered_comments.filter(pl.col("author_name").str.contains(author_filter))
  
  if author_regex:
    try:
      # Convert to Python objects, filter with regex, then get back only matching IDs
      author_regex_pattern = re.compile(author_regex)
      comment_dicts = filtered_comments.to_dicts()
      matching_ids = [
        comment["comment_id"] for comment in comment_dicts 
        if author_regex_pattern.search(comment["author_name"]) # pyright: ignore[reportAny]
      ]
      filtered_comments = filtered_comments.filter(pl.col("comment_id").is_in(matching_ids))
    except re.error as e:
      console.print(f"[bold red]Invalid author regex pattern: {e}[/bold red]")
      sys.exit(1)
  
  return filtered_comments

def get_classification_counts(processed_df: pl.DataFrame) -> tuple[int, int]:
  """Get counts of bot and normal comments in the processed DataFrame."""
  if len(processed_df) == 0:
    return 0, 0
  
  bot_count = processed_df.filter(pl.col("is_bot_comment")).height
  normal_count = processed_df.filter(~pl.col("is_bot_comment")).height
  return bot_count, normal_count

def show_classification_summary(processed_df: pl.DataFrame):
  """Display a summary of the classification statistics."""
  total_processed = len(processed_df)
  
  if total_processed == 0:
    console.print("[bold red]No processed comments found![/bold red]")
    return
    
  bot_comments = processed_df.filter(pl.col("is_bot_comment")).height
  normal_comments = processed_df.filter(~pl.col("is_bot_comment")).height
  
  summary_table = Table(title="Classification Summary")
  summary_table.add_column("Metric", style="cyan")
  summary_table.add_column("Value", style="green")
  
  summary_table.add_row("Total Processed", str(total_processed))
  summary_table.add_row("Bot Comments", str(bot_comments))
  summary_table.add_row("Regular Comments", str(normal_comments))
  summary_table.add_row("Balance Ratio", f"{bot_comments/total_processed:.2%} bot / {normal_comments/total_processed:.2%} normal")
  summary_table.add_row("Balance Difference", str(abs(bot_comments - normal_comments)))
  
  console.print(summary_table)

class ExportFormat(str, Enum):
  CSV = "csv"
  JSON = "json"
  PARQUET = "parquet"
  AVRO = "avro"

@app.command()
def export(
  output_file: Path = typer.Argument(..., help="Output file path"), # pyright: ignore[reportCallInDefaultInitializer]
  format: ExportFormat = typer.Option(ExportFormat.CSV, "--format", "-f", help="Export format"), # pyright: ignore[reportCallInDefaultInitializer]
  bot_only: bool = typer.Option(False, "--bot-only", help="Export only bot/spam comments"), # pyright: ignore[reportCallInDefaultInitializer]
  normal_only: bool = typer.Option(False, "--normal-only", help="Export only normal comments"), # pyright: ignore[reportCallInDefaultInitializer]
):
  """
  Export classified comments to various formats.
  
  Supports CSV, JSON, Parquet, and Avro formats.
  """
  processed_df = read_cached_avro("processed.avro")
  
  if len(processed_df) == 0:
    console.print("[bold red]No processed comments to export![/bold red]")
    return
  
  # Apply filters if needed
  if bot_only:
    processed_df = processed_df.filter(pl.col("is_bot_comment"))
    console.print(f"[cyan]Exporting {len(processed_df)} bot/spam comments[/cyan]")
  elif normal_only:
    processed_df = processed_df.filter(~pl.col("is_bot_comment"))
    console.print(f"[cyan]Exporting {len(processed_df)} normal comments[/cyan]")
  else:
    console.print(f"[cyan]Exporting {len(processed_df)} comments[/cyan]")
  
  # Export to the selected format
  if format == ExportFormat.CSV:
    processed_df.write_csv(output_file)
  elif format == ExportFormat.JSON:
    processed_df.write_json(output_file)
  elif format == ExportFormat.PARQUET:
    processed_df.write_parquet(output_file)
  elif format == ExportFormat.AVRO:
    processed_df.write_avro(output_file)
  
  console.print(f"[bold green]Successfully exported data to {output_file}[/bold green]")

@app.command()
def stats():
  """
  Show statistics about the current classification dataset.
  
  Displays counts of bot/spam and normal comments, as well as balance information.
  """
  processed_df = read_cached_avro("processed.avro")
  show_classification_summary(processed_df)

@app.command()
def classify(
  default: str = typer.Option("n", "--default", "-d", help="Default choice for classification (y/n)"), # pyright: ignore[reportCallInDefaultInitializer]
  filter: str | None = typer.Option(None, "--filter", "-f", help="Filter comments by content"), # pyright: ignore[reportCallInDefaultInitializer]
  author_filter: str | None = typer.Option(None, "--author-filter", "-a", help="Filter comments by author name"), # pyright: ignore[reportCallInDefaultInitializer]
  author_regex: str | None = typer.Option(None, "--author-regex", "-r", help="Filter comments by author name using regex"), # pyright: ignore[reportCallInDefaultInitializer]
  auto: str | None = typer.Option(None, "--auto", "-A", help="Auto-classify all comments as bot/spam (y) or normal (n)"), # pyright: ignore[reportCallInDefaultInitializer]
  modify_processed: bool = typer.Option(False, "--modify", "-m", help="Modify already processed comments"), # pyright: ignore[reportCallInDefaultInitializer]
  balance: bool = typer.Option(False, "--balance", "-b", help="Auto-classify to achieve balanced dataset"), # pyright: ignore[reportCallInDefaultInitializer]
):
  """
  Manually classify YouTube comments as bot/spam or normal.
  
  This tool allows for interactive classification of comments with various filtering options.
  Use --auto for automatic classification or --balance to auto-balance the dataset.
  """
  # Load data
  comments = read_cached_avro("comments.avro")
  global original_comments
  original_comments = deepcopy(comments)
  processed_df = read_cached_avro("processed.avro")
  
  # Get current classification counts
  bot_count, normal_count = get_classification_counts(processed_df)
  
  if modify_processed:
    # We're modifying processed comments
    if len(processed_df) == 0:
      console.print("[bold red]No processed comments found to modify![/bold red]")
      return
    
    # Apply filters to processed comments
    working_df = apply_filters(processed_df, filter, author_filter, author_regex)
    
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
    working_df = apply_filters(comments, filter, author_filter, author_regex)
    
    if len(working_df) == 0:
      console.print("[bold red]No comments match your filter criteria or all comments have been processed![/bold red]")
      return
    
    console.print(f"[bold]Found {len(working_df)} unprocessed comments matching your criteria[/bold]")
  
  # Auto mode message
  if auto:
    console.print(f"[bold cyan]AUTO MODE: All comments will be classified as {'bot/spam' if auto == 'y' else 'normal'}[/bold cyan]")
    console.print("[bold cyan]Press Ctrl+C to stop at any time[/bold cyan]")
    console.print("[bold cyan]Processing will begin in 3 seconds...[/bold cyan]")
    time.sleep(3)
  elif balance:
    console.print(f"[bold cyan]BALANCE MODE: Currently {bot_count} bot/spam and {normal_count} normal comments[/bold cyan]")
    if bot_count > normal_count:
      console.print("[bold cyan]Will classify comments as normal until balanced[/bold cyan]")
    elif normal_count > bot_count:
      console.print("[bold cyan]Will classify comments as bot/spam until balanced[/bold cyan]")
    else:
      console.print("[bold cyan]Dataset is already balanced. Will use default classification.[/bold cyan]")
    console.print("[bold cyan]Press Ctrl+C to stop at any time[/bold cyan]")
    console.print("[bold cyan]Processing will begin in 3 seconds...[/bold cyan]")
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
      
      if modify_processed:
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
      display_default = default
      if modify_processed and current_classification is not None:
        display_default = "y" if current_classification else "n"
      
      # Handle balance mode - update bot_count and normal_count first
      auto_balance_choice = None
      if balance and not modify_processed:
        bot_count, normal_count = get_classification_counts(processed_df)
        if bot_count > normal_count:
          # More bot comments, so classify as normal
          auto_balance_choice = "n"
        elif normal_count > bot_count:
          # More normal comments, so classify as bot
          auto_balance_choice = "y"
        # If equal, don't auto-classify
      
      choice = display_comment(
        data, 
        i, 
        len(comments_list), 
        default_choice=display_default, 
        auto_choice=auto or auto_balance_choice,
        eta_info=eta_info,
        current_classification=current_classification if modify_processed else None
      )
      
      if choice == "q":
        console.print("[yellow]Quitting...[/yellow]")
        break
      elif choice == "s" and not auto and not auto_balance_choice:
        console.print("[yellow]Skipping...[/yellow]")
        continue
      
      # Process the comment
      new_processed_data = ProcessedData(
        is_bot_comment=(choice == "y"),
        **data.model_dump() # pyright: ignore[reportAny]
      )
      
      if modify_processed:
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
        
        # If in balance mode, show current balance status
        if balance:
          bot_count, normal_count = get_classification_counts(processed_df)
          console.print(f"[cyan]Current balance: {bot_count} bot/spam, {normal_count} normal (difference: {abs(bot_count - normal_count)})[/cyan]")
          if bot_count == normal_count:
            console.print("[bold green]Dataset is now perfectly balanced![/bold green]")
  except KeyboardInterrupt:
    console.print("[bold red]Process interrupted by user[/bold red]")
  
  # Show summary
  total_processed = len(processed_df)
  bot_comments = processed_df.filter(pl.col("is_bot_comment")).height
  normal_comments = processed_df.filter(~pl.col("is_bot_comment")).height
  
  summary_table = Table(title="Classification Summary")
  summary_table.add_column("Metric", style="cyan")
  summary_table.add_column("Value", style="green")
  
  summary_table.add_row("Total Processed", str(total_processed))
  summary_table.add_row("Bot Comments", str(bot_comments))
  summary_table.add_row("Regular Comments", str(normal_comments))
  summary_table.add_row("Balance Ratio", f"{bot_comments/total_processed:.2%} bot / {normal_comments/total_processed:.2%} normal")
  summary_table.add_row(f"{'Modified' if modify_processed else 'Classified'} This Session", str(processed_count))
  
  # Add time statistics to summary
  if processed_count > 0:
    total_time = time.time() - start_time
    seconds_per_comment = total_time / processed_count
    
    # Format time as HH:MM:SS
    formatted_time = str(timedelta(seconds=int(total_time)))
    
    summary_table.add_row("Total Time", formatted_time)
    summary_table.add_row("Seconds per Comment", f"{seconds_per_comment:.2f}s")
    
    # Add projected time for all remaining comments
    if not modify_processed:
      remaining_all = len(comments) - processed_count
      if remaining_all > 0:
        projected_time = remaining_all * seconds_per_comment
        projected_formatted = str(timedelta(seconds=int(projected_time)))
        summary_table.add_row("Projected Time for All Remaining", projected_formatted)
  
  # Show mode and filter information
  console.print("\n[bold]Session Settings:[/bold]")
  console.print(f"Target: {'Modifying existing entries' if modify_processed else 'Processing new entries'}")
  
  if auto:
    console.print(f"Mode: Auto (forced '{auto}' classification)")
  elif balance:
    console.print("Mode: Balance (auto-classifying to achieve equal counts)")
  else:
    console.print(f"Mode: Manual (default: '{default}')")
    
  if any([filter, author_filter, author_regex]):
    if filter:
      console.print(f"Content filter: '{filter}'")
    if author_filter:
      console.print(f"Author filter: '{author_filter}'")
    if author_regex:
      console.print(f"Author regex: '{author_regex}'")
  
  console.print(summary_table)

if __name__ == "__main__":
  app(prog_name="classify-manual")