# noMoreSpam

A bot comment detection system using kcElectra and machine learning. (currently support youtube only)
imported from https://github.com/MisileLab/h3/commits/main/projects/dsb/vivian

## Versions

- [v0](https://static.marimo.app/static/vivian-jcxs)
- [v1](https://static.marimo.app/static/vivian-44de)
- [v1.1](https://static.marimo.app/static/nomorespam-zvfn)
  - Updated dataset to [v2](https://huggingface.co/datasets/MisileLab/youtube-bot-comments-v2)
    - Manual & regex classification
    - 50% user, 50% bot

## Overview

noMoreSpam is a tool designed to identify and filter bot comments on YouTube videos. It uses embeddings and machine learning techniques to classify comments as either bot-generated or human-written.

## Features

- YouTube comment collection and processing
- Bot comment classification using transformer-based models
- Interactive UI for model training and evaluation using Marimo notebooks
- Support for Korean text via KcELECTRA model
- Data visualization tools for model performance analysis

## Requirements

- Python 3.13.4 or higher
- YouTube API key for comment collection
- PyTorch (CPU or ROCm version available)
- OpenAI API key (for LLM-based classification)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/misilelab/noMoreSpam
   cd noMoreSpam
   ```

2. Set up the environment:
   ```
   uv sync
   ```

3. Set up your API keys as environment variables:
   ```
   export YOUTUBE_API_KEY=your_api_key_here
   export OPENAI_KEY=your_openai_api_key_here
   ```

## Usage

### Data Collection

1. Collect YouTube videos:
   ```
   python data/get_videos.py
   ```

2. Collect comments from videos:
   ```
   python main.py data/get_comments.py
   ```

### Classification

1. Run the ML-based classification model:
   ```
   python classify.py
   ```

2. Run the LLM-based classification model:
   ```
   python classify_llm.py
   ```

3. Evaluate comments interactively:
   ```
   python evaluate.py
   ```

4. Train the model with custom data:
   ```
   python train.py
   ```

### Data Processing

1. Split data into training and test sets:
   ```
   python data/train_test_split.py
   ```

2. Merge processed data:
   ```
   python merge.py
   ```

3. Clear temporary data:
   ```
   python clear.py
   ```

## Project Structure

- `classify.py`: Runs the ML-based bot comment classification model
- `classify_llm.py`: Runs the LLM-based bot comment classification using OpenAI models
- `clear.py`: Clears temporary OpenAI files
- `data/`: Directory containing data processing scripts
  - `get_comments.py`: Collects comments from YouTube videos
  - `get_videos.py`: Collects YouTube video information
  - `train_test_split.py`: Splits data into training and test sets
- `evaluate.py`: Interactive tool for evaluating comments with the trained model
- `main.py`: Entry point for running modules
- `merge.py`: Merges processed embedding data
- `train.py`: Trains the bot detection model
- `utils.py`: Utility functions and data models

## Model Architecture

The bot detection system uses a SpamUserClassifier based on the KcELECTRA model with:
- Frozen initial transformer layers
- Custom classification layers with dropout for regularization
- Focal Loss to handle class imbalance
- Combined CLS token and mean pooling for improved performance

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
